"""
CNN-NonStatic + NBSVM best-configuration, three-seed runner
======================================================

Google Colab:
    !pip install -q pandas pyarrow scikit-learn gensim matplotlib

Edit PATHS below, then run this file in Google Colab.

Protocol:
- Concatenate disjoint train/validation/test Parquet partitions.
- Stratified 10-fold outer CV, repeated independently for seeds 21, 42 and 100.
- Within each outer training fold: 85% model-fit / 15% calibration.
- Fit CNN and NBSVM on the same model-fit data.
- Calibrate each model's scores on the calibration subset.
- Average calibrated probabilities 50/50.
- Select threshold by calibration macro-F1.
- Evaluate once on the untouched outer test fold.

The NBSVM fold model/calibrated predictions are cached and reused across
CNN-only hyperparameter blocks. CNN fold results are checkpointed after
each fold and are skipped on restart when their config signature matches.
"""

# =========================
# 0. IMPORTS AND CONFIG
# =========================
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import json
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from gensim.models import KeyedVectors

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, Dropout, Dense, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adadelta


PATHS = {
    "train": "/content/drive/MyDrive/data/train.parquet",
    "validation": "/content/drive/MyDrive/data/validation.parquet",
    "test": "/content/drive/MyDrive/data/test.parquet",
    "word2vec": "/content/drive/MyDrive/data/GoogleNews-vectors-negative300.bin",
}

TEXT_COL = "text"
LABEL_COL = "label"

BASE_CONFIG = {
    "seed": 42,
    "max_vocab": 30000,
    "embedding_dim": 300,
    "filter_sizes": [3, 4, 5, 6],
    "num_filters": 50,
    "dropout": 0.3,
    "max_norm": 3.0,
    "batch_size": 50,
    "epochs": 15,
    "learning_rate": 1.0,
    "adadelta_rho": 0.95,
    "adadelta_epsilon": 1e-6,
    "nb_alpha": 1.0,
    "nb_C": 0.1,
    "nb_min_df": 2,
    "nb_max_features": 200000,
    "fit_fraction": 0.85,
    "n_splits": 10,
}

# Run the selected best CNN configuration across three experimental seeds.
# Kim (2014) does not report a reproducible CV random seed.
RUN_BLOCKS = ["seeds"]

EXPERIMENTS = {
    "seeds": [21,42],
}

OUTPUT_DIR = Path("/content/drive/MyDrive/results")
CACHE_DIR = OUTPUT_DIR / "_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1. REPRODUCIBILITY / DATA
# =========================
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def encode_labels(series):
    """Map negative -> 0 and positive -> 1."""
    if pd.api.types.is_numeric_dtype(series):
        values = set(series.dropna().unique())
        if values == {0, 1}:
            return series.astype(int).to_numpy()
        if values == {-1, 1}:
            return series.map({-1: 0, 1: 1}).astype(int).to_numpy()

    labels = series.astype(str).str.strip().str.lower()
    mapping = {
        "negative": 0, "positive": 1,
        "neg": 0, "pos": 1,
        "0": 0, "1": 1,
    }
    if not labels.isin(mapping).all():
        raise ValueError(
            "Labels must be 0/1, -1/1, negative/positive, or neg/pos. "
            f"Found: {sorted(labels.unique())}"
        )
    return labels.map(mapping).astype(int).to_numpy()


def load_all_data():
    frames = []
    for split in ("train", "validation", "test"):
        path = PATHS[split]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parquet file not found: {path}")
        part = pd.read_parquet(path)
        if TEXT_COL not in part or LABEL_COL not in part:
            raise ValueError(
                f"{path} must contain '{TEXT_COL}' and '{LABEL_COL}'."
            )
        part = part[[TEXT_COL, LABEL_COL]].copy()
        frames.append(part)

    # Only valid if these are disjoint partitions of the same corpus.
    df = pd.concat(frames, ignore_index=True)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    y = encode_labels(df[LABEL_COL])

    if len(np.unique(y)) != 2:
        raise ValueError("Expected exactly two sentiment classes.")

    print(f"Combined reviews: {len(df):,}")
    print("Class counts [negative, positive]:", np.bincount(y))
    return df[TEXT_COL].tolist(), y


# =========================
# 2. CALIBRATION / METRICS
# =========================
def calibrate_scores(
    calibration_scores, calibration_labels, eval_scores, random_state=42
):
    """
    Platt-style calibration using a one-dimensional logistic regression.
    Scores must come from models trained without the calibration examples.
    """
    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    calibrator.fit(
        np.asarray(calibration_scores).reshape(-1, 1),
        calibration_labels,
    )
    p_cal = calibrator.predict_proba(
        np.asarray(calibration_scores).reshape(-1, 1)
    )[:, 1]
    p_eval = calibrator.predict_proba(
        np.asarray(eval_scores).reshape(-1, 1)
    )[:, 1]
    return p_cal, p_eval, calibrator


def best_macro_f1_threshold(y_true, probabilities):
    """Search thresholds on calibration data only."""
    candidates = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 181),
        np.asarray(probabilities, dtype=float),
    ]))
    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc = np.nan
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "macro_f1": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "auc": auc,
    }


# =========================
# 3. NBSVM (CACHED PER FOLD)
# =========================
def nbsvm_fold_cache_path(fold, seed):
    # NBSVM folds/scores can differ by seed, so never share cache across seeds.
    return CACHE_DIR / f"nbsvm_seed_{seed}_fold_{fold:02d}.npz"


def fit_or_load_nbsvm_fold(
    fold, seed, X_fit_text, y_fit, X_cal_text, y_cal, X_test_text
):
    """
    Fit NBSVM only once per fold. Cache calibrated calibration/test
    probabilities, so CNN hyperparameter changes do not retrain NBSVM.
    """
    cache_path = nbsvm_fold_cache_path(fold, seed)
    if cache_path.exists():
        cached = np.load(cache_path)
        if (
            len(cached["y_cal"]) == len(y_cal)
            and np.array_equal(cached["y_cal"], y_cal)
        ):
            print(f"  Reusing cached NBSVM predictions for fold {fold}.")
            return (
                cached["nb_p_cal"],
                cached["nb_p_test"],
            )

    vectorizer = CountVectorizer(
        binary=True,
        ngram_range=(1, 2),
        min_df=BASE_CONFIG["nb_min_df"],
        max_features=BASE_CONFIG["nb_max_features"],
        dtype=np.float64,
    )
    X_fit = vectorizer.fit_transform(X_fit_text)
    X_cal = vectorizer.transform(X_cal_text)
    X_test = vectorizer.transform(X_test_text)

    pos_counts = np.asarray(X_fit[y_fit == 1].sum(axis=0)).ravel()
    neg_counts = np.asarray(X_fit[y_fit == 0].sum(axis=0)).ravel()
    alpha = BASE_CONFIG["nb_alpha"]

    pos_probs = pos_counts + alpha
    neg_probs = neg_counts + alpha
    pos_probs /= pos_probs.sum()
    neg_probs /= neg_probs.sum()
    log_ratio = np.log(pos_probs / neg_probs)

    X_fit_nb = X_fit.multiply(log_ratio).tocsr()
    X_cal_nb = X_cal.multiply(log_ratio).tocsr()
    X_test_nb = X_test.multiply(log_ratio).tocsr()

    svm = LinearSVC(
        C=BASE_CONFIG["nb_C"],
        max_iter=5000,
        random_state=seed,
    )
    svm.fit(X_fit_nb, y_fit)

    score_cal = svm.decision_function(X_cal_nb)
    score_test = svm.decision_function(X_test_nb)

    nb_p_cal, nb_p_test, _ = calibrate_scores(
        score_cal, y_cal, score_test, random_state=seed
    )

    np.savez_compressed(
        cache_path,
        y_cal=y_cal,
        nb_p_cal=nb_p_cal,
        nb_p_test=nb_p_test,
    )
    del vectorizer, X_fit, X_cal, X_test
    del X_fit_nb, X_cal_nb, X_test_nb, svm
    gc.collect()
    return nb_p_cal, nb_p_test


# =========================
# 4. CNN PREPARATION / MODEL
# =========================
class MaskedGlobalMaxPooling1D(tf.keras.layers.Layer):
    """Max pooling that ignores convolution windows containing padding."""

    def __init__(self, filter_size, **kwargs):
        super().__init__(**kwargs)
        self.filter_size = int(filter_size)

    def call(self, inputs):
        conv_output, token_ids = inputs
        lengths = tf.reduce_sum(
            tf.cast(token_ids != 0, tf.int32), axis=1
        )
        valid_lengths = tf.maximum(
            lengths - self.filter_size + 1, 0
        )
        mask = tf.sequence_mask(
            valid_lengths, maxlen=tf.shape(conv_output)[1]
        )
        mask = tf.expand_dims(mask, axis=-1)
        masked = tf.where(
            mask,
            conv_output,
            tf.cast(-1e9, conv_output.dtype),
        )
        pooled = tf.reduce_max(masked, axis=1)
        return tf.where(
            valid_lengths[:, None] > 0,
            pooled,
            tf.zeros_like(pooled),
        )

    def get_config(self):
        config = super().get_config()
        config.update({"filter_size": self.filter_size})
        return config


def prepare_cnn_fold(X_fit_text, X_cal_text, X_test_text, config):
    """
    Fit tokenizer on model-fit text only. Return padded sequences,
    tokenizer and the vocabulary size for this fold.
    """
    tokenizer = Tokenizer(
        num_words=config["max_vocab"],
        oov_token="<OOV>",
        lower=True,
    )
    tokenizer.fit_on_texts(X_fit_text)

    fit_seq = tokenizer.texts_to_sequences(X_fit_text)
    cal_seq = tokenizer.texts_to_sequences(X_cal_text)
    test_seq = tokenizer.texts_to_sequences(X_test_text)

    observed_max = max(
        [len(s) for s in fit_seq] + [max(config["filter_sizes"])]
    )
    max_len = max(observed_max, max(config["filter_sizes"]))

    def pad(sequences):
        return pad_sequences(
            sequences,
            maxlen=max_len,
            padding="post",
            truncating="post",
        )

    vocab_size = min(
        config["max_vocab"], len(tokenizer.word_index) + 1
    )
    return (
        pad(fit_seq), pad(cal_seq), pad(test_seq),
        tokenizer, vocab_size, max_len,
    )


def make_embedding_matrix(tokenizer, vocab_size, word_vectors, config):
    rng = np.random.default_rng(config["seed"])
    matrix = rng.uniform(
        -0.25, 0.25,
        size=(vocab_size, config["embedding_dim"]),
    ).astype(np.float32)
    matrix[0] = 0.0

    found = 0
    total = 0
    for word, index in tokenizer.word_index.items():
        if index >= vocab_size:
            continue
        total += 1
        if word in word_vectors:
            matrix[index] = word_vectors[word]
            found += 1

    print(
        f"  Word2Vec coverage: {found}/{max(total, 1)} "
        f"({found / max(total, 1):.1%})"
    )
    return matrix


def build_cnn(vocab_size, max_len, embedding_matrix, config):
    # The fold-specific seed is set at the start of run_configuration().

    inputs = Input(
        shape=(max_len,), dtype="int32", name="review_input"
    )
    embedded = Embedding(
        input_dim=vocab_size,
        output_dim=config["embedding_dim"],
        weights=[embedding_matrix],
        trainable=True,  # CNN-NonStatic
        name="word_embeddings",
    )(inputs)

    pooled_outputs = []
    for width in config["filter_sizes"]:
        conv = Conv1D(
            filters=config["num_filters"],
            kernel_size=width,
            activation="relu",
            padding="valid",
            kernel_constraint=max_norm(config["max_norm"], axis=[0, 1]),
            name=f"conv_{width}",
        )(embedded)

        pooled = MaskedGlobalMaxPooling1D(
            filter_size=width, name=f"pool_{width}"
        )([conv, inputs])
        pooled_outputs.append(pooled)

    merged = Concatenate(name="feature_concatenation")(pooled_outputs)
    dropped = Dropout(config["dropout"], name="dropout")(merged)
    outputs = Dense(
        1,
        activation="sigmoid",
        kernel_constraint=max_norm(config["max_norm"], axis=0),
        name="sentiment_output",
    )(dropped)

    model = Model(inputs=inputs, outputs=outputs, name="CNN_NonStatic")
    optimizer = Adadelta(
        learning_rate=config["learning_rate"],
        rho=config["adadelta_rho"],
        epsilon=config["adadelta_epsilon"],
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =========================
# 5. CHECKPOINTING / OUTPUT
# =========================
def config_signature(block_name, value, config):
    payload = {
        "block": block_name,
        "value": value,
        "config": config,
        "seed": BASE_CONFIG["seed"],
        "folds": BASE_CONFIG["n_splits"],
        "fit_fraction": BASE_CONFIG["fit_fraction"],
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def fold_csv_path(block_name):
    folder = OUTPUT_DIR / f"{list(EXPERIMENTS).index(block_name)+1:02d}_{block_name}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{block_name}_folds.csv"


def completed_folds(csv_path, signature):
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        rows = df[df["signature"] == signature]
        return set(rows["fold"].astype(int).tolist())
    except Exception:
        return set()


def append_checkpoint(csv_path, row):
    row_df = pd.DataFrame([row])
    if csv_path.exists():
        row_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(csv_path, index=False)


def summarize_and_plot(block_name, csv_path):
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    folder = csv_path.parent
    summary_rows = []
    for signature, group in df.groupby("signature"):
        row = {
            "signature": signature,
            "configuration": group["configuration"].iloc[0],
            "value": group["value"].iloc[0],
            "folds_completed": group["fold"].nunique(),
        }
        for metric in ("accuracy", "precision", "recall", "macro_f1", "auc"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(folder / f"{block_name}_summary.csv", index=False)

    # Plot mean macro-F1 and accuracy for each configuration.
    plot_df = summary.sort_values("value", key=lambda s: s.astype(str))
    labels = plot_df["value"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width/2, plot_df["accuracy_mean"], width,
        yerr=plot_df["accuracy_std"], capsize=4, label="Accuracy"
    )
    ax.bar(
        x + width/2, plot_df["macro_f1_mean"], width,
        yerr=plot_df["macro_f1_std"], capsize=4, label="Macro-F1"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(block_name)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"{block_name}: 10-fold ensemble comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / f"{block_name}_comparison.png", dpi=160)
    plt.close(fig)


# =========================
# 6. RUN ONE CONFIGURATION
# =========================
def run_configuration(
    block_name, value, X_all, y_all, word_vectors
):
    config = dict(BASE_CONFIG)
    if block_name == "seeds":
        # Run the baseline CNN configuration under each requested seed.
        config["seed"] = int(value)
    elif block_name == "filter_sizes":
        config[block_name] = list(value)
    else:
        config[block_name] = value

    signature = config_signature(block_name, value, config)
    csv_path = fold_csv_path(block_name)
    done = completed_folds(csv_path, signature)

    print("\n" + "=" * 72)
    print(f"BLOCK: {block_name} | VALUE: {value}")
    print(f"CONFIG SIGNATURE: {signature}")
    print(f"Completed folds already saved: {sorted(done)}")
    print("=" * 72)

    cv = StratifiedKFold(
        n_splits=config["n_splits"],
        shuffle=True,
        random_state=config["seed"],
    )

    for fold, (outer_train_idx, outer_test_idx) in enumerate(
        cv.split(np.zeros(len(y_all)), y_all), start=1
    ):
        if fold in done:
            print(f"Fold {fold}/10: already complete, skipping.")
            continue

        print(f"\n--- {block_name}={value} | Fold {fold}/10 ---")
        set_seed(config["seed"] + fold)

        # Outer test remains untouched until final evaluation.
        X_outer_train = [X_all[i] for i in outer_train_idx]
        y_outer_train = y_all[outer_train_idx]
        X_test_text = [X_all[i] for i in outer_test_idx]
        y_test = y_all[outer_test_idx]

        fit_idx, cal_idx = train_test_split(
            np.arange(len(y_outer_train)),
            train_size=config["fit_fraction"],
            stratify=y_outer_train,
            random_state=config["seed"] + fold,
        )

        X_fit_text = [X_outer_train[i] for i in fit_idx]
        y_fit = y_outer_train[fit_idx]
        X_cal_text = [X_outer_train[i] for i in cal_idx]
        y_cal = y_outer_train[cal_idx]

        # NBSVM predictions are independent of CNN hyperparameters.
        nb_p_cal, nb_p_test = fit_or_load_nbsvm_fold(
            fold,
            config["seed"],
            X_fit_text, y_fit,
            X_cal_text, y_cal,
            X_test_text,
        )

        # Fold-specific tokenizer and padded sequences.
        X_fit, X_cal, X_test, tokenizer, vocab_size, max_len = (
            prepare_cnn_fold(
                X_fit_text, X_cal_text, X_test_text, config
            )
        )

        embedding_matrix = make_embedding_matrix(
            tokenizer, vocab_size, word_vectors, config
        )

        tf.keras.backend.clear_session()
        gc.collect()

        cnn = build_cnn(
            vocab_size, max_len, embedding_matrix, config
        )
        cnn.fit(
            X_fit,
            y_fit.astype(np.float32),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            shuffle=True,
            verbose=0,
        )

        cnn_score_cal = cnn.predict(
            X_cal, batch_size=config["batch_size"], verbose=0
        ).ravel()
        cnn_score_test = cnn.predict(
            X_test, batch_size=config["batch_size"], verbose=0
        ).ravel()

        cnn_p_cal, cnn_p_test, _ = calibrate_scores(
            cnn_score_cal, y_cal, cnn_score_test,
            random_state=config["seed"]
        )

        # Equal-weight calibrated ensemble.
        ensemble_p_cal = 0.5 * cnn_p_cal + 0.5 * nb_p_cal
        ensemble_p_test = 0.5 * cnn_p_test + 0.5 * nb_p_test
        # ==========================================
        # SAVE INDIVIDUAL OUT-OF-FOLD PREDICTIONS
        # ==========================================

        # MAP THESE TO YOUR EXISTING VARIABLES:
        # FOLD                 -> current fold number
        # TEST_INDICES         -> original row indices of the held-out fold
        # Y_TEST               -> true labels of the held-out fold
        # ENSEMBLE_PROB        -> calibrated ensemble P(positive)
        # SELECTED_THRESHOLD   -> threshold selected on calibration data
        # ALL_REVIEWS          -> complete review-text array (optional)

        prediction_dir = os.path.join(
        OUTPUT_DIR,
        "05_seeds",
        "predictions"
        )

        os.makedirs(prediction_dir, exist_ok=True)

        # Convert probabilities and labels to NumPy arrays
        p_positive = np.asarray(ensemble_p_test).reshape(-1)
        y_true = np.asarray(y_test).reshape(-1)
        test_indices = np.asarray(outer_test_idx).reshape(-1)

        # Apply the threshold selected using calibration data
        y_pred = (p_positive >= threshold).astype(int)

        # Confidence = probability assigned to the predicted class
        confidence = np.where(
        y_pred == 1,
        p_positive,
        1 - p_positive
        )

        fold_predictions = pd.DataFrame({
        "sample_id": test_indices,
        "seed": config["seed"],
        "fold": fold,
        "true_label": y_true,
        "predicted_label": y_pred,
        "positive_probability": p_positive,
        "confidence": confidence,
        "threshold": threshold,
        })

        # Optional: include original review text
        fold_predictions["review"] = np.asarray(X_all)[test_indices]

        # Save one file per seed and fold
        prediction_file = os.path.join(
        prediction_dir,
        f"seed_{config["seed"]}_fold_{fold}.csv"
        )

        fold_predictions.to_csv(prediction_file, index=False)

        print(f"Saved predictions: {prediction_file}")

        threshold, calibration_macro_f1 = best_macro_f1_threshold(
            y_cal, ensemble_p_cal
        )
        metrics = calculate_metrics(
            y_test, ensemble_p_test, threshold
        )

        row = {
            "block": block_name,
            "seed": config["seed"],
            "value": json.dumps(value),
            "configuration": json.dumps(config, sort_keys=True),
            "signature": signature,
            "fold": fold,
            "threshold": threshold,
            "calibration_macro_f1": calibration_macro_f1,
            **metrics,
        }
        append_checkpoint(csv_path, row)

        print(
            f"Fold {fold}: acc={metrics['accuracy']:.4f}, "
            f"macro-F1={metrics['macro_f1']:.4f}, "
            f"AUC={metrics['auc']:.4f}, threshold={threshold:.3f}"
        )

        # Free GPU/RAM between configurations/folds.
        del cnn, embedding_matrix, tokenizer
        del X_fit, X_cal, X_test
        del cnn_score_cal, cnn_score_test, cnn_p_cal, cnn_p_test
        del ensemble_p_cal, ensemble_p_test
        tf.keras.backend.clear_session()
        gc.collect()

    summarize_and_plot(block_name, csv_path)


# =========================
# 7. MAIN
# =========================
def main():
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("\\nFixed CNN configuration:")
    for key in (
        "filter_sizes", "num_filters", "dropout", "epochs",
        "batch_size", "embedding_dim", "learning_rate",
    ):
        print(f"  {key}: {BASE_CONFIG[key]}")
    print("Seeds:", EXPERIMENTS["seeds"])

    X_all, y_all = load_all_data()

    w2v_path = PATHS["word2vec"]
    if not os.path.exists(w2v_path):
        raise FileNotFoundError(
            f"Google News Word2Vec file not found: {w2v_path}"
        )

    print("\nLoading Google News Word2Vec vectors once...")
    word_vectors = KeyedVectors.load_word2vec_format(
        w2v_path, binary=True
    )
    if word_vectors.vector_size != BASE_CONFIG["embedding_dim"]:
        raise ValueError(
            f"Expected {BASE_CONFIG['embedding_dim']}D Word2Vec, "
            f"found {word_vectors.vector_size}D."
        )

    for block_name in RUN_BLOCKS:
        if block_name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment block: {block_name}")

        for value in EXPERIMENTS[block_name]:
            run_configuration(
                block_name, value, X_all, y_all, word_vectors
            )

    print("\nAll selected experiment blocks have finished.")
    print("Results directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
