"""
CNN–NBSVM sentiment experiment with prediction-level output and safe resume.

Protocol:
- Combine train/validation/test Parquet files (10-fold CV over all 10,662 rows).
- For each seed, create stratified 10-fold outer CV.
- Split each outer training fold into 85% model-fit / 15% calibration.
- Train CNN-NonStatic and NBSVM on model-fit data only.
- Fit Platt-style logistic calibrators on calibration scores.
- Average calibrated probabilities 50/50.
- Select the ensemble threshold by calibration macro-F1.
- Evaluate once on the untouched outer test fold.
- Save fold predictions, calibration predictions, fold metrics, and summaries.

IMPORTANT:
This is a new, self-contained implementation of the documented protocol and
configuration. It is not a byte-for-byte patch of the prior .py file, which
was not attached in the current conversation.
"""

import os
os.environ.setdefault("PYTHONHASHSEED", "0")

import gc
import glob
import hashlib
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from gensim.models import KeyedVectors

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, Dropout, Dense, Concatenate, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# 1. PATHS — update only if your Google Drive structure differs
# ============================================================

PATHS = {
    "train": "/content/drive/MyDrive/data/train.parquet",
    "validation": "/content/drive/MyDrive/data/validation.parquet",
    "test": "/content/drive/MyDrive/data/test.parquet",
    "word2vec": "/content/drive/MyDrive/data/GoogleNews-vectors-negative300.bin",
}

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# The output directory from your latest run.
OUTPUT_DIR = Path("/content/drive/MyDrive/results/results_movie_review")

# Prediction files are kept in a dedicated subfolder so old metric-only
# checkpoints cannot cause this script to skip prediction generation.
PREDICTION_DIR = OUTPUT_DIR / "predictions"
FOLD_METRICS_DIR = OUTPUT_DIR / "fold_metrics"
CALIBRATION_DIR = OUTPUT_DIR / "calibration_predictions"


# ============================================================
# 2. EXPERIMENT CONFIGURATION — matches the latest log provided
# ============================================================

CONFIG = {
    "n_splits": 10,
    "seeds": [21, 42],

    "filter_sizes": [3, 4, 5, 6],
    "num_filters": 50,
    "dropout": 0.3,
    "epochs": 15,
    "batch_size": 50,
    "embedding_dim": 300,
    "learning_rate": 1.0,

    "max_vocab": 30000,
    "max_length": None,  # determine from each outer training fold
    "max_norm": 3.0,

    "adadelta_rho": 0.95,
    "adadelta_epsilon": 1e-6,

    # NBSVM
    "nb_alpha": 1.0,
    "nb_C": 0.1,
    "nb_max_iter": 5000,
    "nb_min_df": 2,
    "nb_max_features": 200000,
    "nb_ngram_range": [1, 2],

    # Ensemble and calibration
    "cnn_weight": 0.5,
    "nbsvm_weight": 0.5,
    "calibration_fraction": 0.15,
    "threshold_min": 0.05,
    "threshold_max": 0.95,
    "threshold_steps": 181,

    # Reproducibility
    "deterministic_ops": True,
}


# ============================================================
# 3. GENERAL UTILITIES
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if CONFIG["deterministic_ops"]:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def atomic_write_csv(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(destination)


def encode_labels(series: pd.Series) -> np.ndarray:
    """Encode negative=0 and positive=1, accepting common label formats."""
    if pd.api.types.is_numeric_dtype(series):
        values = set(pd.Series(series).dropna().unique().tolist())
        if values.issubset({0, 1}) and values:
            return series.astype(int).to_numpy()
        if values.issubset({-1, 1}) and values:
            return series.map({-1: 0, 1: 1}).astype(int).to_numpy()

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "negative": 0, "positive": 1,
        "neg": 0, "pos": 1,
        "0": 0, "1": 1,
        "-1": 0,
    }
    unknown = sorted(set(normalized.unique()) - set(mapping))
    if unknown:
        raise ValueError(
            "Unrecognized label values: "
            f"{unknown[:20]}. Expected negative/positive or 0/1."
        )
    return normalized.map(mapping).astype(int).to_numpy()


def load_dataset():
    for key, path in PATHS.items():
        if key != "word2vec" and not Path(path).exists():
            raise FileNotFoundError(f"Parquet file not found ({key}): {path}")

    train_df = pd.read_parquet(PATHS["train"])
    val_df = pd.read_parquet(PATHS["validation"])
    test_df = pd.read_parquet(PATHS["test"])

    for name, frame in [
        ("train", train_df), ("validation", val_df), ("test", test_df)
    ]:
        missing = {TEXT_COLUMN, LABEL_COLUMN} - set(frame.columns)
        if missing:
            raise KeyError(
                f"{name} file is missing columns {sorted(missing)}. "
                f"Available columns: {frame.columns.tolist()}"
            )

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    df = df.reset_index(drop=True)

    texts = df[TEXT_COLUMN].tolist()
    labels = encode_labels(df[LABEL_COLUMN])

    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"Expected both classes 0 and 1; got {np.unique(labels)}")

    df.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    df["true_label"] = labels

    print(f"Combined reviews: {len(df):,}")
    print(
        "Class counts [negative, positive]:",
        np.bincount(labels, minlength=2)
    )
    return df, texts, labels


def config_signature(texts, labels, seed):
    payload = {
        "config": CONFIG,
        "seed": int(seed),
        "text_column": TEXT_COLUMN,
        "label_column": LABEL_COLUMN,
        "dataset_rows": len(texts),
    }
    config_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(config_bytes)
    # Include a deterministic data fingerprint so changed data cannot reuse
    # old fold predictions accidentally.
    row_hashes = pd.util.hash_pandas_object(
        pd.Series(texts, dtype="object"), index=False
    ).to_numpy(dtype=np.uint64)
    digest.update(row_hashes.tobytes())
    digest.update(np.asarray(labels, dtype=np.int8).tobytes())
    return digest.hexdigest()[:16]


def load_word2vec():
    path = Path(PATHS["word2vec"])
    if not path.exists():
        raise FileNotFoundError(
            "Google News Word2Vec file not found:\n"
            f"{path}\n"
            "Check PATHS['word2vec'] in the configuration."
        )
    print("\nLoading Google News Word2Vec vectors once...")
    vectors = KeyedVectors.load_word2vec_format(str(path), binary=True)
    if vectors.vector_size != CONFIG["embedding_dim"]:
        raise ValueError(
            f"Expected {CONFIG['embedding_dim']}D vectors; "
            f"found {vectors.vector_size}D."
        )
    print("Word2Vec vocabulary:", len(vectors))
    return vectors


# ============================================================
# 4. CNN — Kim-style CNN-NonStatic with pretrained Word2Vec
# ============================================================

class MaskedGlobalMaxPooling1D(Layer):
    """Max-pool only over convolution windows containing real tokens."""

    def __init__(self, filter_size, **kwargs):
        super().__init__(**kwargs)
        self.filter_size = int(filter_size)

    def call(self, inputs):
        conv_output, token_ids = inputs
        lengths = tf.reduce_sum(
            tf.cast(tf.not_equal(token_ids, 0), tf.int32), axis=1
        )
        valid_lengths = tf.maximum(lengths - self.filter_size + 1, 0)
        mask = tf.sequence_mask(
            valid_lengths, maxlen=tf.shape(conv_output)[1]
        )
        mask = tf.expand_dims(mask, axis=-1)

        masked = tf.where(
            mask,
            conv_output,
            tf.cast(-1e9, conv_output.dtype)
        )
        pooled = tf.reduce_max(masked, axis=1)
        return tf.where(
            valid_lengths[:, None] > 0,
            pooled,
            tf.zeros_like(pooled)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"filter_size": self.filter_size})
        return config


def create_embedding_matrix(tokenizer, vocab_size, word_vectors, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(
        -0.25, 0.25,
        size=(vocab_size, CONFIG["embedding_dim"])
    ).astype(np.float32)
    matrix[0] = 0.0

    found = 0
    considered = 0
    for word, index in tokenizer.word_index.items():
        if index >= vocab_size:
            continue
        considered += 1

        # Google News vectors are case-sensitive; try common variants.
        for candidate in (word, word.capitalize(), word.upper()):
            if candidate in word_vectors:
                matrix[index] = word_vectors[candidate]
                found += 1
                break

    coverage = found / max(considered, 1)
    print(
        f"Word2Vec coverage: {found:,}/{considered:,} "
        f"({coverage:.2%})"
    )
    return matrix, found, considered


def build_cnn_nonstatic(vocab_size, max_length, embedding_matrix, seed):
    set_seed(seed)

    inputs = Input(shape=(max_length,), dtype="int32", name="review_input")
    embedded = Embedding(
        input_dim=vocab_size,
        output_dim=CONFIG["embedding_dim"],
        weights=[embedding_matrix],
        trainable=True,
        name="word_embeddings",
    )(inputs)

    pooled_outputs = []
    for width in CONFIG["filter_sizes"]:
        conv = Conv1D(
            filters=CONFIG["num_filters"],
            kernel_size=width,
            activation="relu",
            padding="valid",
            kernel_constraint=max_norm(CONFIG["max_norm"], axis=[0, 1]),
            name=f"conv_{width}",
        )(embedded)
        pooled = MaskedGlobalMaxPooling1D(
            filter_size=width, name=f"masked_pool_{width}"
        )([conv, inputs])
        pooled_outputs.append(pooled)

    merged = Concatenate(name="feature_concatenation")(pooled_outputs)
    dropped = Dropout(CONFIG["dropout"], name="dropout")(merged)
    output = Dense(
        1,
        activation="sigmoid",
        kernel_constraint=max_norm(CONFIG["max_norm"], axis=0),
        name="sentiment_output",
    )(dropped)

    model = Model(inputs=inputs, outputs=output, name="CNN_NonStatic")
    optimizer = Adadelta(
        learning_rate=CONFIG["learning_rate"],
        rho=CONFIG["adadelta_rho"],
        epsilon=CONFIG["adadelta_epsilon"],
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def prepare_cnn_data(train_text, calibration_text, test_text, seed, word_vectors):
    tokenizer = Tokenizer(
        num_words=CONFIG["max_vocab"],
        oov_token="<OOV>",
        lower=True,
    )
    tokenizer.fit_on_texts(train_text)

    train_seq = tokenizer.texts_to_sequences(train_text)
    cal_seq = tokenizer.texts_to_sequences(calibration_text)
    test_seq = tokenizer.texts_to_sequences(test_text)

    if CONFIG["max_length"] is None:
        max_length = max(
            max(len(seq) for seq in train_seq),
            max(CONFIG["filter_sizes"]),
            5,
        )
    else:
        max_length = max(
            int(CONFIG["max_length"]),
            max(CONFIG["filter_sizes"]),
            5,
        )

    pad_args = {
        "maxlen": max_length,
        "padding": "post",
        "truncating": "post",
    }
    X_fit = pad_sequences(train_seq, **pad_args)
    X_cal = pad_sequences(cal_seq, **pad_args)
    X_test = pad_sequences(test_seq, **pad_args)

    vocab_size = min(
        CONFIG["max_vocab"], len(tokenizer.word_index) + 1
    )
    matrix, found, considered = create_embedding_matrix(
        tokenizer, vocab_size, word_vectors, seed
    )

    return (
        X_fit, X_cal, X_test, vocab_size, max_length,
        matrix, found, considered
    )


# ============================================================
# 5. NBSVM — binary unigram/bigram features + LinearSVC
# ============================================================

def train_nbsvm_predict(train_text, y_fit, calibration_text, test_text, seed):
    vectorizer = CountVectorizer(
        binary=True,
        ngram_range=tuple(CONFIG["nb_ngram_range"]),
        min_df=CONFIG["nb_min_df"],
        max_features=CONFIG["nb_max_features"],
        dtype=np.float64,
    )

    X_fit = vectorizer.fit_transform(train_text)
    X_cal = vectorizer.transform(calibration_text)
    X_test = vectorizer.transform(test_text)

    positive_counts = np.asarray(X_fit[y_fit == 1].sum(axis=0)).ravel()
    negative_counts = np.asarray(X_fit[y_fit == 0].sum(axis=0)).ravel()

    alpha = CONFIG["nb_alpha"]
    positive_probs = positive_counts + alpha
    negative_probs = negative_counts + alpha
    positive_probs /= positive_probs.sum()
    negative_probs /= negative_probs.sum()

    log_count_ratio = np.log(positive_probs / negative_probs)

    X_fit_nb = X_fit.multiply(log_count_ratio).tocsr()
    X_cal_nb = X_cal.multiply(log_count_ratio).tocsr()
    X_test_nb = X_test.multiply(log_count_ratio).tocsr()

    classifier = LinearSVC(
        C=CONFIG["nb_C"],
        max_iter=CONFIG["nb_max_iter"],
        random_state=seed,
    )
    classifier.fit(X_fit_nb, y_fit)

    cal_scores = classifier.decision_function(X_cal_nb)
    test_scores = classifier.decision_function(X_test_nb)

    return cal_scores, test_scores, vectorizer, classifier


# ============================================================
# 6. Calibration and threshold selection
# ============================================================

def safe_logit(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1.0 - p))


def fit_score_calibrator(calibration_scores, y_cal, seed):
    """
    Platt-style calibration: logistic regression maps one-dimensional
    decision scores to calibrated positive-class probabilities.
    """
    scores = np.asarray(calibration_scores, dtype=float).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
    calibrator.fit(scores, y_cal)
    return calibrator


def calibrated_positive_probability(calibrator, scores):
    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    positive_column = list(calibrator.classes_).index(1)
    return calibrator.predict_proba(scores)[:, positive_column]


def best_macro_f1_threshold(y_true, probabilities):
    thresholds = np.linspace(
        CONFIG["threshold_min"],
        CONFIG["threshold_max"],
        CONFIG["threshold_steps"],
    )
    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(
            y_true, predictions, average="macro", zero_division=0
        )
        # Deterministic tie-break: prefer the threshold nearest 0.5.
        if (
            score > best_score
            or (
                np.isclose(score, best_score)
                and abs(threshold - 0.5) < abs(best_threshold - 0.5)
            )
        ):
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, float(best_score)


# ============================================================
# 7. Prediction and metric tables
# ============================================================

def make_prediction_frame(
    source_df,
    indices,
    y_true,
    cnn_raw_probability,
    nb_raw_score,
    cnn_cal_probability,
    nb_cal_probability,
    ensemble_probability,
    threshold,
    seed,
    fold,
    signature,
    split_name,
):
    predicted = (ensemble_probability >= threshold).astype(int)
    confidence = np.where(
        predicted == 1,
        ensemble_probability,
        1.0 - ensemble_probability,
    )

    result = pd.DataFrame({
        "sample_id": source_df.iloc[indices]["sample_id"].to_numpy(),
        "seed": seed,
        "fold": fold,
        "split": split_name,
        "text": source_df.iloc[indices][TEXT_COLUMN].to_numpy(),
        "true_label": np.asarray(y_true, dtype=int),
        "true_sentiment": np.where(np.asarray(y_true) == 1, "positive", "negative"),
        "cnn_raw_positive_probability": np.asarray(cnn_raw_probability),
        "nbsvm_raw_decision_score": np.asarray(nb_raw_score),
        "cnn_calibrated_positive_probability": np.asarray(cnn_cal_probability),
        "nbsvm_calibrated_positive_probability": np.asarray(nb_cal_probability),
        "ensemble_positive_probability": np.asarray(ensemble_probability),
        "decision_threshold": float(threshold),
        "predicted_label": predicted,
        "predicted_sentiment": np.where(predicted == 1, "positive", "negative"),
        "confidence": confidence,
        "correct": predicted == np.asarray(y_true, dtype=int),
        "config_signature": signature,
    })
    return result


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision_positive": precision_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "recall_positive": recall_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "f1_positive": f1_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "precision_negative": precision_score(
            y_true, predictions, pos_label=0, zero_division=0
        ),
        "recall_negative": recall_score(
            y_true, predictions, pos_label=0, zero_division=0
        ),
        "f1_negative": f1_score(
            y_true, predictions, pos_label=0, zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "tn": int(confusion_matrix(y_true, predictions, labels=[0, 1])[0, 0]),
        "fp": int(confusion_matrix(y_true, predictions, labels=[0, 1])[0, 1]),
        "fn": int(confusion_matrix(y_true, predictions, labels=[0, 1])[1, 0]),
        "tp": int(confusion_matrix(y_true, predictions, labels=[0, 1])[1, 1]),
    }


# ============================================================
# 8. One outer fold
# ============================================================

def run_one_fold(
    source_df, texts, labels, train_indices, test_indices,
    seed, fold_number, signature, word_vectors
):
    print("\n" + "=" * 72)
    print(f"SEED {seed} | FOLD {fold_number}/10")
    print("=" * 72)

    # The calibration set is carved only from the outer training fold.
    fit_indices, cal_indices = train_test_split(
        train_indices,
        test_size=CONFIG["calibration_fraction"],
        random_state=seed * 1000 + fold_number,
        stratify=labels[train_indices],
    )

    fit_text = [texts[i] for i in fit_indices]
    cal_text = [texts[i] for i in cal_indices]
    test_text = [texts[i] for i in test_indices]

    y_fit = labels[fit_indices]
    y_cal = labels[cal_indices]
    y_test = labels[test_indices]

    print(
        f"Fit={len(fit_indices):,} | Calibration={len(cal_indices):,} "
        f"| Outer test={len(test_indices):,}"
    )

    # ---------------- CNN-NonStatic ----------------
    set_seed(seed * 1000 + fold_number)

    (
        X_fit, X_cal, X_test, vocab_size, max_length,
        embedding_matrix, w2v_found, w2v_total
    ) = prepare_cnn_data(
        fit_text, cal_text, test_text,
        seed * 1000 + fold_number, word_vectors
    )

    cnn = build_cnn_nonstatic(
        vocab_size, max_length, embedding_matrix,
        seed * 1000 + fold_number
    )

    history = cnn.fit(
        X_fit,
        y_fit,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        verbose=1,
    )

    cnn_cal_raw = cnn.predict(
        X_cal, batch_size=CONFIG["batch_size"], verbose=0
    ).ravel()
    cnn_test_raw = cnn.predict(
        X_test, batch_size=CONFIG["batch_size"], verbose=0
    ).ravel()

    # ---------------- NBSVM ----------------
    nb_cal_scores, nb_test_scores, nb_vectorizer, nb_classifier = (
        train_nbsvm_predict(
            fit_text, y_fit, cal_text, test_text,
            seed * 1000 + fold_number
        )
    )

    # ---------------- Calibrate both models ----------------
    cnn_calibrator = fit_score_calibrator(
        safe_logit(cnn_cal_raw), y_cal, seed * 1000 + fold_number
    )
    nb_calibrator = fit_score_calibrator(
        nb_cal_scores, y_cal, seed * 1000 + fold_number + 1
    )

    cnn_p_cal = calibrated_positive_probability(
        cnn_calibrator, safe_logit(cnn_cal_raw)
    )
    cnn_p_test = calibrated_positive_probability(
        cnn_calibrator, safe_logit(cnn_test_raw)
    )

    nb_p_cal = calibrated_positive_probability(
        nb_calibrator, nb_cal_scores
    )
    nb_p_test = calibrated_positive_probability(
        nb_calibrator, nb_test_scores
    )

    ensemble_p_cal = (
        CONFIG["cnn_weight"] * cnn_p_cal
        + CONFIG["nbsvm_weight"] * nb_p_cal
    )
    ensemble_p_test = (
        CONFIG["cnn_weight"] * cnn_p_test
        + CONFIG["nbsvm_weight"] * nb_p_test
    )

    threshold, calibration_macro_f1 = best_macro_f1_threshold(
        y_cal, ensemble_p_cal
    )

    # ---------------- Save calibration predictions ----------------
    cal_frame = make_prediction_frame(
        source_df, cal_indices, y_cal,
        cnn_cal_raw, nb_cal_scores,
        cnn_p_cal, nb_p_cal, ensemble_p_cal,
        threshold, seed, fold_number, signature, "calibration",
    )
    cal_path = CALIBRATION_DIR / f"seed_{seed}_fold_{fold_number:02d}_calibration.csv"
    atomic_write_csv(cal_frame, cal_path)

    # ---------------- Save outer-test predictions ----------------
    test_frame = make_prediction_frame(
        source_df, test_indices, y_test,
        cnn_test_raw, nb_test_scores,
        cnn_p_test, nb_p_test, ensemble_p_test,
        threshold, seed, fold_number, signature, "outer_test",
    )
    prediction_path = (
        PREDICTION_DIR / f"seed_{seed}_fold_{fold_number:02d}_predictions.csv"
    )
    atomic_write_csv(test_frame, prediction_path)

    # Save per-fold metrics and diagnostic information.
    metrics = calculate_metrics(y_test, ensemble_p_test, threshold)
    fold_record = {
        "seed": seed,
        "fold": fold_number,
        "config_signature": signature,
        "n_fit": len(fit_indices),
        "n_calibration": len(cal_indices),
        "n_outer_test": len(test_indices),
        "threshold": threshold,
        "calibration_macro_f1": calibration_macro_f1,
        "cnn_epochs_trained": len(history.history["loss"]),
        "cnn_final_training_loss": float(history.history["loss"][-1]),
        "word2vec_found": w2v_found,
        "word2vec_considered": w2v_total,
        "word2vec_coverage": w2v_found / max(w2v_total, 1),
        "prediction_file": str(prediction_path),
        **metrics,
    }

    metrics_path = (
        FOLD_METRICS_DIR / f"seed_{seed}_fold_{fold_number:02d}_metrics.csv"
    )
    atomic_write_csv(pd.DataFrame([fold_record]), metrics_path)

    print("\nOuter-fold ensemble results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"Threshold selected on calibration data: {threshold:.3f}")
    print(f"Saved predictions: {prediction_path}")

    # Free fold-specific objects before moving to the next fold.
    del cnn, nb_vectorizer, nb_classifier
    del X_fit, X_cal, X_test, embedding_matrix
    del history
    tf.keras.backend.clear_session()
    gc.collect()

    return fold_record


# ============================================================
# 9. Safe resume and aggregation
# ============================================================

def valid_existing_prediction(path, signature):
    if not path.exists():
        return False
    try:
        existing = pd.read_csv(path, usecols=["config_signature"])
        return (
            len(existing) > 0
            and existing["config_signature"].astype(str).eq(signature).all()
        )
    except Exception:
        return False


def collect_fold_metrics():
    files = sorted(FOLD_METRICS_DIR.glob("seed_*_fold_*_metrics.csv"))
    frames = []
    for file in files:
        try:
            frames.append(pd.read_csv(file))
        except Exception as exc:
            warnings.warn(f"Could not read {file}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def rebuild_combined_predictions():
    files = sorted(PREDICTION_DIR.glob("seed_*_fold_*_predictions.csv"))
    frames = []
    for file in files:
        try:
            frames.append(pd.read_csv(file))
        except Exception as exc:
            warnings.warn(f"Could not read prediction file {file}: {exc}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(
            ["seed", "sample_id"]
        ).reset_index(drop=True)
        atomic_write_csv(
            combined, OUTPUT_DIR / "all_outer_test_predictions.csv"
        )

        # One row per sample per seed is expected: every sample is an outer
        # test example exactly once for each seed.
        counts = combined.groupby("seed")["sample_id"].nunique()
        print("\nUnique outer-test samples saved per seed:")
        print(counts.to_string())

        # Convenience export: most confident incorrect predictions first.
        mistakes = combined.loc[~combined["correct"].astype(bool)].copy()
        mistakes = mistakes.sort_values(
            "confidence", ascending=False
        )
        atomic_write_csv(
            mistakes, OUTPUT_DIR / "misclassified_predictions.csv"
        )

        print(
            "\nCombined prediction file:",
            OUTPUT_DIR / "all_outer_test_predictions.csv"
        )
        print(
            "Misclassified-review file:",
            OUTPUT_DIR / "misclassified_predictions.csv"
        )
    else:
        print("No fold prediction files were found to combine.")


def write_summary():
    metrics = collect_fold_metrics()
    if metrics.empty:
        print("No fold metrics available yet.")
        return

    atomic_write_csv(metrics, OUTPUT_DIR / "all_fold_metrics.csv")

    metric_columns = [
        "accuracy", "macro_f1", "roc_auc",
        "precision_positive", "recall_positive", "f1_positive",
        "precision_negative", "recall_negative", "f1_negative",
    ]
    summary_rows = []
    for seed, group in metrics.groupby("seed"):
        row = {
            "seed": int(seed),
            "completed_folds": int(group["fold"].nunique()),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_std"] = group[column].std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    atomic_write_csv(summary, OUTPUT_DIR / "seed_summary.csv")
    print("\nSeed-level summary:")
    print(summary.to_string(index=False))


# ============================================================
# 10. Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("\nFixed CNN configuration:")
    for key in [
        "filter_sizes", "num_filters", "dropout", "epochs",
        "batch_size", "embedding_dim", "learning_rate"
    ]:
        print(f"  {key}: {CONFIG[key]}")
    print("Seeds:", CONFIG["seeds"])

    source_df, texts, labels = load_dataset()
    word_vectors = load_word2vec()

    splitter_cache = {}
    for seed in CONFIG["seeds"]:
        signature = config_signature(texts, labels, seed)
        print("\n" + "=" * 72)
        print(f"SEED: {seed}")
        print("CONFIG SIGNATURE:", signature)
        print("=" * 72)

        splitter = StratifiedKFold(
            n_splits=CONFIG["n_splits"],
            shuffle=True,
            random_state=seed,
        )

        completed = []
        for fold_number, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(labels)), labels), start=1
        ):
            prediction_path = (
                PREDICTION_DIR
                / f"seed_{seed}_fold_{fold_number:02d}_predictions.csv"
            )
            metrics_path = (
                FOLD_METRICS_DIR
                / f"seed_{seed}_fold_{fold_number:02d}_metrics.csv"
            )

            # Crucial: a fold is skipped only if its prediction file exists
            # and matches this exact data/config signature, AND its metrics
            # checkpoint exists. Old metrics-only checkpoints are insufficient.
            if (
                valid_existing_prediction(prediction_path, signature)
                and metrics_path.exists()
            ):
                print(
                    f"Fold {fold_number}/10: prediction file already exists, "
                    "skipping."
                )
                completed.append(fold_number)
                continue

            print(f"Fold {fold_number}/10: running.")
            run_one_fold(
                source_df, texts, labels,
                train_idx, test_idx,
                seed, fold_number, signature, word_vectors,
            )

        print("Completed prediction folds:", completed if completed else "newly generated")

    # Rebuild global artifacts from all individual fold files.
    rebuild_combined_predictions()
    write_summary()

    # Save a readable configuration snapshot.
    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": CONFIG,
                "paths": PATHS,
                "text_column": TEXT_COLUMN,
                "label_column": LABEL_COLUMN,
                "protocol": (
                    "Stratified 10-fold outer CV; 85/15 fit/calibration "
                    "split within outer training fold; calibrated 50/50 "
                    "CNN-NonStatic/NBSVM ensemble; threshold selected on "
                    "calibration predictions."
                ),
            },
            f,
            indent=2,
        )

    print("\nAll selected experiment blocks have finished.")
    print("Results directory:", OUTPUT_DIR)
    print("Predictions directory:", PREDICTION_DIR)


if __name__ == "__main__":
    main()
