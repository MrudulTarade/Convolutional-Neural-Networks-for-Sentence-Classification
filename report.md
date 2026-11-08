# Reproducing and Extending CNN-Based Sentiment Classification: Model Optimization, Cross-Domain Evaluation, and Confidence Estimation

**Deep Learning · M.Sc. Applied Data Science and AI · SRH University Munich · Winter semester 2026/27**
**Author(s):** Ashok Rai & Mrudul Tarade · **Repository:** (https://github.com/MrudulTarade/Convolutional-Neural-Networks-for-Sentence-Classification) · **Date:** 22-10-2026

---

## Abstract

[Four or five sentences: the task, the paper, the data, your reproduced result next to the published one, your improvement and its measured effect (mean ± spread).]

## 1 · The paper and the task (research understanding, part of 15 %)

- **Paper:** Yoon Kim, Convolutional Neural Networks for Sentence Classification, EMNLP, 2014, doi.org/10.3115/v1/D14-1181
- **Task:** Movie reviews with one sentence per review goes in and Classification involving detecting positive/negative reviews comes out,Customer reviews of various products (cameras, MP3s etc.) goes in and prediction of positive/negative reviews comes out.
- **Their method in your own words:** Represent words with trainable embeddings, apply convolution filters of widths 3, 4 and 5, then max-pool the strongest features.
- **Their main result:** 

| Model | Movie Review (MR) | Customer Review (CR) |
|---|---|---|
| CNN-rand (random word vectors) | 76.1 % | 79.8 % |
| CNN-static (word2vec, frozen) | 81.0 % | 84.7 % |
| CNN-non-static (word2vec, fine-tuned) | 81.5 % | 84.3 % |
| NBSVM (Wang and Manning 2012), not a network | 79.4 % | 81.8 % |

## 2 · The data (start this week)

| Dataset | Movie Review (MR) | Customer Review (CR) |
|---|---|---|
| Source and version | Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1.0.0 , 15-10-2026 | Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1.0.0 , 15-10-2026 |
| Licence | Apache 2.0 | Unknown (From Research Paper: https://www.cs.uic.edu/~liub/FBS/opinion-mining-final-WSDM.pdf) |
| Size | 10,662 samples, avg_len = 114 char/s | 3,770 samples, avg_len = 19 - 20 words |
| Classes or targets | [1 0], [5331 5331] (balanced) | [1 0], [2405 1365] (unbalanced) |
| Splits | 10 folds CV; 10 folds CV, because other splits give less accurate results. | Stratified 10-fold CV (seed 42), as in the paper (CR has no official split); HF train/test merged; 10 % of each training part = dev set for early stopping. Same folds for all models. |
| Preprocessing | Already cleaned data | Kim's `clean_str` tokenizer (lowercase, split clitics/punctuation); vocab from training folds only (unseen → `UNK`); zero-padding; no stop-word removal (keeps "not"/"no"). |
| Leakage checks | No duplicates, balanced | 5 exact duplicates removed (3,770 → 3,765; 2,400 pos / 1,365 neg), 0 conflicting labels; all vectorisers/vocab fitted inside training folds; imbalanced → majority floor 63.75 %, macro-F1 also reported. |
| Shared with Biomedical Image Analysis? | NO | NO |

EXAMPLE · NEGATIVE · Movie Review
“The film was dull and the story was disappointing.”

EXAMPLE · POSITIVE · Movie Review
“A wonderful film with excellent performances.”

EXAMPLE · NEGATIVE · Customer Review
“there is no way the battery supplied lasts 14 hours !”

EXAMPLE · POSITIVE · Customer Review
“the appearance is okay ed ( at least not ugly )”

## 3 · Reproducing the baseline (part of 15 %, and 25 % implementation)

**The recipe:** read the paper (or its code) like a recipe. One row per ingredient; leave nothing empty: if the paper does not say it, write "not stated" and what you chose.

| | The paper says | Where (page, table, code) | What we use (MR) | What we use (CR) |
|---|---|---|---|---|
| Data: version, split | 1.0.0, 10 fold CV | page 1748 | 1.0.0, 10 fold CV | SetFit/CR (Hu & Liu 2004), 3,765 sentences; stratified 10-fold CV + 10 % dev |
| Model: layers, sizes, pretrained? |One convolutional layer; filter widths 3, 4, 5; 100 feature maps per width; 300-dimensional pretrained word2vec embeddings; max-over-time pooling; fully connected softmax output; ReLU activation. | page 1748/49 | NBSVM + CNN(pre-trained embeddings) + Confidence Level | Kim's CNN: filter widths 3, 4, 5 (100 each), ReLU, max-pooling, dropout 0.5, softmax. Three embedding versions: random (CNN-rand), word2vec frozen (CNN-static), word2vec fine-tuned (CNN-non-static). Baselines: TF-IDF + LogReg, NBSVM |
| Training: optimiser, learning rate, schedule |Adadelta; explicit learning rate and external schedule not reported; early stopping on development data.| Not Specified | Training NBSVM as it is and training CNN-static, then get an ensembled result which gives us a confidence level of the classification| Adadelta (lr 1.0, ρ 0.95, ε 1e-6, from Kim's code); no schedule; max-norm 3; early stopping = best dev epoch |
| Training: epochs, batch size | 25 (from Kim's code), 50 | Kim's code | 25 (from Kim's code), 50 | 25 (from Kim's code), 50 |
| Augmentation and preprocessing |Tokenization, Padding, Embedding | page 1748/49 | Tokenization, Embedding using Word2Vec | No augmentation; `clean_str` tokenization, padding (+4 each side), random embeddings |
| Metric, and how it was averaged | Accuracy (%); 10-fold cross-validation for datasets without standard train/test splits. Exact fold aggregation and macro/micro averaging are not specified.| | Validation accuracy of NBSVM + Validation accuracy of CNN-Static (Both averaged over 10 folds) + A threshold (confidence) | Accuracy (%) mean ± SD over 10 test folds (+ macro-F1); 1 seed per fold |

- **Setup:** [framework and versions; anything not in the table]
  - **CR:** Python 3.11, PyTorch 2.15, scikit-learn 1.9.1, Hugging Face datasets 5.1.0, gensim 4.4.0. Apple MPS; the word2vec runs on a Colab T4 GPU.
- **What the paper did not say,** and what you chose instead: [list, from the "not stated" rows]
  - **CR:** Number of epochs (we use 25), Adadelta settings (from Kim's code), dev-set size (10 %), fold seed (42), how folds are averaged (mean ± sample SD), which sentences build the vocabulary (training part only), number of seeds (we use 3).

  **MR**
  ### Experimental Setup and Implementation Details

The Movie Review (MR) dataset consists of 10,662 movie reviews, comprising 5,331 positive and 5,331 negative samples. The experiments use Python 3.11, scikit-learn 1.9.1, Hugging Face datasets 5.1.0 and Gensim 4.4.0. Model training is conducted using the specified deep learning environment, with Apple MPS for local computation and Google Colab T4 GPU for Word2Vec training.

The CNN experiments follow the architecture proposed by Kim (2014), using three convolutional filter widths (3, 4 and 5), 100 feature maps per filter, 300-dimensional word embeddings, ReLU activation, max-over-time pooling, dropout of 0.5 and a max-norm constraint of 3.

To ensure reproducibility, experimental decisions not explicitly specified in the paper were fixed as follows: 25 training epochs, Adadelta optimizer settings based on Kim's original implementation, a development set comprising 10% of the outer training fold, and a fold-assignment random seed of 42. Vocabulary construction is restricted to the training portion of each fold to prevent data leakage. Model performance is evaluated using stratified 10-fold cross-validation and repeated across three random seeds. Results are reported as mean performance ± sample standard deviation.


| | Paper | Ours (mean ± spread, n seeds) |
|---|---|---|
| CR ·TF-IDF + LogReg	| not in the paper |	78.5 ± 1.8 % (10 folds, 1 seed each) |
| CR ·NBSVM |	81.8 % |	80.2 ± 2.0 % (10 folds, 1 seed each)	|
| CR ·CNN-rand |	79.8 % |	79.1 ± 1.9 %	(10 folds, 1 seed each) |
| CR ·CNN-static (word2vec, frozen) |	84.7 % |	82.8 ± 1.5 % (10 folds, 1 seed each)|
| CR ·CNN-non-static (word2vec, fine-tuned) |	84.3 % |	83.2 ± 1.5 % (10 folds, 1 seed each)|
| MR · CNN-rand accuracy |  |  |
| MR · CNN-static accuracy | 81.8 % | 80.2 ± 1.9 % (10 folds) |
| MR · CNN-NonStatic accuracy | 79.8 % | 79.3 ± 1.7 % (10 folds, 1 seed each) |
| MR · NBSVM accuracy | 81.8 % | 80.2 ± 1.9 % (10 folds) |
[Why the numbers differ, if they do: data version, preprocessing, compute, missing details. A difference that you explain is a result, too.]

**(CR):** CNN-rand (−0.7 pp), NBSVM (−1.6) and CNN-non-static (−1.1) are within one SD of the paper; CNN-static (−1.9, SD 1.5) is just outside. Likely causes: our vocabulary comes from the training folds only, we have 10 fewer sentences, our random folds differ, our NBSVM is simplified, and the hardware differs. word2vec clearly helps: +3.6 pp (static) and +4.1 pp (non-static) over CNN-rand, better on 9 and 10 of the 10 folds (p ≤ 0.004, results/cr_word2vec_paired.csv). The mid-course slides showed CNN-rand at 79.3 %: that was an earlier single-seed run, and Apple MPS is not fully deterministic.

## 4 · Your improvement (20 %)

- **Idea:** [what you change]
- **Why it should help:** [the reasoning: a weakness of the baseline you observed, a method from the course or from another paper]
- **Expected effect:** [what you expect to see, and on which metric]

**(CR · Ashok)**
 
- **Idea:** pretrain the Kim CNN on MR (10,662 movie-review sentences), then fine-tune all layers on each CR training fold. The fine-tuning vocabulary is MR ∪ CR-train; CR words not in the MR vocabulary keep the random U[−0.25, 0.25] initialisation.
- **Why it should help:** CR is small (3,765 sentences), so randomly initialised embeddings have little data to learn from. A zero-shot check shows the MR-trained model sits at the majority floor on CR (63.9 % vs 63.8 %) — domain shift (8.5 % of CR tokens have no MR vocab entry) and prior shift (MR is 50/50, CR is 64/36) stop direct transfer — but the learned polarity filters should still transfer after fine-tuning.
- **Expected effect:** higher accuracy and macro-F1 than CNN-rand on the same folds.
- **Second change (from feedback):** class balancing on the training folds only — a class-weighted cross-entropy loss (weights n/(2·n_c)) and, separately, random oversampling of the minority (negative) class. Expected: macro-F1 and negative recall up, accuracy roughly flat.

## 5 · Experiments (part of 20 % and 15 %)

- **Design:** one change at a time; the same data splits, the same budget; at least three seeds per setting.

| Setting | [metric] mean | ± spread | seeds |
|---|---|---|---|
| baseline | | | |
| + [change 1] | | | |
| + [change 2] | | | |

- **Statistical comparison:** [test and result, for example a paired test over seeds or folds (Session 31)]
- **Test set:** used **once**, for the final models only.

**(MR Mrudul)**
\- \*\*Idea:\*\* train the Kim CNN on the MR dataset (10,662 movie reviews), comparing three embedding strategies: CNN-Rand (randomly initialised and trainable embeddings), CNN-Static (pretrained Word2Vec embeddings kept frozen), and CNN-NonStatic (pretrained Word2Vec embeddings fine-tuned during training).

\- \*\*Why it should help:\*\* pretrained Word2Vec embeddings provide semantic information learned from a much larger text corpus, giving the CNN a stronger starting point than randomly initialised embeddings. The CNN can then learn sentiment-related phrase patterns through its convolutional filters. Fine-tuning the embeddings may allow the model to adapt the pretrained representations to movie-review sentiment.

\- \*\*Expected effect:\*\* higher accuracy and macro-F1 for CNN-Static and CNN-NonStatic than CNN-Rand, with CNN-NonStatic potentially performing best. Kim (2014) reported 76.1% accuracy for CNN-Rand, 81.0% for CNN-Static and 81.5% for CNN-NonStatic on MR. These are reference results, not guaranteed outcomes for our implementation.

\- \*\*Second change (from feedback):\*\* combine the CNN with NBSVM using a soft-voting ensemble. Train both models independently on the same training folds, then combine their predicted probabilities using a weighted average. Expected: improved accuracy and macro-F1 if the two models capture complementary sentiment patterns and make different classification errors.

Important distinction: MR contains 5,331 positive and 5,331 negative reviews, so it is already balanced. Unlike CR, class weighting or minority-class oversampling is not necessary for MR. The ensemble is a more meaningful second experiment here.&#x20;

Model results hypothesis.txt

Write CNN Model Code.txt


**(CR · Ashok)**
 
**Design:** we change one thing at a time and keep everything else fixed: the same 10 folds, the same dev split, 25 epochs, 3 seeds per setting. Numbers are mean ± sample SD over the 10 folds (`results/cr_ablation_summary.csv`).
 
| Setting | Accuracy % | Macro-F1 % | Negative recall % |
|---|---|---|---|
| CNN-rand (reference, see section 3) | 79.1 ± 1.9 | 76.8 ± 2.2 | 65.9 ± 5.0 |
| + class-weighted loss | 79.1 ± 2.0 | 76.9 ± 2.5 | 67.1 ± 6.2 |
| + oversampling of negatives (training part only) | 79.3 ± 2.3 | 77.2 ± 2.3 | 67.8 ± 3.3 |
| **MR → CR fine-tune (our improvement)** | **80.5 ± 1.7** | **78.7 ± 1.9** | **70.8 ± 2.9** |
 
**Is it significant?** Paired test over the 10 folds (paired t-test and Wilcoxon, `results/cr_paired_tests.csv`):
- **MR → CR fine-tune vs CNN-rand:** +1.4 pp accuracy (better on 9 of 10 folds, p = 0.005 / 0.004) and +1.9 pp macro-F1 (10 of 10, p = 0.0006 / 0.002). **Significant.**
- **Class weights** (0.0 pp, p = 0.84) and **oversampling** (+0.2 pp, p = 0.58): no significant change. Adding oversampling to the fine-tuned model gives 80.7 ± 1.4 % (+0.2 pp, p = 0.50): also not significant.
- **For the linear baselines, class weights do help:** NBSVM +1.1 pp accuracy (p = 0.001), LogReg +4.5 pp macro-F1.
 
**Test folds:** each fold is the test set exactly once; nothing was tuned on a test fold.
 
**In context:** the word2vec CNN (83.2 %, section 3) is still the most accurate model. Our improvement is the best result without external word vectors.

## 6 · Evaluation and error analysis (15 %)

- [Metrics that fit the task (not only accuracy): confusion matrix, precision, recall, F1, IoU, …]
- [Where does the model fail? Examples of its most confident mistakes; which classes are confused, and why]
- [Limitations: what your results do **not** show]

**(CR · Ashok)**
- **Metrics:** accuracy, macro-F1 and negative recall. CR is 64 % positive, so accuracy alone rewards always saying "positive" (63.8 %).
- **Best CNN without word2vec** (MR → CR fine-tune + oversampling, seed 0, all 3,765 sentences): negative precision 0.75, recall 0.70; positive precision 0.84, recall 0.86; accuracy 0.80. Confusion matrix: `figures/cr_confusion_best_cnn.png`.
- **Where it fails:** its most confident mistakes are negative sentences predicted positive with p ≈ 1.0. They are mixed sentences ("although the sd500 takes great quality photos, the sensitivity of the lcd compromises the quality overall") or sentences whose label depends on context ("i replaced it with a router raiser and it works much better"). Some are hard even for a human.
- **Limitations:** one dataset only; the 10 folds share training data, so the p-values are a little optimistic (Nadeau & Bengio 2003); the word2vec runs used a different computer (Colab T4 GPU) than the rest (Apple MPS), with the same folds, seeds and code.

## 7 · Conclusion

[What you reproduced, what you improved, what you would try next.]

**(CR · Ashok):** We reproduced Kim's CR results within about one SD: CNN-rand 79.1 % (paper 79.8 %), NBSVM 80.2 % (81.8 %), CNN-non-static 83.2 % (84.3 %). Our improvement, pretraining on movie reviews and then fine-tuning on CR, beats CNN-rand by +1.4 pp accuracy and +1.9 pp macro-F1, significant over the 10 folds. Class balancing did not help the CNN. Next steps: combine both ideas (start from word2vec, then pretrain on MR), and try freezing the convolution filters during fine-tuning.

## 8 · Use of AI tools (5 %)

[Which tools you used, for what (code, text, debugging), and how you **checked** their output. Be specific.]


## 9 · Reproducibility (part of 25 %)
 
**(CR · Ashok)**
- **Install:** `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (pinned versions).
- **Quick check (about 1 min, no training):** `python scripts/cr_tables.py` prints the results tables from `results/`.
- **Full run:** open `notebook/customer_review.ipynb` and run all cells. About 45–60 min on Apple MPS; the word2vec part also downloads 1.7 GB, and we ran it on a Colab T4 GPU. A full run rewrites `results/`; expect the same numbers within the ± spread. The data downloads automatically from Hugging Face (no token needed).
- **Seeds and hardware:** folds `StratifiedKFold(10, shuffle=True, random_state=42)`; dev split `random_state=fold`; model seeds 0, 1, 2. Apple MPS (word2vec: Colab T4 GPU). Checkpoints (`*.pt`) are not in Git; the notebook recreates them.
- **Where the results come from** (notebook parts = the numbered headings in `notebook/customer_review.ipynb`):
 
| Notebook part | Result files (in `results/` unless noted) |
|---|---|
| 1–12: data, LogReg, NBSVM, CNN-rand, MR zero-shot | `cr_baseline_cv.csv`, `cr_cnn_rand_cv.csv`, `cr_transfer_mr.csv`, `figures/cr_class_balance.png`, `figures/cr_confusion_logreg.png` |
| 13–15: ablation, paired tests, error analysis | `cr_ablation_cv.csv`, `cr_ablation_summary.csv`, `cr_paired_tests.csv`, `cr_oof_seed0.csv`, `cr_error_analysis.txt`, `figures/cr_confusion_best_cnn.png` |
| 16–17: word2vec CNNs | `cr_word2vec_cv.csv`, `cr_word2vec_summary.csv`, `cr_word2vec_paired.csv` |
 
## References
 
- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP, doi.org/10.3115/v1/D14-1181.
- Hu, M. & Liu, B. (2004). Mining and Summarizing Customer Reviews. KDD (CR dataset).
- Pang, B. & Lee, L. (2005). Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with Respect to Rating Scales. ACL (MR dataset).
- Wang, S. & Manning, C. D. (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. ACL (NBSVM).
- Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. arXiv:1212.5701.
- Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR 15.
- Nadeau, C. & Bengio, Y. (2003). Inference for the Generalization Error. Machine Learning 52.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. NeurIPS.
- Code: yoonkim/CNN_sentence (github.com/yoonkim/CNN_sentence).
- Data: Hugging Face `SetFit/CR`; `cornell-movie-review-data/rotten_tomatoes`.
- Libraries: PyTorch; scikit-learn (Pedregosa et al., JMLR 12). 
