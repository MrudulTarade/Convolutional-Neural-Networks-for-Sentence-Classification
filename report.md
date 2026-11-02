# [Title: what you reproduced and how you improved it]

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
| **CNN-non-static** (word2vec, fine-tuned) | **81.5 %** | 84.3 % |
| CNN-multichannel | 81.1 % | 85.0 % |
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

EXAMPLE · NEGATIVE
“The film was dull and the story was disappointing.”

EXAMPLE · POSITIVE
“A wonderful film with excellent performances.”

## 3 · Reproducing the baseline (part of 15 %, and 25 % implementation)

**The recipe:** read the paper (or its code) like a recipe. One row per ingredient; leave nothing empty: if the paper does not say it, write "not stated" and what you chose.

| | The paper says | Where (page, table, code) | What we use (MR) | What we use (CR) |
|---|---|---|---|---|
| Data: version, split | 1.0.0, 10 fold CV | page 1748 | 1.0.0, 10 fold CV | SetFit/CR (Hu & Liu 2004), 3,765 sentences; stratified 10-fold CV + 10 % dev |
| Model: layers, sizes, pretrained? |One convolutional layer; filter widths 3, 4, 5; 100 feature maps per width; 300-dimensional pretrained word2vec embeddings; max-over-time pooling; fully connected softmax output; ReLU activation. | page 1748/49 | NBSVM + CNN(pre-trained embeddings) | CNN-rand: same architecture (widths 3/4/5 × 100, ReLU, max-pool, dropout 0.5, softmax) but 300-d embeddings random (U[-0.25, 0.25]), not word2vec. Baselines: TF-IDF + LogReg, NBSVM |
| Training: optimiser, learning rate, schedule |Adadelta; explicit learning rate and external schedule not reported; early stopping on development data.| Not Specified | Training NBSVM as it is and training CNN-static, then get an ensembled result which gives us a threshold of the classification| Adadelta (lr 1.0, ρ 0.95, ε 1e-6, from Kim's code); no schedule; max-norm 3; early stopping = best dev epoch |
| Training: epochs, batch size | Not Explicit, 50 | | | 25 (from Kim's code), 50 |
| Augmentation and preprocessing |Tokenization, Padding, Embedding | | Tokenization, Embedding using Word2Vec | No augmentation; `clean_str` tokenization, padding (+4 each side), random embeddings |
| Metric, and how it was averaged | Accuracy (%); 10-fold cross-validation for datasets without standard train/test splits. Exact fold aggregation and macro/micro averaging are not specified.| | Validation accuracy of NBSVM + Validation accuracy of CNN-Static (Both averaged over 10 folds) + A threshold (confidence) | Accuracy (%) mean ± SD over 10 test folds (+ macro-F1); 1 seed per fold |

- **Setup:** [framework and versions; anything not in the table]
  - **CR:** Python 3.11, PyTorch 2.15 (Apple MPS), scikit-learn 1.9.1, HF datasets 5.1.0
- **What the paper did not say,** and what you chose instead: [list, from the "not stated" rows]
  - **CR:** epochs (25), Adadelta ρ/ε (0.95 / 1e-6), dev size (10 %), fold seed (42), fold averaging (mean ± SD), vocab scope (training folds only)

| | Paper | Ours (mean ± spread, n seeds) |
|---|---|---|
| [metric] | [value] | [value] |
| CR · CNN-rand accuracy | 79.8 % | 79.3 ± 1.7 % (10 folds, 1 seed each) |
| CR · NBSVM accuracy | 81.8 % | 80.2 ± 1.9 % (10 folds) |

[Why the numbers differ, if they do: data version, preprocessing, compute, missing details. A difference that you explain is a result, too.]

**CR:** Both gaps are within 1 SD (0.5 vs 1.7; 1.6 vs 1.9) → within noise. Causes: vocab from training folds only, different folds, 10 fewer sentences, simplified NBSVM (no interpolation, C = 0.1).

## 4 · Your improvement (20 %)

- **Idea:** [what you change]
- **Why it should help:** [the reasoning: a weakness of the baseline you observed, a method from the course or from another paper]
- **Expected effect:** [what you expect to see, and on which metric]

## 5 · Experiments (part of 20 % and 15 %)

- **Design:** one change at a time; the same data splits, the same budget; at least three seeds per setting.

| Setting | [metric] mean | ± spread | seeds |
|---|---|---|---|
| baseline | | | |
| + [change 1] | | | |
| + [change 2] | | | |

- **Statistical comparison:** [test and result, for example a paired test over seeds or folds (Session 31)]
- **Test set:** used **once**, for the final models only.

## 6 · Evaluation and error analysis (15 %)

- [Metrics that fit the task (not only accuracy): confusion matrix, precision, recall, F1, IoU, …]
- [Where does the model fail? Examples of its most confident mistakes; which classes are confused, and why]
- [Limitations: what your results do **not** show]

## 7 · Conclusion

[What you reproduced, what you improved, what you would try next.]

## 8 · Use of AI tools (5 %)

[Which tools you used, for what (code, text, debugging), and how you **checked** their output. Be specific.]

## 9 · Reproducibility (part of 25 %)

- **Run it:** [the commands or notebook order; the environment, for example `requirements.txt`]
- **Seeds and hardware:** [seeds used; GPU or CPU; training time]
- **Where the results come from:** [which notebook/script makes which table and figure]

## References

[All papers, data sets, code and websites you used.]
