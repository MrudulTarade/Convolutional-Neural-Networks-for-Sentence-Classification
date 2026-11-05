# Convolutional Neural Networks for Sentence Classification

### A Deep Learning Project where CNN is used to classify Movie reviews and Customer Reviews, to see whether deep learning in NLP can be a key ingredient.

Based on the research paper "Kim: Convolutional neural networks for sentence classification. EMNLP 2014. doi.org/10.3115/v1/D14-1181", we have tried to reproduce the results of using CNN, along with pre-trained word-vectors for classifying movie reviews and customer reviews into different categories.

According to the research paper a CNN with little hyperparameter tuning, and multiple variants, seem to perform well for movie review classification and customer review classification.

### Two experiments are carried out to see whether we can improve the results which are:
#### Carried out by Mrudul:
1. Combine NBSVM with CNN on pre-trained word-vectors, to capture word-level sentiment and phrase pattern.
2. Adding a threshold and probability calibration to really know how confident the model is with it's prediction.

#### Carried out by Ashok:
1. Using a fine-tuned Movie Review pretrained CNN rather than a word2vec CNN trained on the ~3,700 samples of Customer Reviews.


## How to Run This Repository

#### 1. Requirements

Make sure Python is installed on your system. Install the required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
You will also require ` GoogleNews-vectors-Negative300.bin `(3.64GB) for CNN models that use pre-trained word vectors. 
Download-link: https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

#### 2. Clone the Repository

Clone the repository using Git:

```bash
git clone <https://github.com/MrudulTarade/Convolutional-Neural-Networks-for-Sentence-Classification>
```

Navigate into the project directory:

```bash
cd <Convolutional Neural Networks for Sentence Classification>
```

#### 3. Repository Structure

The repository is organized as follows:

```text
├── data/
│   └── Movie and Customer Review datasets
│
├── notebooks/
│   ├── movie_review.ipynb
│   └── customer_review.ipynb
│
├── results/
│   └── Model evaluation results
│
├── figures/
│   └── Generated plots and visualizations
│
├── requirements.txt
└── README.md
```

**Directory overview:**

- **data/** – Contains the datasets used for movie review and customer review sentiment classification.
- **notebooks/** – Contains the Jupyter notebooks with the respective experiments:
  - `movie_review.ipynb` – Movie review sentiment classification.
  - `customer_review.ipynb` – Customer review sentiment classification.
- **results/** – Contains the evaluation results and model performance comparisons.
- **figures/** – Contains plots and visualizations generated during the experiments.

#### 4. Running the Experiments

The experiments are implemented in Jupyter notebooks.

1. Navigate to the `notebooks/` directory.
2. Open the notebook corresponding to the dataset you want to experiment with.
3. Execute the notebook cells sequentially, starting from the imports and data preparation sections.

The notebooks cover the data preparation, model training, evaluation, and comparison processes.

**CNN implementation:** The CNN experiments are designed to run using Google Colab with GPU acceleration. Upload the relevant notebook and dataset to Colab, select a GPU runtime under **Runtime → Change runtime type**, and execute the notebook cells sequentially.

Import the ` GoogleNews-vectors-Negative300.bin ` in collab for the CNN.

### 5. Report

Refer to the project report for a detailed explanation of the work carried out. It documents the methodology, model architectures, experimental setup, evaluation metrics, results, and comparisons.

The report provides the context needed to understand the implementation and interpret the results obtained from the notebooks.

### Contributors: 
### ` Ashok Rai ` ` Mrudul Tarade `