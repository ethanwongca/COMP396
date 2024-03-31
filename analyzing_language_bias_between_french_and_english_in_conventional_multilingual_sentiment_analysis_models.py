# -*- coding: utf-8 -*-
"""Analyzing Language Bias Between French and English in Conventional Multilingual Sentiment Analysis Models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_O-T1aAsTBb8ryEzo50xbscrlzkP_wl8

# Calculating Bias for Multilingual Support Vector Machine and Naive-Bayes for Sentiment Analysis

## Library Imports

Libraries Used


*   **Sklearn**: Used to use the Multinomial Naive-Bayes and Support Vector Machine Model, build the Tf-Idf Matrix, use proper train test splitting, and build accuracy reports.
*   **Pandas**: Used for building DataFrames
*   **Numpy**: Provides operations for the DataFrames
*   **FairLearn**: Builds specified bias metrics in models
"""

!pip install fairlearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate, equalized_odds_ratio, demographic_parity_ratio
from google.colab import drive
drive.mount('/content/drive')

"""## Pre-Processing

Installing spaCy's French and English stop words and pre-processing tools, removing taggers as dataset is unordered.
"""

import spacy
!python -m spacy download en_core_web_sm
!python -m spacy download fr_core_news_sm

nlp_en = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
nlp_fr = spacy.load("fr_core_news_sm", disable=["ner", "parser", "tagger"])

"""## Pre-Processing Functions

*   **Review Parsing** - Parses through the dataset and reconstructs the text from the word frequencies provided, along with the sentiment labels as well.
*   **Batch Pre-Processing** - Pre-processes the text via spaCy using batch pre-processing with a batch size of 100.
*   **Load Dataset** - The fully pre-process data is placed in a dataframe.
"""

def review_parsing(line):
    """
    Parses lines from the dataset to reconstruct the text by multiplying by the
    proper word frequencies and extracting the
    sentiment labels as well.

    Args:
      String: The lines in the dataset

    Returns:
      Dict {str:str}: A dictionary that has text and sentiment as the keys and
      the reconstructed text and sentiment as values.
    """
    words = []
    parts = line.strip().split()
    sentiment = None

    for part in parts:
        if part.startswith("#label#"):
            sentiment = part.split(":")[1]
        else:
            word, freq = part.split(":")
            words.extend([word] * int(freq))

    reconstructed_text = " ".join(words)
    return {'text': reconstructed_text, 'sentiment': sentiment}

def batch_preprocess_en(texts):
    """
    Batch pre-process English texts.

    Args:
      List[str]: All of the English texts.

    Return:
      List[str]: All of the English texts fully pre-processed.
    """
    processed_texts = []
    for doc in nlp_en.pipe(texts, batch_size=100):
        tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
        processed_texts.append(' '.join(tokens))
    return processed_texts

def batch_preprocess_fr(texts):
    """
    Batch pre-process French texts.

    Args:
      List[str]: All of the French texts.

    Return:
      List[str]: All of the French texts fully pre-processed.
    """
    processed_texts = []
    for doc in nlp_fr.pipe(texts, batch_size=20):
        tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
        processed_texts.append(' '.join(tokens))
    return processed_texts


def load_dataset_to_dataframe_en(file_path):
    """
    Transforming the English pre-processed data into a dataframe.

    Args:
      FILE: The CSV file with all the English data.

    Return:
      DataFrame: A dataframe that has the pre-processed texts along with the
      sentiments.
    """
    texts, sentiments = [], []

    with open(file_path, 'r') as file:
        for line in file:
            parsed_line = review_parsing(line)
            texts.append(parsed_line['text'])
            sentiments.append(parsed_line['sentiment'])

    processed_texts = batch_preprocess_en(texts)

    df = pd.DataFrame({
        'ProcessedText': processed_texts,
        'Sentiment': sentiments
    })

    return df

def load_dataset_to_dataframe_fr(file_path):
    """
    Transforming the French pre-processed data into a dataframe.

    Args:
      FILE: The CSV file with all the French data.

    Return:
      DataFrame: A dataframe that has the pre-processed texts along with the
      sentiments.
    """
    texts, sentiments = [], []

    with open(file_path, 'r') as file:
        for line in file:
            parsed_line = review_parsing(line)
            texts.append(parsed_line['text'])
            sentiments.append(parsed_line['sentiment'])

    processed_texts = batch_preprocess_fr(texts)

    df = pd.DataFrame({
        'ProcessedText': processed_texts,
        'Sentiment': sentiments
    })

    return df

"""## Loading the Particular Datasets

The file path to the Webis-CLS-10 Dataset
"""

FILE_PATH_EN = '/content/drive/My Drive/dataset/en/music/unlabeled.processed'
FILE_PATH_FR = '/content/drive/My Drive/dataset/fr/music/unlabeled.processed'

FILE_PATH_EN_dvd = '/content/drive/My Drive/dataset/en/dvd/unlabeled.processed'
FILE_PATH_FR_dvd = '/content/drive/My Drive/dataset/fr/dvd/unlabeled.processed'

FILE_PATH_EN_books = '/content/drive/My Drive/dataset/en/books/unlabeled.processed'
FILE_PATH_FR_books = '/content/drive/My Drive/dataset/fr/books/unlabeled.processed'

"""Loading the French and English dataset from the Webis-CLS-10 Dataset. We are taking the all three sub-categories of the dataset which are the dvd, music, and books categories."""

data_en = load_dataset_to_dataframe_en(FILE_PATH_EN)
data_fr = load_dataset_to_dataframe_fr(FILE_PATH_FR)

data_en_dvd = load_dataset_to_dataframe_en(FILE_PATH_EN_dvd)
data_fr_dvd = load_dataset_to_dataframe_fr(FILE_PATH_FR_dvd)

data_en_books = load_dataset_to_dataframe_en(FILE_PATH_EN_books)
data_fr_books = load_dataset_to_dataframe_fr(FILE_PATH_FR_books)

"""## Caching the Datasets

Caching the datasets so there is no need to keep pre-processing the data
"""

data_en.to_csv("/content/drive/My Drive/dataset/en/music/unlabeled.csv", index=False)
data_fr.to_csv("/content/drive/My Drive/dataset/fr/music/unlabeled.csv", index=False)

data_en_dvd.to_csv("/content/drive/My Drive/dataset/en/dvd/unlabeled.csv", index=False)
data_fr_dvd.to_csv("/content/drive/My Drive/dataset/fr/dvd/unlabeled.csv", index=False)

data_en_books.to_csv("/content/drive/My Drive/dataset/en/books/unlabeled.csv", index=False)
data_fr_books.to_csv("/content/drive/My Drive/dataset/fr/books/unlabeled.csv", index=False)

data_en = pd.read_csv("/content/drive/My Drive/dataset/en/music/unlabeled.csv")
data_fr = pd.read_csv("/content/drive/My Drive/dataset/fr/music/unlabeled.csv")

data_en_dvd = pd.read_csv("/content/drive/My Drive/dataset/en/dvd/unlabeled.csv")
data_fr_dvd = pd.read_csv("/content/drive/My Drive/dataset/fr/dvd/unlabeled.csv")

data_en_books = pd.read_csv("/content/drive/My Drive/dataset/en/books/unlabeled.csv")
data_fr_books = pd.read_csv("/content/drive/My Drive/dataset/fr/books/unlabeled.csv")

"""# The Dataset Pre-Processed"""

display(data_en)

display(data_fr)

"""## Creating the Multi-Lingual Dataset

**Sample Data** A function that ensures equal number of Englsih and French Reviews in the Dataframe.
"""

def sample_data(df_en, df_fr, perc_en, perc_fr):
    """
    Adjusts sampling to ensure an equal number of English and French samples,
    maximizing the amount of data used while respecting the specified percentages.

    Args:
      Dataframe: The Pre-Processed English DataFrame
      Dataframe: The Pre-Proecessed French DataFrame
      Float: The percentage of English reviews in the dataset
      Float: The percentage of French reviews in the dataset

    Returns:
      Dataframe: The multilingual dataset
    """
    # Determine the maximum number of samples we can take equally from both datasets
    max_samples_en = int(len(df_en))
    max_samples_fr = int(len(df_fr))

    max_total = max_samples_en + max_samples_fr

    # The actual number of samples to take from each is the minimum of these two numbers
    if min(max_samples_en, max_samples_fr) < perc_en * max_total:
      actual_samples = min(max_samples_en, max_samples_fr)
    else:
      actual_samples = perc_en * max_total

    # Sample these numbers from each dataframe
    sample_en = df_en.sample(n=actual_samples, random_state=42)
    sample_fr = df_fr.sample(n=actual_samples, random_state=42)

    # Mark each sample with its language
    sample_en['Language'] = 'English'
    sample_fr['Language'] = 'French'

    # Combine and return the samples
    return pd.concat([sample_en, sample_fr], ignore_index=True)

"""### Building the Tf-Idf Matrix

A Tf-Idf Matrix must be built to have the SVM and Naive Bayes to work.
"""

def preprocess_and_vectorize(df):
    """
    Pre-Processes and vectorizes the text data in the DataFrame.

    Args:
      DataFrame: The Multi-Lingual Dataset

    Returns:
      DataFrame: Tf-Idf of the shape of the samples and feature representing the
      vectorized text data.
      DataFrame: The sentiment labels associated with each text
      TfidfVectorizer: Contains the vocabulary and idf scores of each term
      NumpyArray: An array of every text entry's language
    """
    tfidf = TfidfVectorizer(max_features=10000)
    X = tfidf.fit_transform(df['ProcessedText'])
    y = df['Sentiment'].values
    return X, y, tfidf, df['Language'].values

"""## Calculating the Specified Bias Metrics, Training the SVM and Naive Bayes, and Splitting the Data

**Ensure Binary Labels**: Ensures the specified labels are binary, so the bias metrics function works as intended.

**Calculate Bias Metrics**: Calculates the demographic parity ratiom equalized odds ratio, demographic parity difference, and equalized odds difference.

**Map Labels**: Maps the sentiment labels to have binary classification.

**Train and Evaluate:** Trains the SVM and Naive Bayes models, and calculating the corresponding precision, recall and f1-scores for each langugage. Also outputs the bias metrics for the models.
"""

def ensure_binary_labels(y):
    """
    Ensures the specified labels are binary, so the bias metrics function works
    as intended

    Args:
      DataFrame: The sentiment dataframe
    """
    unique_labels = np.unique(y)
    if set(unique_labels) == {0, 1} or set(unique_labels) == {-1, 1}:
        return np.where(y == -1, 0, y)
    else:
        raise ValueError("Labels must be binary and in {0, 1} or {-1, 1}.")

def calculate_bias_metrics(y_true, y_pred, sensitive_features):
    """
    Calculates the demographic parity ratiom equalized odds ratio,
    demographic parity difference, and equalized odds difference.

    Args:
      DataFrame: Dataframe the contains the actual sentiment labels for the
      specified text in the dataset
      DataFrame: Dataframe the contains the predicted sentiment labels for the
      specified text in the dataset
      DataFrame: Contains the dataframe with the languages corresponding to
      the sentiment labels
    """
    y_true_binary = ensure_binary_labels(y_true)
    y_pred_binary = ensure_binary_labels(y_pred)

    m_dpr = demographic_parity_ratio(y_true_binary, y_pred_binary, sensitive_features=sensitive_features)
    m_eqo = equalized_odds_ratio(y_true_binary, y_pred_binary, sensitive_features=sensitive_features)
    m_dpr_2 = demographic_parity_difference(y_true_binary, y_pred_binary, sensitive_features=sensitive_features)
    m_eqo_2 = equalized_odds_difference(y_true_binary, y_pred_binary, sensitive_features=sensitive_features)

    print(f"The demographic parity ratio is {m_dpr}")
    print(f"The equalized odds ratio is {m_eqo}")
    print(f"The demographic parity differece is {m_dpr_2}")
    print(f"The equalized odds difference is {m_eqo_2}")

def map_labels(y):
    """
    Maps the positve and negative setiments to be binary

    Args:
      DataFrame: The dataset without binary labels for sentiment

    Return:
      DataFrame: The dataset with binary labels

    """

    return np.where(y == 'positive', 1, 0)

def train_and_evaluate(X_train, y_train, X_test, y_test, languages_test, model, model_name="Model"):
    """
    Trains the SVM and Naive Bayes models, and calculating the corresponding
    precision, recall, and f1-scores for each language. Also outputs the
    bias metrics for the models.

    Args:
      DataFrame: The training dataframe with data other than sentiment
      DataFrame: The training dataframe with the sentiment data
      DataFrame: The testing dataframe with data other than sentiment
      DataFrame: The testing dataframe with the sentiment data
      DataFrame: A language dataframe that corresponds to the y_test sentiment data
      SkLearn Model: The specified model to be trained
    """

    y_train_mapped = map_labels(y_train)
    y_test_mapped = map_labels(y_test)

    model.fit(X_train, y_train_mapped)
    y_pred_mapped = model.predict(X_test)

    bias_metrics = calculate_bias_metrics(y_test_mapped, y_pred_mapped, languages_test)

    y_pred = np.where(y_pred_mapped == 1, 'positive', 'negative')
    print(f"Results for {model_name}:")
    print("Overall Accuracy:", accuracy_score(y_test, y_pred))
    print("Overall Classification Report:")
    print(classification_report(y_test, y_pred))

    for language in ['English', 'French']:
        idx = languages_test == language
        y_test_lang = y_test[idx]
        y_pred_lang = y_pred[idx]

        print(f"Accuracy on {language}: {accuracy_score(y_test_lang, y_pred_lang)}")
        print(f"Classification Report for {language}:")
        print(classification_report(y_test_lang, y_pred_lang))

    print("----------------------------------------------------")

"""# Training the Music Data Using SVM and Naive Bayes

Splitting up the data, using an 80-20 split and calling the corresponding functions for a properly trained model.
"""

df_sampled = sample_data(data_en, data_fr, 0.5, 0.5)

X, y, tfidf, languages = preprocess_and_vectorize(df_sampled)

# Split the data, ensuring languages array is split consistently with X and y
X_train, X_test, y_train, y_test, languages_train, languages_test = train_test_split(X, y, languages, test_size=0.2, random_state=42)

# Initialize models
svm_model = SVC(kernel='linear')
nb_model = MultinomialNB()

# Train and evaluate models, including language-specific performance
print("Evaluating SVM...")
train_and_evaluate(X_train, y_train, X_test, y_test, languages_test, svm_model, "SVM")

print("Evaluating Naive Bayes...")
train_and_evaluate(X_train, y_train, X_test, y_test, languages_test, nb_model, "Naive Bayes")

"""# Training the DVD Data Using SVM and Naive-Bayes"""

df_sampled = sample_data(data_en_dvd, data_fr_dvd, 0.5, 0.5)
X, y, tfidf, languages = preprocess_and_vectorize(df_sampled)

# Split the data, ensuring languages array is split consistently with X and y
X_train, X_test, y_train, y_test, languages_train, languages_test = train_test_split(X, y, languages, test_size=0.2, random_state=42)

# Initialize models
svm_model = SVC(kernel='linear')
nb_model = MultinomialNB()

# Train and evaluate models, including language-specific performance
print("Evaluating SVM...")
train_and_evaluate(X_train, y_train, X_test, y_test, languages_test, svm_model, "SVM")

print("Evaluating Naive Bayes...")
train_and_evaluate(X_train, y_train, X_test, y_test, languages_test, nb_model, "Naive Bayes")

"""# Training the Books Data Using SVM and Naive-Bayes"""

df_sampled = sample_data(data_en_books, data_fr_books, 0.5, 0.5)
X, y, tfidf, languages = preprocess_and_vectorize(df_sampled)

X_train, X_test, y_train, y_test, languages_train, languages_test = train_test_split(X, y, languages, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
nb_model = MultinomialNB()

print("Evaluating SVM...")
train_and_evaluate(X_train, y_train, X_test, y_test, languages_test, svm_model, "SVM")

print("Evaluating Naive Bayes...")
train_and_evaluate(X_train, y_train, X_test, y_test, languages_test, nb_model, "Naive Bayes")

"""# Transforming the Notebook into a Latex File"""

!apt-get -q install texlive-xetex texlive-fonts-recommended texlive-plain-generic

!jupyter nbconvert --to pdf "/content/drive/My Drive/Colab Notebooks/Analyzing Language Bias Between French and English in Conventional Multilingual Sentiment Analysis Models.ipynb"

import os
from google.colab import files
files.download(f"/content/drive/My Drive/Colab Notebooks/Analyzing Language Bias Between French and English in Conventional Multilingual Sentiment Analysis Models.pdf")