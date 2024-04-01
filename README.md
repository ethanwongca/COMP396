# Analyzing Language Bias Between French and English in Conventional Multilingual Sentiment Analysis Models <img height=40 width=40 src="https://github.com/ethanwongca/COMP396/assets/87055387/8ab34a73-38e2-4cee-aecd-4a66c02d19b7">

## Table of Contents 
1. [Project Abstract](#project-abstract)
2. [Development Environment](#development-environmnet)
3. [Libraries](#libraries)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
   - [Pre-Processing](#pre-processing)
   - [Feature Engineering](#feature-engineering)
   - [Model Building](#model-building)
   - [Bias Metrics](#bias-metrics)
6. [Contributors](#contributors)
7. [Citations](#citations)
8. [Interactive Notebook](#interactive-notebook)


## Project Abstract
**Abstract**: Inspired by the 'Bias Considerations in Bilingual Natural Language Processing' report by Statistics Canada, this study delves into potential biases in cross-lingual sentiment analysis between English and French. Addressing the report's highlight of inconsistent bias trends, we investigate the presence of language biases or the continuity of these trends through the lens of sentiment analysis. By employing Support Vector Machine (SVM) and Naive Bayes models on three balanced datasets, we aim to reveal potential biases in multilingual sentiment classification. Utilizing Fairlearn, a tool for assessing bias in machine learning models, our findings reveal nuanced outcomes: French data outperforms English across accuracy, recall, and F1 score metrics in both models, hinting at a language bias favoring French. However, Fairlearn's metrics indicate SVM approaches equitable levels with a demographic parity ratio of 0.9997, suggesting near-equitable treatment across languages. In contrast, Naive Bayes demonstrates greater disparities, evidenced by a demographic parity ratio of 0.8327. These results underscore the need for developing equitable multilingual Natural Language Processing (NLP) systems and highlight the importance of incorporating fairness metrics in sentiment classification across languages. This study advocates for the integration of bias-aware considerations in development of multilingual NLP models. <br/>

## Development Environmnet 
**IDE**: Google Colab (Python 3 Google Compute Engine)<br/>

**Programming Languages**: Python 3<br/>

## Libraries
- **Sklearn (v1.2.2):** For machine learning models, Tf-Idf matrix creation, and accuracy reports. [Sklearn Documentation](https://scikit-learn.org/stable/)
- **Pandas (v2.2.1):** For data manipulation and analysis. [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- **NumPy (v1.25.2):** Provides operations for the DataFrames. [NumPy Documentation](https://numpy.org/doc/)
- **FairLearn (v0.10.0):** For bias metric evaluation. [FairLearn Documentation](https://fairlearn.org/)
- **SpaCy (v3.7.4):** For data pre-processing in French and English. [SpaCy Documentation](https://spacy.io/)

>[!NOTE]
> Versions can be found in the **requirenments.txt**. <br/>
> Versions used are installed in Google Colab's IDE, with the exception of Fairlearn.

## Dataset 
Utilized the **Webis-CLS-10 Dataset**, featuring pre-processed multilingual Amazon product reviews with sentiment labels, encompassing French, English, German, and Japanese.

## Methodology 

### Pre-Processing 
**SpaCy** was used for pre-processing a second time to maintain a level of consistency throughout the data. </br>
Batch pre-processing was used to pre-process, the sample code for the pre-processing functions are the following: </br>
#### Review Parsing 
This function is important for reconstructing the text as the text was in this format. </br>
`vallenatos:3 !:2 rien:2 .:2 ":2 les:2 modernisés:1 pop:1 rappeler:1 puisse:1 pas:1 mais:1 avec:1 à:1 ,:1 'est:1 cela:1 'a:1 qui:1 du:1 voir:1 c:1 mème:1 n:1 plus:1 #label#:negative`
```python
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
```

#### Batch Pre-Processing and Dataset Load
For the batch pre-processing and dataset building separate functions were used for the French and English datasets, as it makes it manageable to pre-process. <br/>
Each batch consisted of 100 reviews, as it made pre-processing process significantly quicker. 

```python
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
```
### Creating the Multilingual Dataset 
To balance the English and French data in the dataset but maximize the amount of reviews used the following function was used.
This basically takes the max of the smallest dataset and then add data from the other dataset to equal the smallest dataset.
```python
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

    sample_en = df_en.sample(n=actual_samples, random_state=42)
    sample_fr = df_fr.sample(n=actual_samples, random_state=42)

    sample_en['Language'] = 'English'
    sample_fr['Language'] = 'French'

    return pd.concat([sample_en, sample_fr], ignore_index=True)
```
### Feature Engineering
The tf-idf matrix was build using the Sklearn library using 1000 features. 
```python
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
```
### Model Building 
Both Naive Bayes and Support Vector Machines were built using the following function. This function interacts with the <br/>
bias metrics function adding the bias metrics on top of the function. 
```python
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
```
This would result in the following table. <br/>
<img width="396" alt="Screenshot 2024-04-01 at 2 25 45 PM" src="https://github.com/ethanwongca/COMP396/assets/87055387/1d79adbd-42b2-4f96-94a3-6b69468bc5af">


### Bias Metrics 
The bias metrics were calculated using the Fairlearn library. Binary classification was needed for each sentiment for these metrics to work. 
```python
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
```
## Contributors 
### Author
* Ethan Wong
* Email: ethanwongca@gmail.com
### Supervisor 
* Faten M'hiri PhD

## Citations 
**Dataset:** <br/>
Prettenhofer, P., & Stein, B. (2010). Webis Cross-Lingual Sentiment Dataset 2010 (Webis-CLS-  <br/>
    10) [Data set]. *48th Annual Meeting of the Association of Computational Linguistics* <br/> 
    *(ACL 10)*. Zenodo. https://doi.org/10.5281/zenodo.3251672 <br/>
**Sci-kit Learn:** <br/>
Pedregosa, F., Varoquaux, G., Gramfort , A., Michel, V., Thirion, B., Grisel, O., Blondel, M., <br/>
    Muller, A., Nothman, J., Louppe, G., Prettenhofer, P., Weiss , R., Dubourg , V., Vanderplas, <br/>
    J., Passos, A., Cournapeau , D., Brucher , M., Perrot, M., & Duchesnay, E. (2011). Scikit- <br/>
    learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. <br/>
    https://doi.org/10.48550/arXiv.1201.049017 <br/>
**Fairlearn:** <br/>
Bird, S., Diduk, M., Edgar, R., Horn, B., Lutz, R., Milan, V., Sameki, M., Wallach, H., & Walker, K. (2020). <br/>
    *Fairlearn: A toolkit for assessing and improving fairness in AI*. Microsoft.

## Interactive Notebook
For a hands-on experience with our project, access our interactive Jupyter Notebooks hosted on Google Colab:
- [Multilingual Sentiment Analysis Notebook](https://colab.research.google.com/drive/1_O-T1aAsTBb8ryEzo50xbscrlzkP_wl8?usp=sharing)

>Note: Ensure you have a Google account to interact with the notebook. Follow the instructions within the notebook for setup and execution guidance.
