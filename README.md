# Analyzing Language Bias Between French and English in Traditional Multilingual Sentiment Analysis Models <img height=25 width=25 src="https://github.com/ethanwongca/COMP396/assets/87055387/8ab34a73-38e2-4cee-aecd-4a66c02d19b7">
 

## Project Abstract
**Abstract**: Inspired by the 'Bias Considerations in Bilingual Natural Language Processing' report by Statistics Canada, this study delves into potential biases in cross-lingual sentiment analysis between English and French. Addressing the report's highlight of inconsistent bias trends, we investigate the presence of language biases or the continuity of these trends through the lens of sentiment analysis. By employing Support Vector Machine (SVM) and Naive Bayes models on three balanced datasets, we aim to reveal potential biases in multilingual sentiment classification. Utilizing Fairlearn, a tool for assessing bias in machine learning models, our findings reveal nuanced outcomes: French data outperforms English across accuracy, recall, and F1 score metrics in both models, hinting at a language bias favoring French. However, Fairlearn's metrics indicate SVM approaches equitable levels with a demographic parity ratio of 0.9997, suggesting near-equitable treatment across languages. In contrast, Naive Bayes demonstrates greater disparities, evidenced by a demographic parity ratio of 0.8327. These results underscore the need for developing equitable multilingual Natural Language Processing (NLP) systems and highlight the importance of incorporating fairness metrics in sentiment classification across languages. This study advocates for the integration of bias-aware considerations in development of multilingual NLP models. <br/>

## IDE 
**Google Colab**: Libraries and necessary dependencies were installed. This includes the sci-kit learn, numpy, and pandas library; 

## Libraries Used
**Sklearn:** Used to use the Multinomial Naive-Bayes and Support Vector Machine Model, build the Tf-Idf Matrix, use proper train test splitting, and build accuracy reports <br/>

**Pandas:** Used for building DataFrames <br/>

**NumPy:** Provides operations for the DataFrames <br/>

**FairLearn:** Builds specified bias metrics in models <br/>

**spaCy:** Pre-process the French and English Data

## Dataset 
**Webis-CLS-10 Dataset:** The dataset contains pre-processed multilingual data from Amazon Product Reviews with a positive or negative sentiment. 


