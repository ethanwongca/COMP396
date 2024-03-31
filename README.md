# Analyzing Language Bias Between French and English in Traditional Multilingual Sentiment Analysis Models <img height=25 width=25 src="https://github.com/ethanwongca/COMP396/assets/87055387/8ab34a73-38e2-4cee-aecd-4a66c02d19b7">
 

## Project Description 
**Abstract**: Inspired by the "Bias Considerations in Bilingual Natural Language Processing" report by Statistics Canada, this study investigates bias in cross-lingual sentiment analysis between English and French. Observing the report's identification of inconsistent bias trends, we examine whether sentiment analysis across these languages exhibits language bias or maintains these inconsistent trends. Through the construction and analysis of two machine learning models, Support Vector Machine (SVM) and Naive Bayes, across three balanced datasets (ensuring a 50:50 split between English and French data), we aimed to uncover potential biases in sentiment classification. Our results indicate that the French data consistently outperformed English data in terms of accuracy, recall, and F1 score, suggesting a language bias in favor of French. These findings underscore the necessity of developing equitable multilingual Natural Language Processing (NLP) systems, highlighting the importance of fairness in sentiment classification across languages. This study not only identifies potential biases but also proposes considerations for future development of bias-aware multilingual NLP systems. <br/>

## IDE 
**Google Colab**: Libraries and necessary dependencies were already installed. 

## Libraries Used
**Sklearn:** Used to use the Multinomial Naive-Bayes and Support Vector Machine Model, build the Tf-Idf Matrix, use proper train test splitting, and build accuracy reports <br/>

**Pandas:** Used for building DataFrames <br/>

**NumPy:** Provides operations for the DataFrames <br/>

**FairLearn:** Builds specified bias metrics in models <br/>

**spaCy:** Pre-process the French and English Data

## Dataset 
**Webis-CLS-10 Dataset:** The dataset contains pre-processed multilingual data from Amazon Product Reviews with a positive or negative sentiment. 


