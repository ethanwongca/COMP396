# Analyzing Language Bias Between French and English in Conventional Multilingual Sentiment Analysis Models

## Project Description 
Inspired by the "Bias Considerations in Bilingual Natural Language Processing" report by Statistics Canada, this study investigates bias in cross-lingual sentiment analysis between English and French. Observing the report's identification of inconsistent bias trends, we examine whether sentiment analysis across these languages exhibits language bias or maintains these inconsistent trends. Through the construction and analysis of two machine learning models, Support Vector Machine (SVM) and Naive Bayes, across three balanced datasets (ensuring a 50:50 split between English and French data), we aimed to uncover potential biases in sentiment classification. Our results indicate that the French data consistently outperformed English data in terms of accuracy, recall, and F1 score, suggesting a language bias in favor of French. These findings underscore the necessity of developing equitable multilingual Natural Language Processing (NLP) systems, highlighting the importance of fairness in sentiment classification across languages. This study not only identifies potential biases but also proposes considerations for future development of bias-aware multilingual NLP systems. <br/>

## Libraries Used
**Sklearn:** Used to use the Multinomial Naive-Bayes and Support Vector Machine Model, build the Tf-Idf Matrix, use proper train test splitting, and build accuracy reports <br/>
**Pandas:** Used for building DataFrames <br/>
**NumPy:** Provides operations for the DataFrames <br/>
**FairLearn:** Builds specified bias metrics in models <br/>
**spaCy:** Pre-process the French and English Data

## Dataset 
Webis-CLS-10 Dataset: The dataset contains pre-processed multilingual data from Amazon Product Reviews with a positive or negative sentiment. 
