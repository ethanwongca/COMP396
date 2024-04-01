# Analyzing Language Bias Between French and English in Conventional Multilingual Sentiment Analysis Models <img height=30 width=30 src="https://github.com/ethanwongca/COMP396/assets/87055387/8ab34a73-38e2-4cee-aecd-4a66c02d19b7">
 

## Project Abstract
**Abstract**: Inspired by the 'Bias Considerations in Bilingual Natural Language Processing' report by Statistics Canada, this study delves into potential biases in cross-lingual sentiment analysis between English and French. Addressing the report's highlight of inconsistent bias trends, we investigate the presence of language biases or the continuity of these trends through the lens of sentiment analysis. By employing Support Vector Machine (SVM) and Naive Bayes models on three balanced datasets, we aim to reveal potential biases in multilingual sentiment classification. Utilizing Fairlearn, a tool for assessing bias in machine learning models, our findings reveal nuanced outcomes: French data outperforms English across accuracy, recall, and F1 score metrics in both models, hinting at a language bias favoring French. However, Fairlearn's metrics indicate SVM approaches equitable levels with a demographic parity ratio of 0.9997, suggesting near-equitable treatment across languages. In contrast, Naive Bayes demonstrates greater disparities, evidenced by a demographic parity ratio of 0.8327. These results underscore the need for developing equitable multilingual Natural Language Processing (NLP) systems and highlight the importance of incorporating fairness metrics in sentiment classification across languages. This study advocates for the integration of bias-aware considerations in development of multilingual NLP models. <br/>

## IDE and Language
**Google Colab**: Libraries and necessary dependencies were installed. This includes the Sci-kit learn, NumPy, Pandas, and SpaCy libraries. <br/>

**Python 3**: This is the language in Google Colab's framework. <br/>

## Libraries Used
**Sklearn (Version 1.2.2):** Used to use the Multinomial Naive-Bayes and Support Vector Machine Model, build the Tf-Idf Matrix, use proper train test splitting, and build accuracy reports. <br/>

**Pandas (Version 2.2.1):** Used for building DataFrames <br/>

**NumPy (Version 1.25.2):** Provides operations for the DataFrames <br/>

**FairLearn (Version 0.10.0):** Builds specified bias metrics in models <br/>

**SpaCy (Version 3.7.4):** Pre-process the French and English Data

## Dataset 
**Webis-CLS-10 Dataset:** The dataset contains pre-processed multilingual data, (French, English, German, and Japanese) from Amazon Product Reviews with a positive or negative sentiment. 

## Contributors 
### Author
* Ethan Wong
* Email: ethanwongca@gmail.com
### Supervisor 
* Faten M'hiri PhD

## Citations 
**Dataset:** <br/>
Prettenhofer, P., & Stein, B. (2010). Webis Cross-Lingual Sentiment Dataset 2010 (Webis-CLS-
10) [Data set]. 48th Annual Meeting of the Association of Computational Linguistics (ACL 10).
Zenodo. https://doi.org/10.5281/zenodo.3251672 <br/>
**Sci-kit Learn:** <br/>
 Pedregosa, F., Varoquaux, G., Gramfort , A., Michel, V., Thirion, B., Grisel, O., Blondel, M.,
Muller, A., Nothman, J., Louppe, G., Prettenhofer, P., Weiss , R., Dubourg , V., Vanderplas,
J., Passos, A., Cournapeau , D., Brucher , M., Perrot, M., & Duchesnay, E. (2011). Scikit-
learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
https://doi.org/10.48550/arXiv.1201.049017 <br/>
**Fairlearn:** <br/>
Bird, S., Diduk, M., Edgar, R., Horn, B., Lutz, R., Milan, V., Sameki, M., Wallach, H., &
Walker, K. (2020). Fairlearn: A toolkit for assessing and improving fairness in AI. Microsoft.



