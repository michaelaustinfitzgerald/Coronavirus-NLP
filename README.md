# Coronavirus-NLP: Project Overview

## Code and Resources Used
**Python Version:** 3.7\
**Packages:** pandas, plotly, nltk, seaborn, sklearn, numpy\
**Data Source:** https://www.kaggle.com/datatattle/covid-19-nlp-text-classification\
## EDA
### WordCloud
![Word Cloud](https://github.com/michaelaustinfitzgerald/Coronavirus-NLP/blob/main/WordCloud.png)
### Word Count - Post Cleaning
![Word Count](https://github.com/michaelaustinfitzgerald/Coronavirus-NLP/blob/main/word_count.png)
## Sentiment - Frequency Count
### Pre-Simplification
![Pre](https://github.com/michaelaustinfitzgerald/Coronavirus-NLP/blob/main/sentiment_pre.png)
### Post-Simplifying 
![Post](https://github.com/michaelaustinfitzgerald/Coronavirus-NLP/blob/main/sentiment_post.png)

## Model and Parameter Tuning
**Model Type:** MultinomialNB\
**Best Parameters:**
* **alpha** = 1.5
* **fit_prior** = False

**Accuracy:** 0.78932
## Model Analysis
### Confusion Matrix
![Confusion Matrix](https://github.com/michaelaustinfitzgerald/Coronavirus-NLP/blob/main/cm.png)
### ROC Curve
![ROC Curve](https://github.com/michaelaustinfitzgerald/Coronavirus-NLP/blob/main/roc_curve.png)

