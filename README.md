# SC1015-Data-Science-Real-or-Fake-Job-Ads

## About
This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence)
Dataset: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
Fake/Real Job Prediction

## Contributions
- @raidake (Wei Feng) - EDA and improvements to all codes
- @ttan-999 (Tony) - Classification models and prediction accuracy
- @jvnsjh (Jovan) - Implementation of NLP

## Problem Definition
With the rise in fake job postings, it is important to differentiate real from fake job postings.  
Are we able to predict fraudulent jobs through information gathered from their advertisments?

## Libraries / Models / Algorithms
- SKLearn
  - **SGDClassifiers (Best)**
  - Random Forest Classifier
  - K-Neighbours Classifier
- Pandas
- Tensorflow
- NLTK
  - Tokenizor
  - WordNetLemmatizer
- Wordclouds
- spaCy
- tensorflow

## Files

- reallyCleanData.csv: contains filtered columns of fake_job_postings.csv
- EDA.ipnyb: Final EDA performed on dataset
- finalisedNLP.ipnyb: Contains codes to perform Natural Language Processing on text data obtained from reallyCleanData.csv
- Classifiers on job description (vectorized).ipynb: Apply models to vectorised text data, for prediction.

## Learning Outcomes / Conclusion / Improvements Needed
- Handling imbalanced data with Stratified K-fold for cross validation, configure parameters on classifiers to balance data
- 
- Improvements Needed:
  - Cluster visualisation
  - Keras model

## References:

- https://analyticsindiamag.com/classifying-fake-and-real-job-advertisements-using-machine-learning/

- https://monkeylearn.com/blog/classification-algorithms/

- https://medium.com/mlearning-ai/machine-learning-tools-for-fraudulent-job-post-classification-69cf52c20bdf

- https://towardsdatascience.com/clustering-product-names-with-python-part-1-f9418f8705c8

- https://towardsdatascience.com/clustering-product-names-with-python-part-2-648cc54ca2ac

- https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e



