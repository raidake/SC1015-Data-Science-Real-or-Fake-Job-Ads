# SC1015-Data-Science-Real-or-Fake-Job-Ads

## About
This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence)
Dataset: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
Fake/Real Job Prediction

## Problem Definition
With the rise in fake job postings, it is important to differentiate real from fake job postings. 

What variables can help determine whether a post is fraudulent or not?
Which classification model can help predict a fraudulent post?
## Real-Life Problem
A website that can filter out fraudulent posts can promote more visitors to browse the site, as well as more companies to post their job post

## Files

Tokenized.csv can be downloaded from [here](https://drive.google.com/file/d/1AONsu4uFsm-8Srzmib2j-fEWySy6zjzR/view?usp=sharing).

Presentation Video can be viewed [here](https://youtu.be/O_X9zBPZDwo).


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

## Files

- reallyCleanData.csv: contains filtered columns of fake_job_postings.csv
- EDA.ipnyb: Final EDA performed on dataset
- finalisedNLP.ipnyb: Contains codes to perform Natural Language Processing on text data obtained from reallyCleanData.csv
- Classifiers on job description (vectorized).ipynb: Apply models to vectorised text data, for prediction.

## Learning Outcomes / Conclusion / Improvements Needed
- Handling imbalanced data with Stratified K-fold for cross validation, configure parameters on classifiers to balance data
- Utilising and properly configuring SGDClassifier, Random Forest Classifier, K-Nearest Neighbours from sklearn
- Preparing text data for Natural Language Processing and using CUDA to speed up process
- Understaing why f1-score is a better metric for evaluating classifiers
- Collaborating efficently with Github Desktop
- Improvements Needed:
  - Cluster visualisation
  - Keras model

## Contributions
@raidake (Wei Feng) 
- Data-Cleaning 
- EDA
- TensorFlow
  - Keras
- K-Fold Stratified Cross Validation 
- Improvements to all codes

@ttan-999 (Tony) 
- Classification models
  - SGDClassifier
  - Random Forest Classifier
  - K-Neighbours Classifier
- Confusion Matrix


@jvnsjh (Jovan) 
- Text Cleaning
  - Natural Language Processing
    - NLTK (Tokenization & Lemmatization)
    - spaCy
- Word Clouds




## References:

- https://analyticsindiamag.com/classifying-fake-and-real-job-advertisements-using-machine-learning/

- https://monkeylearn.com/blog/classification-algorithms/

- https://medium.com/mlearning-ai/machine-learning-tools-for-fraudulent-job-post-classification-69cf52c20bdf

- https://towardsdatascience.com/clustering-product-names-with-python-part-1-f9418f8705c8

- https://towardsdatascience.com/clustering-product-names-with-python-part-2-648cc54ca2ac

- https://medium.com/@ageitgey/natural-language-processing-is-fun-9a0bff37854e

- https://datascience.stackexchange.com/questions/65341/f1-score-vs-accuracy-which-metric-is-more-important

- https://scikit-learn.org/


