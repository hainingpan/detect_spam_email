
# Project: Detect spam emails
Construct a spam emails classifier using the naive Bayes and SVM
both from scratch and `sklearn`

#  Data
Data are obtained from [Enron spam emails](http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron{}.tar.gz) with labels indicating spams or not. The full dataset includes six folders with balanced number of spam emails and nonspam emails.

# Method
1. Preprocessing data
    * Clean raw email texts by removing human names, stopwords, and lemmatize each remaining word using `nltk`.
    *  Vectorize cleaned texts within the bag of words model using `sklearn.feature_extraction.text.CountVectorizer`. The converted data is now a sparse matrix, where each value stores the frequency of a feature (word).

2. Bayes
    * Constructed from scratch (`class Bayes`).
        * Calculate the prior probability and likelihood from the training data.
        * Fit the posterior probability for the test data.
        * Split the training and test data using `StratifiedKFold` to maintain the balanced number of classes for two labels. 
        * Cross validation to fine-tune the parameters (e.g., `max_features` in `CountVectorizer`, smoothing factor in the likelihood). `AUC of ROC` and `accuracy` are metrics used to select the optimal model. The first step is a coarse search followed by a finer search of smoothing factor as a second step.

    * Constructed from `sklearn.naive_bayes.MultinomialNB`.
        * Fit the posterior probability of the test data using `sklearn.naive_bayes.MultinomialNB`. 
        * Cross validation to fine-tune to parameters (e.g., `max_features` in `CountVectorizer`, smoothing factor `alpha` in the `MultinomialNB`). `AUC of ROC` and `accuracy` are metrics used to select the optimal model. The first step is a coarse search followed by a finer search of smoothing factor as a second step.

2. SVM
    * Constructed from scratch (`class SVM`). (*To do*)

    * Constructed from `sklearn.svm.SVC` and `sklearn.svm.LinearSVC`.
        * Construct a classifier using `SVC`.
        * Cross validation to choose the kernel between `linear` and `rbf`.
        * Determine the `kernel='linear'`, switch the classifer to `LinearSVC` for better performance, and start a coarse search to fine-tune `max_features` in `CounterVectorizer` and regularization type and strength.
        * Determining the regularization type (L2) and `max_features=None`, proceed to a fine search to find optimal regularization strength.


# Results
**Summarize the optimal result of all above methods**
| model                    | best_parameters                                  |   best_ROC_AUC |   best_accuracy |
|--------------------------|--------------------------------------------------|----------------|-----------------|
| Bayes from scratch       | max_features=None,smoothing=0.01                 |       0.997092 |        0.987276 |
| Bayes from MultinomialNB | max_features=None,smoothing=0.01                 |       0.996957 |        0.987276 |
| SVM from LinearSVC       | max_features=None,kernel=linear,penalty=L2,C=0.1 |       0.997582 |        0.987128 |



# To Do:
* SVM from scratch

## Dependencies
`sklearn`, `nltk` (`names`, `wordnet`)

## Authors
[Haining Pan](https://github.com/hainingpan)

## Reference
Liu, Y. H. (2017). Python Machine Learning By Example. Packt Publishing Ltd.

