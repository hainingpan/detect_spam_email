# Detect spam emails with naive Bayes
Construct a spam emails classifier using the naive Bayes from scratch

## Steps
1. Download data from [Enron spam emails](http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron{}.tar.gz) with labels
2. Preprocess the data by removing names, stopwords in the email, and lemmatize each word
3. Use the bag of words model and convert the text into a sparse matrix, which stores the frequency of words
4. Calculate the prior probability and likelihood
5. Calculate the posterior probability given on the particular pattern of the test sample
6. Split train and test data, validate using confusion matrix, F1 score, the AUC of ROC
7. Test spam and nonspam emails manually copied from personal email
8. Use k-fold cross-validation to fine-tune the parameters (number of max features, Laplace smoothing, etc.)

## Dependencies
`sklearn`,`nltk`

## Authors
[Haining Pan](https://github.com/hainingpan)

## Reference
Liu, Y. H. (2017). Python Machine Learning By Example. Packt Publishing Ltd.

