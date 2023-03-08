# Classification algorithms

term | meaning
---|---
skewed dataset | when some classes are much more frequent than others

## Performance

term | meaning
---|---
TP (true positive) | number of times, when positive classification was correct
FP (false positive) | number of times, when false classification was correct
FN (false negative) | number of times, when false classification was *not* correct
precision | TP / (TP + FP) = how many mistakes?
recall (TPR, true positive rate, sensitivity) | TP / (TP + FN) = how many correct found?
fall-out (FPR, false positive rate) | FP / (TP + FP) = how many incorrect found?
TNR (true negative rate, specify) | FPR = 1 - TNR 
ROC | Recall fall-Out Curve

Notes:
- prefer precision/recall over ROC when positive class is rare or when you care more about the FP vs FN 

## Multiclass

Strategy | Name | Description | Notes
---|---|---|---
OvR | one-versus-rest <br> Known also as OvA (one-versus-all) | Train one binary classifier for each class | Usually faster.
OvO | one-versus-one | Train one binary classifier for each pair of classes, `N*(N-1)/2` | Gives greater confidense and works better when classifier does not scale well with the size of the training data.

## Scikit classifiers

Classifier | Description | Support for multiple classes | Notes
---|---|---|---
[GausianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) | Gaussian Naive Bayes | Yes |
[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | Logistic Regression (aka logit, MaxEnt) classifier. | Yes |
[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | | Yes | 
[SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) | Linear classifiers (SVM, logistic regression, etc.) | No |
[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) | C-Support Vector Classification. | No |
