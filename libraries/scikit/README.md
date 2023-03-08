# Scikit

Type of objects:

- Estimators: estimate some parameters based on a dataset, i.e, Imputer
  - hyperparameters are available via public instance variables (e.g. `imputer.strategy`) or learned parameters (e.g. `imputer.statistics_`)
- Transformers: estimator that can trasform dataset
  - see [Preprocessing and Normalization](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)
- Predictor: estimator that given dataset can do predictions, i.e. Regression

## Transformer

Notes:

- for regresion use [TransformedTargetRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html) to automatically transform back
