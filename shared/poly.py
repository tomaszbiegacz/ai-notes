from sklearn.metrics.pairwise import manhattan_distances
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PolynomialDataset:

  def __init__(self, variablesCount, maxPower):
    fileName = f'003_poly_{variablesCount}_{maxPower}.h5'
    self.model = pd.read_hdf(fileName, "model")
    self.dataset = pd.read_hdf(fileName, "data")
    self.X = self.dataset.drop("result", axis=1)
    self.y = self.dataset["result"].copy()

def manhattan_linearModel(X, y, coef, intercept):
  actual = np.dot(X, np.transpose(coef)) + intercept
  return manhattan_distances([actual], [y.to_numpy()])[0][0]

def LinearRegression_check(X, y):
  lin_reg = LinearRegression().fit(X, y)
  print(f'a: {lin_reg.coef_}, b: {lin_reg.intercept_}')
  print(f'manhattan distance: {manhattan_linearModel(X, y, lin_reg.coef_, lin_reg.intercept_)}')

def PolynomialRegression_check(X, y, degree, include_bias=False):
  pX = PolynomialFeatures(degree=degree, include_bias=include_bias).fit_transform(X)
  lin_reg = LinearRegression().fit(pX, y)
  print(f'a: {lin_reg.coef_}, b: {lin_reg.intercept_}')
  print(f'manhattan distance: {manhattan_linearModel(pX, y, lin_reg.coef_, lin_reg.intercept_)}')

def SGDRegressor_check(X, y):
  sdg_reg = SGDRegressor(random_state=42).fit(X, y)
  print(f'a: {sdg_reg.coef_}, b: {sdg_reg.intercept_}, n_iter: {sdg_reg.n_iter_}')
  print(f'manhattan distance: {manhattan_linearModel(X, y, sdg_reg.coef_, sdg_reg.intercept_)}')

def learningCurve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), figs_size=(4, 4)):
  train_sizes, train_scores, valid_scores = learning_curve(
    model, X, y, 
    train_sizes=train_sizes, scoring="neg_root_mean_squared_error", n_jobs=-1)

  train_errors = -train_scores.mean(axis=1)
  valid_errors = -valid_scores.mean(axis=1)

  plt.figure(figsize=figs_size)
  plt.plot(train_sizes, train_errors, "r-+", label="train")
  plt.plot(train_sizes, valid_errors, "b-", label="valid")
  plt.legend()
  plt.ylabel("RMSE")
  plt.xlabel("training set size")
  plt.show()
