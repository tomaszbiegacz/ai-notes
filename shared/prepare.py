from pandas.plotting import scatter_matrix
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, StandardScaler

import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings

#
# generics
# move to dedicated module
#

def assert_isDataFrame(X):
  assert isinstance(X, pd.DataFrame)

#
# I/O
#

def loadData_csv_pandas(fileName):
  file_path = Path("shared/data/" + fileName)
  return pd.read_csv(file_path)

def buildFileName(setName, variableName):
  return setName + "-" + variableName


def dumpNp(setName, variableName, value):     
  fileName = buildFileName(setName, variableName) + ".npy"
  np.save(fileName, value)

def loadNp(setName, variableName, buildArray=None, forceRebuild=False):
  fileName = buildFileName(setName, variableName) + ".npy"
  if forceRebuild or not os.path.isfile(fileName):
    if buildArray is None:
      raise f'Cannot find array [{fileName}], specify "buildArray" parameter'
    else:
      np.save(fileName, buildArray())
  return np.load(fileName, allow_pickle=True)


def dumpModel(setName, modelName, model):
  fileName = buildFileName(setName, modelName) + ".pkl"
  joblib.dump(model, fileName)

def loadModel(setName, modelName, buildModel=None, forceRebuild=False):
  fileName = buildFileName(setName, modelName) + ".pkl"
  if forceRebuild or not os.path.isfile(fileName):
    if buildModel is None:
      raise f'Cannot find model [{fileName}], specify "buildModel" parameter'
    else:
      joblib.dump(buildModel(), fileName)
  return joblib.load(fileName)

#
# first look
#

def columns_printTable(data):
  column_name = 'Attribute'
  print(f'{column_name:25}| other')
  print('---|---')
  for column_name in data.columns:
    print(f'{column_name:25}|')

def column_describe(X, column_name):
  assert_isDataFrame(X)
  column = X[column_name]
  dtype = X.dtypes[column_name]
  result = [f'{dtype.name:8}']

  if dtype.kind in ['f']:
    na_count = column.isna().sum()
    if na_count > 0:
      result.append(f'nan_count={na_count}')
  else:
    null_count = column.isnull().sum()
    if null_count > 0:
      result.append(f'null_count={null_count}')

  if dtype.kind in ['i', 'u', 'O']:
    result.append(f'unique_count={column.unique().size}')

  if dtype.kind in ['i', 'u', 'f', 'm', 'M']:
    result.append(f'min={column.min()}')
    result.append(f'max={column.max()}')
      
  return ' '.join(result)

def dataFrame_describe(X):
  assert_isDataFrame(X)  
  print(f'rows_count = {X.shape[0]}')
  columns_names = sorted(X.columns)
  print(f'columns = {columns_names}')
  print('---')
  for column_name in columns_names:
    print(f'{column_name:25}: {column_describe(X, column_name)}')

#
# split
#

def split_distr_bin(data, column_name, test_size, distr_bins, random_state=None):
  """" split data into (train, test) preserving data distribution in given column accourding to bins """
  assert_isDataFrame(data) 
  distr_data = pd.cut(data[column_name], bins=distr_bins)
  train, test = train_test_split(data, test_size=test_size, stratify=distr_data, random_state=random_state)
  return (train.sort_index(axis=1), test)

def split_distr(data, column_name, test_size, distr_gran=10, random_state=None):
  """" split data into (train, test) preserving data distribution in given column according to quantiles"""
  assert_isDataFrame(data)
  distr_bins = np.append(
    np.append(
      [-1 * np.inf], 
      data[column_name].quantile(q=np.arange(0, 1, 1.0/distr_gran)).to_numpy()[1:]), 
    [np.inf])
  return split_distr_bin(data, column_name, test_size, distr_bins, random_state)

def split_stratify(data, column_name, test_size, random_state=None):
  """" split data into (train, test) preserving data distribution in given column according to quantiles"""
  assert_isDataFrame(data)
  train, test = train_test_split(data, test_size=test_size, stratify=data[column_name], random_state=random_state)
  return (train.sort_index(axis=1), test)  

#
# visualisation
#

class ClusterSimilarity(BaseEstimator, TransformerMixin):
  def __init__(self, n_clusters, gamma=1.0, random_state=None):
    self.n_clusters = n_clusters
    self.gamma = gamma
    self.random_state=random_state

  def fit(self, X, y=None, sample_weight=None):
    self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state, n_init='auto')
    self.kmeans_.fit(X, sample_weight=sample_weight)
    return self

  def transform(self, X):
    return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

  def get_feature_names_out(self, names=None):
    return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

#
# correlations
#

def correlations_describe(data, columns_names=None, corr_from=0.6, figsize=None):
  """" print correnaltion table for selected columns """
  assert_isDataFrame(data)
  if columns_names is None:
    columns_names = data.columns

  for r in range(0, len(columns_names)):
    c1 = columns_names[r]
    for c in range(0, r):      
      c2 = columns_names[c]
      corr = data[c1].corr(data[c2])
      if abs(corr) >= corr_from:      
        print(f'{c1:25} {c2:25} | {corr:5.2} {data[c2].corr(data[c1]):5.2}')

  if not figsize is None: 
    scatter_matrix(data[columns_names], figsize=figsize)

#
# transformations
#

def get_generic_transformer_names():
  return ["log", "exp", "sqrt", "sqr", "arctan", "yeo-johnson", "box-cox"]


def validate_transformer_usage(name, X):
  """ Return validation error message or None """
  
  # only pandas DataFrame is now supported
  assert_isDataFrame(X)

  if not (name in get_generic_transformer_names()):
    return f'Unknown transformer: [{name}]'

  if X.isnull().values.any():
    return "Use imputer before transformation"

  if name in ["log", "box-cox"]:
    if X.min().min() <= 0:
      return "data set needs to be positive"

  if name in ["sqrt"]:
    if X.min().min() < 0:
      return "data set cannot be negative"

  return None 


def create_transformer(name):
  match name:
    case "log":
      return FunctionTransformer(np.log, inverse_func=np.exp)
    case "exp":
      return FunctionTransformer(np.exp, inverse_func=np.log)
    case "sqrt":
      return FunctionTransformer(np.sqrt, inverse_func=lambda X: X ** 2)
    case "sqr":
      return FunctionTransformer(lambda X: X ** 2, inverse_func=np.sqrt)
    case "arctan":
      return FunctionTransformer(np.arctan, inverse_func=np.tan)
    case "yeo-johnson":
      return PowerTransformer(method="yeo-johnson")
    case "box-cox":
      return PowerTransformer(method="box-cox")
    case _:
      raise f'Unknown transformer: [{name}]'

def get_nrows(nitems, ncols):
  nrows = nitems // ncols
  if nitems % ncols > 0:
    return nrows + 1
  else:
    return nrows

def transformations_compare_column(X, ncols=4, fig_size=(4, 4), nbins=50):
  assert_isDataFrame(X)
  assert len(X.columns) == 1
  metric_name = X.columns[0]

  transformation_names = get_generic_transformer_names()
  validation_errors = list(map(lambda name: validate_transformer_usage(name, X), transformation_names))
  figs_count = np.sum(list(map(lambda x: x is None, validation_errors))) + 1
      
  figs_ncols = np.min([figs_count, ncols])
  figs_nrows = get_nrows(figs_count, figs_ncols)
  figs_size = (fig_size[0] * figs_ncols, fig_size[1] * figs_nrows)
  fig, axs = plt.subplots(ncols=figs_ncols, nrows=figs_nrows, figsize=figs_size)
  fig.suptitle(metric_name)

  if figs_count == 1:
    axs = [ axs ]
  elif figs_count > figs_ncols:
    axs = axs.flatten()

  axs[0].hist(X, bins=nbins);

  axis_pos = 1
  for i in range(len(transformation_names)):
    transformation_name = transformation_names[i]
    if validation_errors[i] is None:
      ax = axs[axis_pos]
      axis_pos += 1

      result = create_transformer(transformation_name).fit_transform(X)
      ax.hist(result, bins=nbins);
      ax.set_title(transformation_name)

  for i in range(axis_pos, len(axs)):
    axs[i].axis('off')
  
  plt.show()
  for i in range(len(transformation_names)):
    if not validation_errors[i] is None:
      warnings.warn(f'{transformation_names[i]}: {validation_errors[i]}')

def transformations_compare(X, except_columns=[]):
  assert_isDataFrame(X)
  for column_name in filter(lambda c: not c in except_columns, X.columns):
    transformations_compare_column(X[[column_name]])

def make_standard(model):
  return make_pipeline(StandardScaler(), model)

#
# estimations
#

def loadCross_pred(setName, variableName, estimator, X, y, cv=None, n_jobs=-1, forceRebuild=False):
  return loadNp(
    setName, variableName, 
    lambda: cross_val_predict(estimator, X, y, cv=cv, n_jobs=n_jobs, method="predict"),
    forceRebuild=forceRebuild)

def loadCross_decisionFunction(setName, variableName, estimator, X, y, cv=None, n_jobs=-1, forceRebuild=False):
  return loadNp(
    setName, variableName, 
    lambda: cross_val_predict(estimator, X, y, cv=cv, n_jobs=n_jobs, method="decision_function"),
    forceRebuild=forceRebuild)
