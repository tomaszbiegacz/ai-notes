# Regression algorithms

## Linear

Start at [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html).

### Ordinary least squares

[LinearRegression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) fits a linear model with coefficients.

### Ridge

[Ridge](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification) regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of the coefficients.

### Lasso

The [Lasso](https://scikit-learn.org/stable/modules/linear_model.html#lasso) is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer non-zero coefficients, effectively reducing the number of features upon which the given solution is dependent. For this reason, Lasso and its variants are fundamental to the field of compressed sensing. Under certain conditions, it can recover the exact set of non-zero coefficients.

### Elastic-Net

[ElasticNet](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net) is a linear regression model trained with both l1 and l2-norm regularization of the coefficients. This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge. We control the convex combination of l1 and l2 using the l1_ratio parameter.

Elastic-net is useful when there are multiple features that are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.

A practical advantage of trading-off between Lasso and Ridge is that it allows Elastic-Net to inherit some of Ridge’s stability under rotation.

### Stochatic gradient descent

The class [SGDRegressor](https://scikit-learn.org/stable/modules/sgd.html#regression) implements a plain stochastic gradient descent learning routine which supports different loss functions and penalties to fit linear regression models. SGDRegressor is well suited for regression problems with a large number of training samples (> 10.000), for other problems we recommend Ridge, Lasso, or ElasticNet.

## Nearest Neighbors Regression

[Neighbors-based regression](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression) can be used in cases where the data labels are continuous rather than discrete variables. The label assigned to a query point is computed based on the mean of the labels of its nearest neighbors.
