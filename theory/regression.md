# Regression algorithms

## Linear

### Ordinary Least Squares

[LinearRegression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) fits a linear model with coefficients.

Notes:
- `LinearRegression` is using [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) that has complexity O(n^2). 
- Both Normal equation and SVD get very slow when number of features grows large. On the other hand both are linear with regard to the number of instances.
- More precise and efficient compared to "gradient" methods, thought not fine for "out-of-core" scenarios.
- Scaling is not required.
- `PolynomialRegression` can be handy for finding features correlations and using SVD for non-linear regression.

#### Ridge

[Ridge](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification) regression addresses some of the problems of Ordinary Least Squares by imposing a l2-norm penalty on the size of the coefficients.

Notes:
- Good for limiting parameters values, thought scalling is required.

#### Lasso

The [Lasso](https://scikit-learn.org/stable/modules/linear_model.html#lasso) is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to prefer solutions with fewer non-zero coefficients, effectively reducing the number of features upon which the given solution is dependent. For this reason, Lasso and its variants are fundamental to the field of compressed sensing. Under certain conditions, it can recover the exact set of non-zero coefficients.

#### Elastic-Net

[ElasticNet](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net) is a linear regression model trained with both l1 and l2-norm regularization of the coefficients. 

Notes:
- Useful when there are multiple features that are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.
- A practical advantage of trading-off between Lasso and Ridge is that it allows Elastic-Net to inherit some of Ridge’s stability under rotation.

### Gradient descent

Term | Definition
---|---
learning rate | Determine the size of the step at each iteration.
simulated annealing | Decrease learning rate with time, to reduce stochatic gradient randomness with time.
learning schedule | Function that determines the learning rate at each iteration


Kind | Definition | Notes
---|---|---
Batch | Calculate delta based on all training set | More stable, but also sensitive to local minima. Slow for large datasets.
Stochatic | Calculate delta for random instance at each step | Less regular, but more likely to find global minumum.
Mini Batch | Calculate delta based on small, randomly selected subset | 

Notes:
- Ensure that all features have a similar scale.
- Compared to `LinearRegression` and it's variants, this scales well with large number of features.

#### Stochatic Gradient Descent

The class [SGDRegressor](https://scikit-learn.org/stable/modules/sgd.html#regression) implements a plain stochastic gradient descent learning routine which supports different loss functions and penalties to fit linear regression models. 

## Nearest Neighbors Regression

[Neighbors-based regression](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression) can be used in cases where the data labels are continuous rather than discrete variables. The label assigned to a query point is computed based on the mean of the labels of its nearest neighbors.
