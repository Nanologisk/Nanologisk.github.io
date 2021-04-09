---
layout: article
title: Using Scikit-learn
permalink: /datascience/scikitlearn
key: datascience-scikitlearn
modify_date: 2021-04-08
pageview: false
sidebar:
    nav: datascience
---

Python Scikit-learn, from Datacamp.

<!--more-->

## Classification
### 1. K-nearest neighbors fit
k-Nearest Neighbors: Fit Having explored the Congressional voting
records dataset, it is time now to build your first classifier.

In this exercise, you will fit a k-Nearest Neighbors classifier to the
voting dataset, which has once again been pre-loaded for you into a
DataFrame df. It is importance to ensure your data adheres to the format
required by the scikit-learn API. The features need to be in an array
where each column is a feature and each row a different observation or
data point - in this case, a Congressman's voting record. The target
needs to be a single column with the same number of observations as the
feature data. We have done this for you in this exercise. Notice we
named the feature array X and response variable `y`: This is in accordance
with the common scikit-learn practice.

Your job is to create an instance of a k-NN classifier with 6 neighbors
(by specifying the n_neighbors parameter) and then fit it to the data.
The data has been pre-loaded into a DataFrame called `df`.

#### INSTRUCTIONS
- Import KNeighborsClassifier from sklearn.neighbors.
- Create arrays `X` and `y` for the features and the target variable. Here
  this has been done for you. Note the use of `.drop()` to drop the target
  variable `'party'` from the feature array X as well as the use of the
  `.values` attribute to ensure `X` and `y` are NumPy arrays. Without using
  `.values`, X and y are a DataFrame and Series respectively; the
  scikit-learn API will accept them in this form also as long as they
  are of the right shape.
- Instantiate a `KNeighborsClassifier` called knn
  with `6` neighbors by specifying the `n_neighbors` parameter.
- Fit the
  classifier to the data using the `.fit()` method.

```py
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)
```

### 2. k-nearest neighbors predict
k-Nearest Neighbors: Predict Having fit a `k-NN classifier`, you can now
use it to predict the label of a new data point. However, there is no
unlabeled data available since all of it was used to fit the model! You
can still use the `.predict()` method on the `X` that was used to fit the
model, but it is not a good indicator of the model's ability to
generalize to new, unseen data.

For now, a random unlabeled data point has been generated and is
available to you as `X_new`. You will use your classifier to predict the
label for this new data point, as well as on the training data `X` that
the model has already seen. Using `.predict()` on `X_new` will generate
1 prediction, while using it on `X` will generate 435 predictions: 1 for
each sample.

The DataFrame has been pre-loaded as df. This time, you will create the
feature array `X` and target variable array `y` yourself.

#### INSTRUCTIONS
Create arrays for the features and the target variable from df. As a
reminder, the target variable is `'party'`. Instantiate a
`KNeighborsClassifier` with `6` neighbors. Fit the classifier to the data.
Predict the labels of the training data, `X`. Predict the label of the new
data point `X_new`.

```py
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))
```

### 3. The Digits recognation dataset
The digits recognition dataset Up until now, you have been performing
binary classification, since the target variable had two possible
outcomes.

In the following exercises, you'll be working with the `MNIST` digits
recognition dataset, which has `10` classes, the digits `0` through `9`! A
reduced version of the `MNIST` dataset is one of scikit-learn's included
datasets, and that is the one we will use in this exercise. Each sample
in this scikit-learn dataset is an `8x8` image representing a handwritten
digit. Each pixel is represented by an integer in the range 0 to 16,
indicating varying levels of black.

Recall that scikit-learn's built-in datasets are of type Bunch, which
are dictionary-like objects. Helpfully for the MNIST dataset,
scikit-learn provides an `'images'` key in addition to the `'data'` and
`'target'` keys that you have seen with the Iris data. Because it is a 2D
array of the images corresponding to each sample, this `'images'` key is
useful for visualizing the images, as you'll see in this exercise (for
more on plotting 2D arrays, see Chapter 2 of DataCamp's course on Data
Visualization with Python).

On the other hand, the 'data' key contains the feature array - that is,
the images as a flattened array of 64 pixels. Notice that you can access
the keys of these Bunch objects in two different ways: By using the `.`
notation, as in `digits.images`, or the `[]` notation, as in
`digits['images']`. For more on the MNIST data, check out this exercise in
Part 1 of DataCamp's Importing Data in Python course. There, the full
version of the `MNIST` dataset is used, in which the images are `28x28`. It
is a famous dataset in machine learning and computer vision, and
frequently used as a benchmark to evaluate the performance of a new
model.

#### INSTRUCTIONS
- Import datasets from `sklearn` and `matplotlib.pyplot` as `plt`.
- Load the `digits` dataset using the `.load_digits()` method on datasets.
- Print the keys and `DESCR` of digits.
- Print the shape of images and data keys using the `.` notation.
- Display the 1010th image using `plt.imshow()`. This has been done for
  you, so hit `'Submit Answer'` to see which handwritten digit this
  happens to be!

```py
# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)
```



### 4. Train/Test Split and Fit/Predict/Accuracy

Now that you have learned about the importance of splitting your data
into training and test sets, it's time to practice doing this on the
`digits` dataset! After creating arrays for the features and target
variable, you will split them into `training` and `test` sets, fit a `k-NN
classifier` to the `training data`, and then compute its accuracy using the
`.score()` method.

#### INSTRUCTIONS
- Import `KNeighborsClassifier` from `sklearn.neighbors` and
`train_test_split` from `sklearn.model_selection`.
- Create an array for the features using `digits.data` and an array for
  the target using `digits.target`.
- Create stratified training and test sets using `0.2` for the size of
  the test set (`test_size`). Use a `random state` of `42`. Stratify the
  split according to the labels so that they are distributed in the
  training and test sets as they are in the original dataset.
- Create a k-NN classifier with `7` neighbors and fit it to the training
  data.
- Compute and print the accuracy of the classifier's predictions
  using the `.score()` method.

```py
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

digits = datasets.load_digits()

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
```

### 5. Overfitting and underfitting

In this exercise, you will compute and plot the training and testing
accuracy scores for a variety of different neighbor values. By observing
how the accuracy scores differ for the training and testing sets with
different values of `k`, you will develop your intuition for overfitting
and underfitting. The training and testing sets are available to you in
the workspace as `X_train`, `X_test`, `y_train`, `y_test`. In addition,
`KNeighborsClassifier` has been imported from `sklearn.neighbors`.

#### INSTRUCTIONS
- Inside the for loop: Setup a `k-NNclassifier` with the number of
  neighbors equal to `k`.
- Fit the classifier with `k` neighbors to the training data.
- Compute accuracy scores the training set and test set separately using
  the `.score()` method and assign the results to the `train_accuracy` and
  `test_accuracy` arrays respectively.

```py
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
```

## Regression

### 1. Importing data for supervised learning
You will work with `Gapminder` data that we have consolidated into one
CSV file available in the workspace as '`gapminder.csv`'. Specifically,
your goal will be to use this data to predict the life expectancy in a
given country based on features such as the country's GDP, fertility
rate, and population. The dataset has been preprocessed. Since the
target variable here is quantitative, this is a regression problem.

To begin, you will fit a linear regression with just one feature:
'`fertility`', which is the average number of children a woman in a
given country gives birth to. In later exercises, you will use all the
features to build regression models. Before that, however, you need to
import the data and get it into the form needed by scikit-learn. This
involves creating feature and target variable arrays.

Furthermore, since you are going to use only one feature to begin with,
you need to do some reshaping using NumPy's `.reshape()` method. Don't
worry too much about this reshaping right now, but it is something you
will have to do occasionally when working with scikit-learn so it is
useful to practice.

#### INSTRUCTIONS
Import numpy and pandas as their standard aliases. Read the file
'`gapminder.csv`' into a DataFrame df using the `read_csv()` function.
Create array `X` for the '`fertility`' feature and array `y` for the '`life`'
target variable. Reshape the arrays by using the `.reshape()` method and
passing in `(-1, 1)`.

```py
# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

# To understand np reshape(), look here: https://www.mikulskibartosz.name/numpy-reshape-explained/

# explore the dataset
df.info()

# correlation matrix
df.corr()

# correlation between two variables
df["life"].corr(df["fertility"])
df['GDP'].corr(df['life'])

# summary statistics for a single variable
df["life"].describe()

# check type of a single column
df["fertility"].dtype
```

### 2. Fit & predict for regression
Now, you will fit a linear regression and predict life expectancy using
just one feature. In this exercise, you will use the '`fertility`' feature
of the `Gapminder` dataset. Since the goal is to predict life expectancy,
the target variable here is '`life`'. The array for the target variable
has been pre-loaded as y and the array for '`fertility`' has been
pre-loaded as `X_fertility`.

A scatter plot with '`fertility`' on the x-axis and '`life`' on the
y-axis has been generated. There is a strongly negative correlation, so
a linear regression should be able to capture this trend. Your job is to
fit a linear regression and then predict the life expectancy, overlaying
these predicted values on the plot to generate a regression line.

You will also compute and print the `R2` score using sckit-learn's
`.score()` method.

#### INSTRUCTIONS
- Import `LinearRegression` from `sklearn.linear_model`.
- Create a LinearRegression regressor called `reg`.
- Set up the prediction space to range from the minimum to the maximum
  of `X_fertility`.
- Fit the regressor to the data (`X_fertility` and `y`) and compute its
  predictions using the `.predict()` method and the prediction_space
  array.
- Compute and print the R2 score using the `.score()` method. Overlay
  the plot with your linear regression line.

```py
# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
# Set up the prediction space to range from the minimum to the maximum of X_fertility
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# np.linspace creates sequences of evenly spaced values within a defined interval: https://www.sharpsightlabs.com/blog/numpy-linspace/
np.linspace(start=0, stop=100, num=5) #gives 0, 25, 50, 75, 100

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()
```

### 3. Train/test split for regression
Train and test sets are vital to ensure that your supervised learning
model is able to generalize well to new data. This was true for
classification models, and is equally true for linear regression models.
In this exercise, you will split the `Gapminder` dataset into training and
testing sets, and then fit and predict a linear regression over all
features. In addition to computing the `R2` score, you will also compute
`the Root Mean Squared Error` (RMSE), which is another commonly used
metric to evaluate regression models. The feature array `X` and target
variable array y have been pre-loaded for you from the DataFrame `df`.

#### INSTRUCTIONS
- Import LinearRegression from `sklearn.linear_model`,
  `mean_squared_error` from `sklearn.metrics`, and `train_test_split`
  from `sklearn.model_selection`.
- Using `X` and `y`, create training and
  test sets such that 30% is used for testing and 70% for training. Use
  a random state of `42`. Create a linear regression regressor called
  `reg_all`, fit it to the training set, and evaluate it on the test
  set.
- Compute and print the `R2` score using the `.score()` method on the
  test set.
- Compute and print the `RMSE`. To do this, first compute the Mean
  Squared Error using the `mean_squared_error()` function with the
  arguments `y_test` and `y_pred`, and then take its square root using
  `np.sqrt()`.

```py
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
# To do this, first compute the Mean Squared Error using the mean_squared_error() function with the arguments y_test and y_pred, and then take its square root using np.sqrt().
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
```

### 4. 5-fold cross-validation
Cross-validation is a vital step in evaluating a model. It maximizes the
amount of data that is used to train the model, as during the course of
training, the model is not only trained, but also tested on all of the
available data.

In this exercise, you will practice 5-fold cross validation on the
`Gapminder` data. By default, scikit-learn's `cross_val_score()` function
uses `R2` as the metric of choice for regression. Since you are performing
5-fold cross-validation, the function will return `5` scores. Your job is
to compute these 5 scores and then take their average.

The DataFrame has been loaded as `df` and split into the feature/target
variable arrays `X` and `y`. The modules `pandas` and `numpy` have been imported
as `pd` and `np`, respectively.

#### INSTRUCTIONS
- Import `LinearRegression from `sklearn.linear_model` and `cross_val_score`
  from `sklearn.model_selection`.
- Create a linear regression regressor called `reg`. Use the
  `cross_val_score()` function to perform 5-fold cross-validation on `X` and
  `y`.
- Compute and print the average cross-validation score. You can use
  NumPy's `mean()` function to compute the average.

```py
# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X,y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

# Compute and print the average cross-validation score. You can use NumPy's mean() function to compute the average.
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
```

### 5. K-Fold CV comparison
Cross validation is essential but do not forget that the more folds you
use, the more computationally expensive cross-validation becomes.

In this exercise, your job is to perform 3-fold cross-validation and
then 10-fold cross-validation on the `Gapminder` dataset. In the IPython
Shell, you can use `%timeit` to see how long each 3-fold CV takes compared
to 10-fold CV by executing the following `cv=3` and `cv=10`:

`%timeit cross_val_score(reg, X, y, cv = ____)`

`pandas` and `numpy` are available in the workspace as `pd` and `np`. The
DataFrame has been loaded as `df` and the feature/target variable arrays
`X` and `y` have been created.

#### INSTRUCTIONS
- Import `LinearRegression` from `sklearn.linear_model` and
`cross_val_score` from `sklearn.model_selection.`
- Create a linear regression regressor called `reg`.
- Perform 3-fold CV and then 10-fold CV. Compare the resulting mean
  scores.

```py
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print(np.mean(cvscores_10))

# to see how long each 3-fold CV takes compared to 10-fold CV by executing the following cv=3 and cv=10:
%timeit cross_val_score(reg, X, y, cv=3)
%timeit cross_val_score(reg, X, y, cv=10)
```

### 6. Regularization I: Lasso regression
In the video, you saw how Lasso selected out the `'RM'` feature as
being the most important for predicting Boston house prices, while
shrinking the coefficients of certain other features to 0. Its ability
to perform feature selection in this way becomes even more useful when
you are dealing with data involving thousands of features.

In this exercise, you will fit a lasso regression to the `Gapminder` data
you have been working with and plot the coefficients. Just as with the
Boston data, you will find that the coefficients of some features are
shrunk to 0, with only the most important ones remaining. The feature
and target variable arrays have been pre-loaded as `X` and `y`.

#### INSTRUCTIONS
- Import `Lasso` from `sklearn.linear_model`.
- Instantiate a Lasso regressor with an alpha of `0.4` and specify
  `normalize=True`.
- Fit the regressor to the data and compute the coefficients using the
  `coef_` attribute.
- Plot the coefficients on the y-axis and column names on the x-axis.
  This has been done for you, so hit 'Submit Answer' to view the plot!

```py
# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()
```

### 7. Regularization II: Ridge regression
Lasso is great for feature selection, but when building regression
models, Ridge regression should be your first choice.

Recall that lasso performs regularization by adding to the loss function
a penalty term of the absolute value of each coefficient multiplied by
some alpha. This is also known as `L1` regularization because the
regularization term is the `L1` norm of the coefficients. This is not the
only way to regularize, however.

If instead you took the sum of the squared values of the coefficients
multiplied by some alpha - like in Ridge regression - you would be
computing the `L2` norm. In this exercise, you will practice fitting ridge
regression models over a range of different alphas, and plot
cross-validated `R2` scores for each, using this function that we have
defined for you, which plots the `R2` score as well as standard error for
each alpha:

```py
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
```

The motivation behind this exercise is for you to see how the R2 score
varies with different alphas, and to understand the importance of
selecting the right value for alpha. You'll learn how to tune alpha in
the next chapter.

#### INSTRUCTIONS
- Instantiate a Ridge regressor and specify `normalize=True`.
- Inside the `for` loop:
- + Specify the alpha value for the regressor to use.
  + Perform 10-fold cross-validation on the regressor with the specified
    alpha. The data is available in the arrays `X` and `y`.
  + Append the average and the standard deviation of the computed
    cross-validated scores. NumPy has been pre-imported for you as
    `np`.
- Use the `display_plot()` function to visualize the scores and standard
  deviations.

```py
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50) # generates 50 numbers, first 10^(-4), last 10^0.
ridge_scores = []
ridge_scores_std = []

# np.logspace(start, stop, num, base). Start: base^start. Stop: base^stop. Base= base- by default, base=10. num: No. of samples to generate.

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)

# The plot:
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
```
