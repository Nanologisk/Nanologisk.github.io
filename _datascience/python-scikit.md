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


## Fine-tuning your model
### 1. Metrics for classification
In Chapter 1, you evaluated the performance of your k-NN classifier
based on its accuracy. However, as Andy discussed, accuracy is not
always an informative metric. In this exercise, you will dive more
deeply into evaluating the performance of binary classifiers by
computing a confusion matrix and generating a classification report.

You may have noticed in the video that the classification report
consisted of three rows, and an additional *support* column. The
*support* gives the number of samples of the true response that lie in
that class - so in the video example, the support was the number of
Republicans or Democrats in the test set on which the classification
report was computed. The *precision*, *recall*, and *f1-score columns*,
then, gave the respective metrics for that particular class.

Here, you'll work with the `PIMA Indians` dataset obtained from the UCI
Machine Learning Repository. The goal is to predict whether or not a
given female patient will contract diabetes based on features such as
BMI, age, and number of pregnancies. Therefore, it is a binary
classification problem. A target value of `0` indicates that the patient
does not have diabetes, while a value of `1` indicates that the patient
does have diabetes. As in Chapters 1 and 2, the dataset has been
preprocessed to deal with missing values.

The dataset has been loaded into a DataFrame `df` and the feature and
target variable arrays `X` and `y` have been created for you. In addition,
`sklearn.model_selection.train_test_split` and
`sklearn.neighbors.KNeighborsClassifier` have already been imported.

Your job is to train a k-NN classifier to the data and evaluate its
performance by generating a confusion matrix and classification report.

#### INSTRUCTIONS
- Import `classification_report` and `confusion_matrix` from
  `sklearn.metrics`.
- Create training and testing sets with `40%` of the data used for
  testing. Use a random state of `42`.
- Instantiate a k-NN classifier with `6` neighbors, fit it to the
  training data, and predict the labels of the test set.
- Compute and print the confusion matrix and classification report using
  the `confusion_matrix()` and `classification_report()` functions.

```py
# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 2. Building a logistic regression model
Scikit-learn makes it very easy to try different models, since the
Train-Test-Split/Instantiate/Fit/Predict paradigm applies to all
classifiers and regressors - which are known in scikit-learn as
'estimators'. You'll see this now for yourself as you train a logistic
regression model on exactly the same data as in the previous exercise.

The feature and target variable arrays `X` and `y` have been pre-loaded, and
`train_test_split` has been imported for you from `sklearn.model_selection`.

#### INSTRUCTIONS
- Import: LogisticRegression from
  `sklearn.linear_model.confusion_matrix` and `classification_report`
  from `sklearn.metrics`.
- Create training and test sets with 40% (or `0.4`) of the data used for
  testing. Use a random state of `42`.
- Instantiate a `LogisticRegression` classifier called `logreg`.
- Fit the classifier to the training data and predict the labels of the
  test set.
- Compute and print the confusion matrix and classification
  report. This has been done for you, so hit 'Submit Answer' to see how
  logistic regression compares to k-NN!

```py
# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 3. Plotting an ROC curve
Classification reports and confusion matrices are great methods to
quantitatively evaluate model performance, while ROC curves provide a
way to visually evaluate models. Most classifiers in scikit-learn have a
`.predict_proba()` method which returns the probability of a given
sample being in a particular class. Having built a logistic regression
model, you'll now evaluate its performance by plotting an ROC curve. In
doing so, you'll make use of the `.predict_proba()` method and become
familiar with its functionality.

Here, you'll continue working with the `PIMA` Indians diabetes dataset.
The classifier has already been fit to the training data and is
available as logreg.

#### INSTRUCTIONS
- Import `roc_curve` from `sklearn.metrics`.
- Using the `logreg` classifier, which has been fit to the training
  data, compute the predicted probabilities of the labels of the test
  set `X_test`. Save the result as `y_pred_prob`.
- Use the `roc_curve()` function with `y_test` and `y_pred_prob` and unpack
  the result into the variables `fpr`, `tpr`, and `thresholds`.
- Plot the ROC curve with `fpr` on the x-axis and `tpr` on the y-axis.

```py
# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

### 4. Precision-recall Curve
When looking at your ROC curve, you may have noticed that the y-axis
(True positive rate) is also known as recall. Indeed, in addition to the
ROC curve, there are other ways to visually evaluate model performance.
One such way is the precision-recall curve, which is generated by
plotting the precision and recall for different thresholds. As a
reminder, precision and recall are defined as:

precision = TP/(TP+FP)

Recall = TP/(TP+FN)

Study the precision-recall curve and then consider the statements given
below. Choose the one statement that is not true. Note that here, the
class is positive (1) if the individual has diabetes.
- A recall of 1 corresponds to a classifier with a low threshold in
  which all females who contract diabetes were correctly classified as
  such, at the expense of many misclassifications of those who did not
  have diabetes.
- Precision is undefined for a classifier which makes no positive
  predictions, that is, classifies everyone as not having diabetes.
- When the threshold is very close to 1, precision is also 1, because
  the classifier is absolutely certain about its predictions. (False)
  :X:
- Precisio and recall take true negatives into consideration.


### 5. AUC computation
Say you have a binary classifier that in fact is just randomly making
guesses. It would be correct approximately 50% of the time, and the
resulting ROC curve would be a diagonal line in which the True Positive
Rate and False Positive Rate are always equal. The Area under this ROC
curve would be 0.5. This is one way in which the AUC, which Hugo
discussed in the video, is an informative metric to evaluate a model. If
the AUC is greater than 0.5, the model is better than random guessing.
Always a good sign!

In this exercise, you'll calculate AUC scores using
the `roc_auc_score()` function from `sklearn.metrics` as well as by
performing cross-validation on the `diabetes` dataset. `X` and `y`, along with
training and test sets `X_train`, `X_test`, `y_train`, `y_test`, have been
pre-loaded for you, and a logistic regression classifier `logreg` has been
fit to the training data.

#### Instructions
- Import `roc_auc_score` from `sklearn.metrics` and `cross_val_score` from
  `sklearn.model_selection`.
- Using the `logreg` classifier, which has been fit to the training
  data, compute the predicted probabilities of the labels of the test
  set `X_test`. Save the result as `y_pred_prob`.
- Compute the AUC score using the `roc_auc_score()` function, the test set
  labels `y_test`, and the predicted probabilities `y_pred_prob`.
- Compute the AUC scores by performing 5-fold cross-validation. Use the
  `cross_val_score()` function and specify the scoring parameter to be
  `'roc_auc'`.

```py
# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc - this is another way of doing the same thing.
cv_auc = cross_val_score(logreg, X, y, cv=5,
scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))
```

### 6. Hyperparameter tuning with GridSearchCV
Hugo demonstrated how to use to tune the `n_neighbors` parameter of the
`KNeighborsClassifier()` using `GridSearchCV` on the voting dataset. You
will now practice this yourself, but by using logistic regression on the
diabetes dataset instead!

Like the alpha parameter of lasso and ridge regularization that you saw
earlier, logistic regression also has a **regularization parameter**:
*C*. *C* controls the inverse of the regularization strength, and this
is what you will tune in this exercise. A large *C* can lead to an
overfit model, while a small *C* can lead to an underfit model.

The hyperparameter space for *C* has been setup for you. Your job is to
use `GridSearchCV` and `logistic regression` to find the optimal *C* in
this hyperparameter space. The feature array is available as `X` and
target variable array is available as `y`.

You may be wondering why you aren't asked to split the data into
training and test sets. Here, we want you to focus on the process of
setting up the hyperparameter grid and performing grid-search
cross-validation. In practice, you will indeed want to hold out a
portion of your data for evaluation purposes.

#### INSTRUCTIONS
- Import `LogisticRegression` from `sklearn.linear_model` and `GridSearchCV`
  from `sklearn.model_selection`.
- Setup the hyperparameter grid by using `c_space` as the grid of values
  to tune *C* over.
- Instantiate a logistic regression classifier called `logreg`.
- Use `GridSearchCV` with 5-fold cross-validation to tune C:
  + Inside `GridSearchCV()`, specify the classifier, parameter grid, and
    number of folds to use.
  + Use the `.fit()` method on the `GridSearchCV` object to fit it to
    the data `X` and `y`.
- Print the best parameter and best score obtained from `GridSearchCV` by
  accessing the `best_params_` and `best_score_` attributes of `logreg_cv`.

```py
# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))
```

### 7. Hyperparameter tuning with RandomizedSearchCV
`GridSearchCV` can belly expensive, especially if you are searching over a
large hyperparameter space and dealing with multiple hyperparameters. A
solution to this is to use `RandomizedSearchCV`, in which not all
hyperparameter values are tried out. Instead, a fixed number of
hyperparameter settings is sampled from specified probability
distributions. You'll practice using `RandomizedSearchCV` in this exercise
and see how this works.

Here, you'll also be introduced to a new model: the Decision Tree. Just
like k-NN, linear regression, and logistic regression, decision trees in
scikit-learn have `.fit()` and `.predict()` methods that you can use in
exactly the same way as before. Decision trees have many parameters that
can be tuned, such as `max_features`, `max_depth`, and `min_samples_leaf`:
This makes it an ideal use case for `RandomizedSearchCV`.

The feature array `X` and target variable array `y` of the diabetes
dataset have been pre-loaded. The hyperparameter settings have been
specified for you. Your goal is to use `RandomizedSearchCV` to find the
optimal hyperparameters.

#### INSTRUCTIONS
- Import `DecisionTreeClassifier` from `sklearn.tree` and `RandomizedSearchCV`
  from `sklearn.model_selection`.
- Specify the parameters and distributions to sample from.
- Instantiate a `DecisionTreeClassifier`. Use `RandomizedSearchCV` with
  5-fold cross-validation to tune the hyperparameters:
  + Inside `RandomizedSearchCV()`, specify the classifier, parameter
    distribution, and number of folds to use.
  + Use the `.fit()` method on the `RandomizedSearchCV` object to fit it
    to the data `X` and `y`.
  + Print the best parameter and best score obtained from
    `RandomizedSearchCV` by accessing the `best_params_` and `best_score_`
    attributes of `tree_cv`.

```py
# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
```

### 8. Hold-out set in practice I: Classification
You will now practice evaluating a model with tuned hyperparameters on a
hold-out set. The feature array and target variable array from the
diabetes dataset have been pre-loaded as `X` and `y`.

In addition to *C*, logistic regression has a `'penalty'` hyperparameter
which specifies whether to use `'l1'` or `'l2'` regularization. Your job in
this exercise is to create a hold-out set, tune the `'C'` and `'penalty'`
hyperparameters of a logistic regression classifier using `GridSearchCV`
on the training set, and then evaluate its performance against the
hold-out set.

#### INSTRUCTIONS
- Create the hyperparameter grid:
  + Use the array c_space as the grid of values for 'C'.
  +  For 'penalty', specify a list consisting of 'l1' and 'l2'.
- Instantiate a logistic regression classifier. Create training and test
  sets. Use a `test_size` of `0.4` and `random_state` of `42`. In practice, the
  test set here will function as the hold-out set.
- Tune the hyperparameters on the training set using `GridSearchCV` with
  5-folds. This involves first instantiating the `GridSearchCV` object
  with the correct parameters and then fitting it to the training data.
- Print the best parameter and best score obtained from
  `GridSearchCV` by accessing the `best_params_` and `best_score_` attributes
  of `logreg_cv`.
-
```py
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))
```

### 8. Hold-out set in practice II: Regression
Remember lasso and ridge regression from the previous chapter? Lasso
used the *L1* penalty to regularize, while ridge used the *L2* penalty.
There is another type of regularized regression known as the elastic
net. In elastic net regularization, the penalty term is a linear
combination of the L1 and L2 penalties:

a∗L1+b∗L2

In scikit-learn, this term is represented by the `'l1_ratio'` parameter:
An `'l1_ratio'` of 1 corresponds to an L1 penalty, and anything lower is
a combination of *L1* and *L2*.

In this exercise, you will `GridSearchCV` to tune the `'l1_ratio'`
of an elastic net model trained on the Gapminder data. As in the
previous exercise, use a hold-out set to evaluate your model's
performance.

#### INSTRUCTIONS
- Import the following modules:
  + `ElasticNet` from `sklearn.linear_model`.
  + `mean_squared_error` from `sklearn.metrics`.
  + `GridSearchCV` and `train_test_split` from
    `sklearn.model_selection`.
- Create training and test sets, with 40% of the data used for the test
  set. Use a random state of `42`.
- Specify the hyperparameter grid for `'l1_ratio'` using `l1_space` as the
  grid of values to search over.
- Instantiate the `ElasticNet` regressor.
- Use `GridSearchCV` with 5-fold cross-validation to tune `'l1_ratio'` on
  the training data `X_train` and `y_train`. This involves first
  instantiating the `GridSearchCV` object with the correct parameters and
  then fitting it to the training data.
- Predict on the test set and compute the R2 and mean squared error.

```py
# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))
```
