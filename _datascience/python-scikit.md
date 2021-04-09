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
### 1. K -nearest neighbors fit
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
