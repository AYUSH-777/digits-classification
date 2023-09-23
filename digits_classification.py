# This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports

# Import datasets, classifiers and performance metrics
from sklearn import metrics
import numpy as np
from skimage import transform
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from itertools import product
from utils import preprocess_data, train_model, split_data, read_digits, train_test_dev_split, predict_and_eval, tune_hparams, get_digits_len_size

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]

test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

X, y = read_digits()

for test_size, dev_size in product(test_sizes, dev_sizes):

    X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)

    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    X_dev = preprocess_data(X_dev)

    best_hparams, best_model, best_acc_so_far = tune_hparams(X_train, y_train, X_dev, y_dev, product(gamma_ranges, C_ranges))

    hparam = {'gamma':best_hparams[0],'C':best_hparams[1]}
    train_acc = predict_and_eval(best_model, X_train, y_train)
    dev_acc = predict_and_eval(best_model, X_dev, y_dev)
    test_acc = predict_and_eval(best_model, X_test, y_test)
    print(f"test_size={test_size} dev_size={dev_size} train_size={1 - test_size - dev_size} "
          f"train_acc={train_acc:.4f} dev_acc={dev_acc:.4f} test_acc={test_acc:.4f} "
          f"best_hparams={hparam}")



#### Q1 Implementation

print("Question 1 implementation for the Quiz - 1")

print("\n")
get_digits_len_size()


### Q3 Implementation

print("\n")
print("Question 3 implementation for the Quiz - 1")
X, y = read_digits()

# Define the image sizes to evaluate
image_sizes = [4, 6, 8]

print("\n")

# Loop through different image sizes
for size in image_sizes:
    # Resize the images
    resized_images = [transform.resize(image, (size, size)) for image in X]

    # Split the data into train, dev, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(resized_images, y, train_size=0.7, test_size=0.3, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, train_size=0.1, test_size=0.2, random_state=42)

    # Convert to NumPy arrays
    X_train_array = np.array(X_train).reshape(len(X_train), -1)
    X_dev_array = np.array(X_dev).reshape(len(X_dev), -1)
    X_test_array = np.array(X_test).reshape(len(X_test), -1)

    # Create and train the classifier (K-Nearest Neighbors in this case)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train_array, y_train)

    # Evaluate the classifier
    train_acc = accuracy_score(y_train, clf.predict(X_train_array))
    dev_acc = accuracy_score(y_dev, clf.predict(X_dev_array))
    test_acc = accuracy_score(y_test, clf.predict(X_test_array))

    # Print the results
    print(f"image size: {size}x{size} train_size: 0.7 dev_size: 0.1 test_size: 0.2 train_acc: {train_acc:.2f} dev_acc: {dev_acc:.2f} test_acc: {test_acc:.2f}")



