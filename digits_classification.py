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
from utils import preprocess_data, train_model, split_data, read_digits, train_test_dev_split, predict_and_eval, tune_hparams, get_digits_len_size, split_train_dev_test
from joblib import load



test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

x,y = read_digits()

print("Total number of samples : ", len(x))

print("(number of samples,length of image,height of image) is:",x.shape)

test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

for test_size in test_sizes:
    for dev_size in dev_sizes:
        # 3. Data splitting
        X_train, X_test,X_dev, y_train, y_test,y_dev = split_train_dev_test(x, y, test_size=test_size, dev_size=dev_size);

        # 4. Data Preprocessing
        X_train = preprocess_data(X_train)
        X_test = preprocess_data(X_test)
        X_dev = preprocess_data(X_dev)

        gama_ranges = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        C_ranges = [0.1,1,2,5,10]
        list_of_all_param_combination = [{'gamma': gamma, 'C': C} for gamma in gama_ranges for C in C_ranges]

        # Predict the value of the digit on the test subset
        # 6.Predict and Evaluate
        best_hparams, best_model_path, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination)
        best_model = load(best_model_path)


        accuracy_test = predict_and_eval(best_model, X_test, y_test)
        accuracy_dev = predict_and_eval(best_model, X_dev, y_dev)
        accuracy_train = predict_and_eval(best_model, X_train, y_train)
        print(f"test_size={test_size} dev_size={dev_size} train_size={1- (dev_size+test_size)} train_acc={accuracy_train} dev_acc={accuracy_dev} test_acc={accuracy_test}")
        print(f"best_gamma={best_hparams['gamma']},best_C={best_hparams['C']}")



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



