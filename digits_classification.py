# This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports

# Import datasets, classifiers and performance metrics
from sklearn import metrics
from itertools import product
from utils import preprocess_data, train_model, split_data, read_digits, train_test_dev_split, predict_and_eval, tune_hparams

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
          f"best_hparam={hparam}")





