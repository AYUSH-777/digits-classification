import os.path

# Import datasets, classifiers and performance metrics
from sklearn import svm,datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from joblib import dump,load




#read gigits
def read_digits():
    digits = datasets.load_digits()
    x = digits.images
    y = digits.target
    return x,y

# We will define utils here :
def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets
def split_data(X,y,test_size=0.5,random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    return X_train, X_test, y_train, y_test

# Create a classifier: a support vector classifier
def train_model(X, y, model_params,model_type = 'svm'):
    if model_type == 'svm':
        clf = svm.SVC(**model_params)
    clf.fit(X, y)
    return clf


def split_data(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5,random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def train_test_dev_split(X, y, test_size, dev_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    X_train_dev, X_dev, y_train_dev, y_dev = train_test_split(X_train, y_train, test_size=dev_size / (1 - test_size), random_state=1)
    return X_train_dev, X_test, X_dev, y_train_dev, y_test, y_dev

def split_train_dev_test(X, y, test_size, dev_size):
    X_train_dev, X_test, y_train_dev, y_test = split_data(X, y, test_size=test_size)
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, y_train_dev, test_size=dev_size/(1-test_size))
    return X_train, X_test,X_dev, y_train, y_test, y_dev


def predict_and_eval(model, X, y):

    predicted = model.predict(X)
    accuracy = accuracy_score(y, predicted)

    return accuracy


def tune_hparams(X_train, Y_train, X_dev, y_dev, list_of_all_param_combination):
    best_accuracy_so_far = -1
    best_model = None
    best_model_path = ""

    for param_combination in list_of_all_param_combination:
        cur_model = train_model(X_train, Y_train, {'gamma': param_combination['gamma'],'C':param_combination['C']}, model_type='svm')
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)
        if cur_accuracy > best_accuracy_so_far:
            best_accuracy_so_far = cur_accuracy
            optimal_gamma = param_combination['gamma']
            optimal_C = param_combination['C']
            best_hparams = {'gamma': optimal_gamma,'C':optimal_C}
            # Replace colons with underscores in the filename
            best_model_path = "./models/best_model" + "_".join(["{}:{}".format(k,v) for k,v in best_hparams.items()]).replace(":", "_") + ".joblib"
            best_model = cur_model

    # save the best model
    dump(best_model, best_model_path)
    return best_hparams, best_model_path, best_accuracy_so_far


def get_digits_len_size():
    digits = datasets.load_digits()
    x = digits.images
    y = digits.target
    print("\n")
    print("2.1 The number of total samples in the dataset:", len(x))
    print("2.2 Size (height and width) of the images in dataset:", x[0].shape)  # Assuming all images have the same size
    return x, y