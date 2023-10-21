import os.path

# Import datasets, classifiers and performance metrics
from sklearn import svm,datasets
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from joblib import dump,load
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier  # Import the Decision Tree classifier





#read gigits
def read_digits():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    return X,y

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
    if model_type == 'tree':
        clf = tree.DecisionTreeClassifier(**model_params)
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


def tune_hparams(X_train, Y_train, X_dev, y_dev, list_of_all_param_combination, model_type='svm'):
    best_accuracy_so_far = -1
    best_model = None
    best_model_path = ""

    for param_combination in list_of_all_param_combination:
        if model_type == 'svm':
            cur_model = train_model(X_train, Y_train, {'gamma': param_combination['gamma'],'C':param_combination['C']}, model_type='svm')
        if model_type == 'tree':
            cur_model = train_model(X_train, Y_train, {'max_depth': param_combination['max_depth']}, model_type='tree')

        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)
        if cur_accuracy > best_accuracy_so_far:
            best_accuracy_so_far = cur_accuracy
            if model_type == 'svm':
                optimal_gamma = param_combination['gamma']
                optimal_C = param_combination['C']
                best_hparams = {'gamma': optimal_gamma,'C':optimal_C}
            if model_type == 'tree':
                optimal_max_depth = param_combination['max_depth']
                best_hparams = {'max_depth': optimal_max_depth}
            best_model_path = "./models/best_decision_tree_model" + "_".join(["{}:{}".format(k, v) for k, v in best_hparams.items()]).replace(":", "_") + ".joblib"

            best_model = cur_model

    # save the best model
    dump(best_model,best_model_path)
    #best_model_path =
    return best_hparams, best_model_path, best_accuracy_so_far


def tune_hparams_using_DT(X_train, Y_train, X_dev, y_dev, list_of_all_param_combination):
    best_accuracy_so_far = -1
    best_model = None
    best_model_path = ""

    for param_combination in list_of_all_param_combination:
        # Train a Decision Tree model instead of SVM
        cur_model = DecisionTreeClassifier(max_depth=param_combination['max_depth'], min_samples_split=param_combination['min_samples_split'])
        cur_model.fit(X_train, Y_train)

        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)
        if cur_accuracy > best_accuracy_so_far:
            best_accuracy_so_far = cur_accuracy
            optimal_max_depth = param_combination['max_depth']
            optimal_min_samples_split = param_combination['min_samples_split']
            best_hparams = {'max_depth': optimal_max_depth, 'min_samples_split': optimal_min_samples_split}
            # Modify the filename to indicate it's a Decision Tree model
            best_model_path = "./models/best_decision_tree_model" + "_".join(["{}:{}".format(k, v) for k, v in best_hparams.items()]).replace(":", "_") + ".joblib"
            best_model = cur_model

    # Save the best Decision Tree model
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

def calculate_scores(y_test, production_predictions, candidate_predictions):
    production_accuracy = accuracy_score(y_test, production_predictions)
    candidate_accuracy = accuracy_score(y_test, candidate_predictions)

    # Step 6: Create a confusion matrix for both models
    production_confusion_matrix = confusion_matrix(y_test, production_predictions)
    candidate_confusion_matrix = confusion_matrix(y_test, candidate_predictions)

    return production_accuracy, candidate_accuracy, production_confusion_matrix, candidate_confusion_matrix

def get_f1_score(y_test, production_predictions, candidate_predictions):
    production_macro_f1 = f1_score(y_test, production_predictions, average='macro')
    candidate_macro_f1 = f1_score(y_test, candidate_predictions, average='macro')
    return production_macro_f1, candidate_macro_f1
