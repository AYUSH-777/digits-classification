# This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# Import datasets, classifiers and performance metrics
from utils import preprocess_data, tune_hparams, split_train_dev_test,read_digits,predict_and_eval, calculate_scores, get_f1_score, get_normalized, split_train_dev_test_lr, tune_hparams_lr
from joblib import load
import pandas as pd
import argparse, sys

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# The digits dataset consists of 8x8 pixel images of digits. The images attribute of the dataset stores 8x8 arrays of grayscale values for each image. We will use these arrays to visualize the first 4 images. The target attribute of the dataset stores the digit each image represents and this is included in the title of the 4 plots below.
# Note: if we were working from image files (e.g., ‘png’ files), we would load them using matplotlib.pyplot.imread.

# 1. Data Loading

x,y = read_digits()
# parser=argparse.ArgumentParser()
#
# parser.add_argument("--runs", help="number of runs")
# args=parser.parse_args()
#
max_runs = 1

test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

gama_ranges = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
C_ranges = [0.1,1,2,5,10]
depth_ranges = [5,10,15,20,50,100]

# test_sizes = [0.2]
# dev_sizes = [0.2]
results = []
models = ['svm','tree']


# for i in range(max_runs):
#     for test_size in test_sizes:
#         for dev_size in dev_sizes:
#             # 3. Data splitting
#             X_train, X_test,X_dev, y_train, y_test,y_dev = split_train_dev_test(x, y, test_size=test_size, dev_size=dev_size);
#
#             # 4. Data Preprocessing
#             X_train = preprocess_data(X_train)
#             X_test = preprocess_data(X_test)
#             X_dev = preprocess_data(X_dev)
#
#             classifer_hparam = {}
#
#             gama_ranges = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
#             C_ranges = [0.1,1,2,5,10]
#             classifer_hparam['svm']= [{'gamma': gamma, 'C': C} for gamma in gama_ranges for C in C_ranges]
#
#             max_depth = [5,10,15,20,50,100]
#             classifer_hparam['tree'] = [{'max_depth': depth} for depth in max_depth]
#
#
#             # Predict the value of the digit on the test subset
#             # 6.Predict and Evaluate
#             for model in models:
#                 best_hparams, best_model_path, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, classifer_hparam[model], model_type=model)
#                 best_model = load(best_model_path)
#
#                 accuracy_test = predict_and_eval(best_model, X_test, y_test)
#                 accuracy_dev = predict_and_eval(best_model, X_dev, y_dev)
#                 accuracy_train = predict_and_eval(best_model, X_train, y_train)
#                 print(f"model={model} run_index={i} test_size={test_size} dev_size={dev_size} train_size={1- (dev_size+test_size)} train_acc={accuracy_train} dev_acc={accuracy_dev} test_acc={accuracy_test}")
#                 results.append([{'model':model,'run_index': i, 'test_size':test_size, 'dev_size':dev_size,'train_size': 1- (dev_size+test_size), 'train_acc':accuracy_train,'dev_acc':accuracy_dev,'test_acc':accuracy_test}])
        #print(f"best_gamma={best_hparams['gamma']},best_C={best_hparams['C']}")



### Q1. # Perform unit normalization

x, y = get_normalized(x,y)

#### Q2.     # Create Logistic Regression model with the specified solver

solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
mean_scores = []
std_scores = []

# for solver in solvers:
#     mean_score, std_score = train_and_evaluate(X, y, solver)
#     mean_scores.append(mean_score)
#     std_scores.append(std_score)

solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

model_name = 'lr'

for test_size in test_sizes:
    for dev_size in dev_sizes:
        x_train, x_dev, x_test, y_train, y_dev, y_test = split_train_dev_test_lr(x,y,test_size,dev_size,42)

        x_train = preprocess_data(x_train)
        x_dev = preprocess_data(x_dev)
        x_test = preprocess_data(x_test)

        best_model_path = tune_hparams_lr(x_train,y_train,x_dev,y_dev,gama_ranges,C_ranges,depth_ranges,solvers,model_name,x_test,y_test)

        best_model = load(best_model_path)

        # test_accuracy = predict_and_eval(best_model,x_test,y_test)
        # print(f"best_model = {best_model} test_accuracy = {test_accuracy}")