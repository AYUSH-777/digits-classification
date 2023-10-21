# This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# Import datasets, classifiers and performance metrics
from utils import preprocess_data, tune_hparams, split_train_dev_test,read_digits,predict_and_eval, calculate_scores, get_f1_score
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
parser=argparse.ArgumentParser()

parser.add_argument("--runs", help="number of runs")
args=parser.parse_args()

max_runs = int(args.runs)


#print("Total number of samples : ", len(x))

#print("(number of samples,length of image,height of image) is:",x.shape)

test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

# test_sizes = [0.2]
# dev_sizes = [0.2]
results = []
models = ['svm','tree']


for i in range(max_runs):
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            # 3. Data splitting
            X_train, X_test,X_dev, y_train, y_test,y_dev = split_train_dev_test(x, y, test_size=test_size, dev_size=dev_size);

            # 4. Data Preprocessing
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)

            classifer_hparam = {}

            gama_ranges = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            C_ranges = [0.1,1,2,5,10]
            classifer_hparam['svm']= [{'gamma': gamma, 'C': C} for gamma in gama_ranges for C in C_ranges]

            max_depth = [5,10,15,20,50,100]
            classifer_hparam['tree'] = [{'max_depth': depth} for depth in max_depth]


            # Predict the value of the digit on the test subset
            # 6.Predict and Evaluate
            for model in models:
                best_hparams, best_model_path, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, classifer_hparam[model], model_type=model)
                best_model = load(best_model_path)

                accuracy_test = predict_and_eval(best_model, X_test, y_test)
                accuracy_dev = predict_and_eval(best_model, X_dev, y_dev)
                accuracy_train = predict_and_eval(best_model, X_train, y_train)
                # print(f"model={model} run_index={i} test_size={test_size} dev_size={dev_size} train_size={1- (dev_size+test_size)} train_acc={accuracy_train} dev_acc={accuracy_dev} test_acc={accuracy_test}")
                # results.append([{'model':model,'run_index': i, 'test_size':test_size, 'dev_size':dev_size,'train_size': 1- (dev_size+test_size), 'train_acc':accuracy_train,'dev_acc':accuracy_dev,'test_acc':accuracy_test}])
        #print(f"best_gamma={best_hparams['gamma']},best_C={best_hparams['C']}")




X, y = read_digits()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train and tune the production model (SVM)
production_model = SVC(C=1.0, kernel='rbf', gamma='scale')
production_model.fit(X_train, y_train)

# Step 3: Train and tune the candidate model (Decision Tree)
candidate_model = DecisionTreeClassifier(max_depth=10, random_state=42)
candidate_model.fit(X_train, y_train)

# Step 4: Make predictions using both models
production_predictions = production_model.predict(X_test)
candidate_predictions = candidate_model.predict(X_test)

production_accuracy, candidate_accuracy, production_confusion_matrix, candidate_confusion_matrix = calculate_scores(y_test, production_predictions, candidate_predictions)

# Step 7: Calculate a 2x2 confusion matrix for samples predicted correctly by the production model but not by the candidate model
correctly_predicted_by_production = (production_predictions == y_test) & (candidate_predictions != production_predictions)
correctly_predicted_by_candidate = (candidate_predictions == y_test) & (candidate_predictions != production_predictions)
confusion_2x2 = np.array([
    [np.sum(correctly_predicted_by_production), np.sum(correctly_predicted_by_candidate)],
    [np.sum(~correctly_predicted_by_production), np.sum(~correctly_predicted_by_candidate)]
])


production_macro_f1, candidate_macro_f1 = get_f1_score(y_test, production_predictions, candidate_predictions)
# Print the results
print("Results for QUiz - 2")
print("Production Model Accuracy:", production_accuracy)
print("Candidate Model Accuracy:", candidate_accuracy)
print("Production Model Confusion Matrix:\n", production_confusion_matrix)
print("Candidate Model Confusion Matrix:\n", candidate_confusion_matrix)
print("2x2 Confusion Matrix:\n", confusion_2x2)
print("Production Model Macro-average F1 Score:", production_macro_f1)
print("Candidate Model Macro-average F1 Score:", candidate_macro_f1)