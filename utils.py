
# Import datasets, classifiers and performance metrics
from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics


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

def split_train_dev_test(X, y, test_size, dev_size):
    # Split the data into training and temp sets
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Calculate the proportion of the original dataset that the development set should have
    dev_prop = dev_size / (1 - test_size)

    # Split the temp set into training and development sets
    X_train, X_dev, y_train, y_dev = train_test_split(X_train_temp, y_train_temp, test_size=dev_prop, random_state=42)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def predict_and_eval(model, X_test, y_test):
    # Make predictions
    predicted = model.predict(X_test)

    # Print classification report
    # classification_report builds a text report showing the main classification metrics.
    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )

    #8. Model Evaluation
    # Plot confusion matrix
    disp = metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    return predicted
