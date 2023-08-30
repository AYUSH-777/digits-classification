# This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import metrics
from utils import preprocess_data, train_model, split_data, read_digits, split_train_dev_test, predict_and_eval

# The digits dataset consists of 8x8 pixel images of digits. The images attribute of the dataset stores 8x8 arrays of grayscale values for each image. We will use these arrays to visualize the first 4 images. The target attribute of the dataset stores the digit each image represents and this is included in the title of the 4 plots below.
# Note: if we were working from image files (e.g., ‘png’ files), we would load them using matplotlib.pyplot.imread.

# 1. Data Loading
x,y = read_digits()

# 3. Data splitting
#X_train, X_test, y_train, y_test = split_data(x, y, test_size=0.3);
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(x, y, test_size=0.3, dev_size=0.2)


# 4. Data Preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# 5. Train the data
model = train_model(X_train, y_train, {'gamma': 0.001}, model_type='svm')

# Predict the value of the digit on the test subset
# 6. Model Predictino
predicted = predict_and_eval(model, X_test, y_test)

# 7. Quantitative sanity check
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


plt.show()