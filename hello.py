from flask import Flask, request
import tensorflow as tf
import numpy as np
import tensorflow as tf
from flask import jsonify
from PIL import Image
import numpy as np
# from tensorflow.keras.preprocessing.image import load_img, img_to_array


app = Flask(__name__)

# model = tf.keras.models.load_model('C:\Users\ayush\OneDrive\Documents\MLOps\digits-classification\models\best_modelgamma_0.001_C_2.joblib')
# model = tf.keras.models.load_model('C:\\Users\\ayush\\OneDrive\\Documents\\MLOps\\digits-classification\\models\\best_modelgamma_0.001_C_2.joblib')


@app.route("/hello/<val>")
def hello_world(val):
    return "<p>Hello, World!</p>" + val


@app.route("/sum/<x>/<y>")
def sum_num(x,y):
    sum = int(x) + int(y)
    return str(sum)


@app.route("/model", methods = ['POST'])
def pred_model():
    js = request.get_json()
    x = js['x']
    y = js['y']
    sum = int(x) + int(y)
    return str(sum)



def compare_images(image1, image2):
    # Open and convert the images to grayscale
    img1 = Image.open(image1).convert('L')
    img2 = Image.open(image2).convert('L')

    # Resize the images to the same dimensions
    img1 = img1.resize((28, 28))
    img2 = img2.resize((28, 28))

    # Convert the images to NumPy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # Compare the NumPy arrays for equality
    is_same_digit = np.array_equal(arr1, arr2)

    return is_same_digit

@app.route('/compare_images', methods=['POST'])
def compare_images_route():
    try:
        # Get the uploaded images from the request
        image1 = request.files['image1']
        image2 = request.files['image2']

        # Compare the images
        is_same_digit = compare_images(image1, image2)

        return jsonify({'is_same_digit': is_same_digit})
    except Exception as e:
        return jsonify({'error': str(e)})