from flask import Flask, request
# import tensorflow as tf
# import numpy as np
# import tensorflow as tf
from flask import jsonify
from joblib import load
# from PIL import Image
import numpy as np
# from utils import preprocess_data, tune_hparams, split_train_dev_test,read_digits,predict_and_eval, calculate_scores, get_f1_score
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
    # print(x)
    # print(y)
    sum = int(x) + int(y)
    return str(sum)

def preprocess_data_dup(data):
    # Convert the input data to a NumPy array
    data = np.array(data, dtype=float)

    # Reshape the data (assuming you want a 1D array)
    data = data.reshape(1, -1)

    return data

@app.route('/predict_images', methods=['POST'])
def compare_images_route():
    try:
        # Get the uploaded images from the request
        best_model_path = './models/best_decision_tree_modelgamma_0.001_C_5.joblib'
        js = request.get_json()
        image1 = js['image1']
        image2 = js['image2']
        image1 = preprocess_data_dup(image1)
        image2 = preprocess_data_dup(image2)
        best_model = load(best_model_path)
        predicted_image1 = best_model.predict(image1)
        predicted_image2 = best_model.predict(image2)
        print(predicted_image1[0])
        print(predicted_image2[0])

        is_same_digit = (predicted_image1[0] == predicted_image2[0])
        print(is_same_digit)
        return str(is_same_digit)
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/predict', methods=['POST'])
def compare_digit_route():
    try:
        # Get the uploaded images from the request
        best_model_path = './models/best_decision_tree_modelgamma_0.001_C_5.joblib'
        js = request.get_json()
        image1 = js['image']
        image2 = image1
        image1 = preprocess_data_dup(image1)
        best_model = load(best_model_path)
        predicted_image1 = best_model.predict(image1)
        predicted_final_digit = image2[-1]
        return predicted_final_digit
    except Exception as e:
        return jsonify({'error': str(e)})