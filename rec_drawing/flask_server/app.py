import flask, pickle, base64, cv2
from flask import Flask, render_template, url_for, request
import numpy as np, tensorflow as tf

init_base64 = 21
label_dict = {0: 'Cat', 1: 'Giraffe', 2: 'Sheep', 3: 'Bat', 4: 'Octopus', 5: 'Camel'}

graph = tf.get_default_graph()
with open(f'model_cnn.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__, template_folder='templates')


@app.route('/')
def home():
    return render_template('draw.html')


@app.route('/predict', methods=['POST'])
def predict():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            final_pred = None
            draw = request.form['url']
            draw = draw[init_base64:]
            draw_decoded = base64.b64decode(draw)
            image = np.asarray(bytearray(draw_decoded), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
            vect = np.asarray(resized, dtype="uint8")
            vect = vect.reshape(1, 1, 28, 28).astype('float32')
            # Launch prediction
            my_prediction = model.predict(vect)
            index = np.argmax(my_prediction[0])
            final_pred = label_dict[index]
    return render_template('results.html', prediction=final_pred)


if __name__ == '__main__':
    app.run(debug=True)
