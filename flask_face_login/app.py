import os
import tensorflow as tf
from tensorflow.keras import layers,Model
from tensorflow.python.keras.backend import set_session
from flask import Flask,request
import cv2,json,numpy as np,base64
from datetime import datetime

graph=tf.get_default_graph()
app=Flask(__name__)
sess=tf.Session()
set_session(sess)

model=tf.keras.models.load_model('facenet_keras.h5')

def img_to_encoding(path,model):
    img1=cv2.imread(path,1)
    img=img1[...,::-1]
    dim=(160,160)
    if img.shape!=(160,160,3):
        img=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
    x_train=np.array([img])
    embedding=model.predict(x_train)
    return embedding

database = {}
database["Sam"] = img_to_encoding("images/klaus.jpg", model)
database["Levi"] = img_to_encoding("images/captain_levi.jpg", model)
database["Eren"] = img_to_encoding("images/eren.jpg", model)
database["Armin"] = img_to_encoding("images/armin.jpg", model)


def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 1000
    #Looping over the names and encodings in the database.
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding-db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 5:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist, identity


@app.route('/verify',methods=['POST'])
def verify():
    img_data = request.get_json()['image64']
    img_name = str(int(datetime.timestamp(datetime.now())))
    with open('images/'+img_name+'.jpg', "wb") as fh:
        fh.write(base64.b64decode(img_data[22:]))
    path = 'images/'+img_name+'.jpg'
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        min_dist, identity = who_is_it(path, database, model)
    os.remove(path)
    if min_dist > 5:
        return json.dumps({"identity": 0})
    return json.dumps({"identity": str(identity)})



@app.route('/register', methods=['POST'])
def register():
    try:
        username = request.get_json()['username']
        img_data = request.get_json()['image64']
        with open('images/'+username+'.jpg', "wb") as fh:
            fh.write(base64.b64decode(img_data[22:]))
        path = 'images/'+username+'.jpg'
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            database[username] = img_to_encoding(path, model)
        return json.dumps({"status": 200})
    except:
        return json.dumps({"status": 500})


