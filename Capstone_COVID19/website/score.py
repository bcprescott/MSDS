# import tensorflow as tf
# import numpy as np
# import json
# import os
# from tensorflow import keras
# from tensorflow.keras import models
#
#
# def init():
#     global model
#     print("Inference Model Loading")
#     model = tf.keras.models.load_model(os.path.join(os.getenv("AZUREML_MODEL_DIR"), "MultiLayer-24-64.h5"))
#
# def run(request):
#
#     data = json.loads(request)
#     data = np.asarray(data['array'])
#
#
#     img = np.divide(data,255.)
#
#     img = np.expand_dims(img, axis=0)
#
#     pred = np.argmax(model.predict(img))
#     return f"The image result is: {pred}"

import tensorflow as tf
import numpy as np
import json
from json import JSONEncoder
import os
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import urllib



def init():
    global model
    print("Inference Model Loading")
    model = tf.keras.models.load_model(os.path.join(os.getenv("AZUREML_MODEL_DIR"), "MultiLayer-24-64.h5"))

def run(request):
    data = json.loads(request)
    data = data['image']
    url = "https://capimages.blob.core.windows.net/test/" + data

    img = tf.keras.utils.get_file('Image', origin=url)
    img = load_img(img, target_size=(300,300))
    img = img_to_array(img)
    img = np.divide(img,255.)
    img = np.expand_dims(img, axis=0)
    pred = np.argmax(model.predict(img))
    return f"The image result is: {pred}"