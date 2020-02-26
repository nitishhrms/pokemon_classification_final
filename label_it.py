from keras.preprocessing import image
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import keras
from keras import backend as k
from sklearn.utils import shuffle
import tensorflow as tf
config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)

m=joblib.load('pokemon_model_final.pkl')
m._make_predict_function()
def ans(path):
    with session.as_default():
        with session.graph.as_default():
            img=image.load_img(path,target_size=(40,40))
            img1=image.img_to_array(img)
            img1=img1/255.0
            img2=img1.reshape(1,4800)
            predictions=m.predict(img2)
            prediction=np.array((predictions))
            labels=np.argmax(prediction,axis=-1)
            if labels==0:
                return 'pikachu'
            if labels==1:
                return 'bulbasor'
            if labels==2:
                return 'charmander'
            return labels

      
