import numpy as np
import tensorflow as tf


def load_model():
  model = tf.keras.models.load_model('/wk2/pc/rotation/Unet_model_all_1115.h5')
  return model
  
def generate_from_model(input_array,model): # (512,512)
  input_image  = np.expand_dims(  np.expand_dims(input_array, axis=2)  , axis=0)  # (512,512) >> (1, 512, 512, 1)
  predicted_output = 5*model.predict(input_image/5)[0,:,:,0]  # (1, 512, 512, 1)  >>  (512,512) ,since it had been normalized while traning process.
  
  return predicted_output

































