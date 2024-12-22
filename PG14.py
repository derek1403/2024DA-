import numpy as np
import tensorflow as tf




def load_model():
  model = tf.keras.models.load_model('/wk2/pc/rotation/Unet_model_all_1115.h5')
  return model
  
def generate_from_model(input_array,model): # (512,512)
  input_image  = np.expand_dims(  np.expand_dims(input_array, axis=2)  , axis=0)  # (512,512) >> (1, 512, 512, 1)
  predicted_output = 5*model.predict(input_image/5)[0,:,:,0]  # (1, 512, 512, 1)  >>  (512,512) ,since it had been normalized while traning process.
  
  return predicted_output



if True:
    model = load_model()
    PG_save , fft_save = np.zeros((51,512,512)) , np.zeros((51,512,512))

    data = np.transpose(np.load('/wk2/pc/rotation/data/zeta14_15_640_5001_0.01_512.npy'), (2, 0, 1))   ## (x,y,t) >> (t,x,y)
    print("good")
    
    PG_save [0,:,:] = generate_from_model( data[0,:,:] ,model)
    fft_save[0,:,:] = data[0,:,:]
    for i in range(1,51):
    

      PG_save [i,:,:] = generate_from_model( PG_save[i-1,:,:] ,model)
      fft_save[i,:,:] = data[i*100,:,:]
      
    

    np.save(f"/wk2/pc/rotation/data/DA14_PG.npy" ,PG_save)
    np.save(f"/wk2/pc/rotation/data/DA14_fft.npy",fft_save)


