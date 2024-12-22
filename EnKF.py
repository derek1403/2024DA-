import numpy as np
import run_model as RM
import random_things as RT
import observation_operator as O
import function_EnKF as function 




### call function

def load_data_fft(load_lightdata = True ):#load data 
  data_fft = np.load('/wk2/pc/rotation/data/DA14_fft.npy')  # (t,x,y) ~ (51,512,512) 
  print('data_fft shape:', data_fft.shape )
  return data_fft

def load_data_PG14():#load data 
  data_PG14 = np.load('/wk2/pc/rotation/data/DA14_PG.npy')  # (t,x,y) ~ (51,512,512) 
  print('data_PG14 shape:', data_PG14.shape )
  return data_PG14




def initial_ensemble_member(initial_matrix): #initial ensemble data + eddy
  initial_ensemble_member_list = []
  for each_ensemble_member in range(K):
    initial_ensemble_member_list += [ RT.add_gaussian_noise(initial_matrix, mu=0, std=1) ]  
  
  return np.array( initial_ensemble_member_list ) # (K,initial_matrix.shape[0],initial_matrix.shape[1]) (20,512,512)
  

def ensemble_member_run_through_model(ensemble_member_array,model): #Ensure that each ensemble member is processed through the model once.
  ##[PG1_i,PG2_i,...,PGk_i] >> [PG1_{i+1},PG2_{i+1},...,PGk_{i+1}]
  New_ensemble_member_list = []
  for each_ensemble_member in range(K):
    New_ensemble_member_list += [ RM.generate_from_model(ensemble_member_array[each_ensemble_member,:,:],model) ]
  return np.array(New_ensemble_member_list)  # (20,512,512)
  
  
def find_Pf(ensemble_member_array): #Pf
  len_flatten_ensemble_member = ensemble_member_array[0,:,:].reshape(-1).shape[0] #512*512 ~ 262144
  ensemble = ensemble_member_array.reshape(K,-1) # (K, 512*512)
  return function.covariance(ensemble) # (d x d)~(262144 x 262144)


def find_determine_of_ensemble(ensemble_member_array):
  return np.mean(ensemble_member_array, axis=0) , np.std(ensemble_member_array, axis=0)


def find_y0_ensemble(alpha, fftdata, PGdata, old_fftdata, old_determinedata, model):
  y0_list = []
  
  old_fftdata_M = RM.generate_from_model(old_fftdata, model) 
  old_determinedata_M = RM.generate_from_model(old_determinedata, model) 
  for each_ensemble_member in range(K):
    A , B , C = RT.generate_random_ABC()
    y0_list += [ alpha*fftdata+(1-alpha)*(A*PGdata+B*old_fftdata_M+C*old_determinedata_M) ]
    
  return np.array(y0_list) #(K,512,512)
    
  

def from_xf_to_xa(ensemble_member_array_f, H, R, alpha, PGdata, old_fftdata, model):   # xf = ensemble_member_array_f , xa = ensemble_member_array_a
  # ensemble_member_array_i ~ (K,512,512)
  Pf = find_Pf(ensemble_member_array_f) # (d x d)~(262144 x 262144)
  Kalman_gain = function.kalman_gain(Pf, H, R) # (d x p)~(262144 x 4096)
  
  determine_mean , determine_std = find_determine_of_ensemble(ensemble_member_array_f)
  y0_ensemble = find_y0_ensemble(alpha, fftdata, PGdata, old_fftdata, determine_mean, model)
  
  xa_list = []
  for each_ensemble_member in range(K):
    xa_list += [ function.xa_k(ensemble_member_array_f, Kalman_gain, y0_ensemble, H) ]
  
  return np.array(xa_list) , y0_ensemble #(K,512,512) , (K,512,512)
  
  
  



if __name__ == '__main__':
  ### set!
  K = 20 # the number of ensumble member

  H = O.observation_operator() # 512,64>killed  | 512,2>killed
  R = 0.1
  
  alpha = 0.2
  
  model = RM.load_model()
  DA_time = 1 #50
  
  load_lightdata = True
  
  ### end set!

  
  data_fft = load_data_fft()   # (51,512,512)
  data_PG14 = load_data_PG14() # (51,512,512)
  initial_ensemble_member = initial_ensemble_member(data_fft[0,:,:]) # (512,512) >> (20,512,512)
  ensemble_member_array_f = ensemble_member_run_through_model(initial_ensemble_member,model)  # (20,512,512) >> (20,512,512)f
  
  ensemble_member_array_a, y0_ensemble = from_xf_to_xa(ensemble_member_array_f, H, R, alpha, data_PG14[0,:,:], data_fft[0,:,:], model) # (20,512,512)f >> (20,512,512)a , (20,512,512)y0 
  
  ###save
  saveout_array = np.zeros((2*K , DA_time+1 ,data_fft.shape[1],data_fft.shape[2])) #
  saveout_array[:K,0,:,:] = ensemble_member_array_a
  saveout_array[K:,0,:,:] = y0_ensemble
  ###end save
  
  for dotime in range(DA_time):
      ensemble_member_array_f = ensemble_member_run_through_model(ensemble_member_array_a,model)# (20,512,512)a >> (20,512,512)f
      ensemble_member_array_a , y0_ensemble= from_xf_to_xa(ensemble_member_array_f, H, R, alpha, data_PG14[dotime+1,:,:], data_fft[dotime,:,:], model)# (20,512,512)f >> (20,512,512)a , (20,512,512)y0
      
      ###save
      saveout_array[:K,dotime+1,:,:] = ensemble_member_array_a
      saveout_array[K:,dotime+1,:,:] = y0_ensemble
      ###end save 
      print(dotime)

  np.save('/wk2/pc/rotation/DA/DA_EnKF_xa_y0',saveout_array)















