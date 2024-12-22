import numpy as np
import run_model as RM
import random_things as RT
import function_EnKF as function 
import addtional_function as add_function
import plottest 
from allways_plot import plot_matrices as pit
from allways_plot import localization


### call function

def load_data_fft(small_shape):#load data 
  data_fft = np.load('/wk2/pc/rotation/data/DA14_fft.npy')  # (t,x,y) ~ (51,512,512) 
  data_fft_small = add_function.downsample_to_average_3d(data_fft, small_shape)
  print('data_fft shape:', data_fft_small.shape )
  return data_fft_small # (51,64,64)

def load_data_PG14(small_shape):#load data 
  data_PG14 = np.load('/wk2/pc/rotation/data/DA14_PG.npy')  # (t,x,y) ~ (51,512,512)
  data_PG14_small = add_function.downsample_to_average_3d(data_PG14, small_shape)
  print('data_PG14 shape:', data_PG14_small.shape )
  return data_PG14_small # (51,64,64)


def initial_ensemble_member(initial_matrix): #initial ensemble data + eddy
  initial_ensemble_member_list = []
  for each_ensemble_member in range(K):
    initial_ensemble_member_list += [ RT.add_gaussian_noise(initial_matrix, mu=0, std=1) ]  
  
  return np.array( initial_ensemble_member_list ) # (K,initial_matrix.shape[0],initial_matrix.shape[1]) (20,512,512)
  

def ensemble_member_run_through_model(ensemble_member_array,model): #Ensure that each ensemble member is processed through the model once.
  ##[PG1_i,PG2_i,...,PGk_i] >> [PG1_{i+1},PG2_{i+1},...,PGk_{i+1}]
  New_ensemble_member_list = []
  for each_ensemble_member in range(K):
    New_ensemble_member_list += [ add_function.downsample_to_average(RM.generate_from_model(add_function.upsample_matrix( ensemble_member_array[each_ensemble_member,:,:], original_shape, method='linear')  ,model) , small_shape)]
  return np.array(New_ensemble_member_list)  # (20,512,512)
  
  
def find_Pf(ensemble_member_array): #Pf
  len_flatten_ensemble_member = ensemble_member_array[0,:,:].reshape(-1).shape[0] #512*512 ~ 262144
  ensemble = ensemble_member_array.reshape(K,-1) # (K, 512*512)
  a = function.covariance(ensemble)
  pit([a,a])
  anew = localization(a,localization_distance)
  pit([anew,a])
  return anew #function.covariance(ensemble) # (d x d)~(262144 x 262144)


def find_determine_of_ensemble(ensemble_member_array):
  return np.mean(ensemble_member_array, axis=0) , np.std(ensemble_member_array, axis=0)


def find_y0_ensemble(alpha, fftdata, PGdata, old_fftdata, old_determinedata, model):
  y0_list = []
  
  old_fftdata_M = add_function.downsample_to_average( RM.generate_from_model(add_function.upsample_matrix(old_fftdata, original_shape, method='linear'), model) , small_shape)
  old_determinedata_M = add_function.downsample_to_average(RM.generate_from_model(add_function.upsample_matrix(old_determinedata, original_shape, method='linear'), model) , small_shape)
  for each_ensemble_member in range(K):
    A , B , C = RT.generate_random_ABC()
    if False :
      y0_list += [ alpha*fftdata+(1-alpha)*(A*PGdata+B*old_fftdata_M+C*old_determinedata_M) ]
    if True :
      y0_list += [ fftdata + np.random.normal(loc=0, scale=0.2)  ]
    
  return np.array(y0_list) #(K,512,512)
    
  

def from_xf_to_xa(ensemble_member_array_f, R, alpha, fftdata, PGdata, old_fftdata, model):   # xf = ensemble_member_array_f , xa = ensemble_member_array_a
  # ensemble_member_array_i ~ (K,512,512)
  Pf = find_Pf(ensemble_member_array_f) # (d x d)~(262144 x 262144)
  Kalman_gain = function.kalman_gain(Pf, R) # (d x p)~(262144 x 4096)
  
  determine_mean , determine_std = find_determine_of_ensemble(ensemble_member_array_f)
  y0_ensemble = find_y0_ensemble(alpha, fftdata, PGdata, old_fftdata, determine_mean, model)
  
  xa_list = []
  for each_ensemble_member in range(K):
    #print(ensemble_member_array_f.shape , Kalman_gain.shape ,  y0_ensemble.shape , H.shape)
    xa_list += [ function.xa_k(ensemble_member_array_f[each_ensemble_member,:,:].reshape(-1), Kalman_gain, y0_ensemble[each_ensemble_member,:,:].reshape(-1)).reshape(ensemble_member_array_f.shape[1:])  ] 
  
  if use_inflation :
    inflation_xa = (1-inflation_alpha)*np.array(xa_list)+inflation_alpha*ensemble_member_array_f
  else:
    inflation_xa = np.array(xa_list)
    
  return inflation_xa , y0_ensemble #(K,512,512) , (K,512,512)
  
  
  



if __name__ == '__main__':
  ### set!
  K = 20 # the number of ensumble member , 40 is good [20,30,40]
  small_shape = (32,32) # [(32,32) (64,64)] (32,32) is good 
  original_shape = (512,512)

  R = 0.1*np.eye(small_shape[0]**2)
  
  localization_distance = 2 # [2,5,10,15] 2 5 is good
  alpha = 0.9
  use_inflation = False
  inflation_alpha = 0.1 # don't use is better  [0,0.1,0.15,0.2,0.3]
  model = RM.load_model()
  DA_time = 50 #50
  
  
  
  
  
  
data_fft = load_data_fft(small_shape)   # (51,64,64)
data_PG14 = load_data_PG14(small_shape) # (51,64,64)
for asdasd in [1]: 
  datasave_name = f'/wk2/pc/rotation/DA/DA_noH/DA_EnKF_xa_y0_alpha1normal_{small_shape[0]}_loc{localization_distance}_{use_inflation}-inflationAlpha{inflation_alpha}_K{K}_.npy'
  
  ### end set!

  

  initial_ensemble_member = initial_ensemble_member(data_fft[0,:,:]) # (512,512) >> (20,512,512)
  ensemble_member_array_f = ensemble_member_run_through_model(initial_ensemble_member,model)  # (20,512,512) >> (20,512,512)f
  
  ensemble_member_array_a, y0_ensemble = from_xf_to_xa(ensemble_member_array_f, R, alpha, data_fft[0,:,:], data_PG14[0,:,:], data_fft[0,:,:], model) # (20,512,512)f >> (20,512,512)a , (20,512,512)y0 
  
  ###save
  saveout_array = np.zeros((2*K , DA_time+1 ,data_fft.shape[1],data_fft.shape[2])) #
  saveout_array[:K,0,:,:] = ensemble_member_array_a
  saveout_array[K:,0,:,:] = y0_ensemble
  ###end save
  print(ensemble_member_array_a[0,:,:].shape,y0_ensemble[0,:,:].shape)
  print(len([ensemble_member_array_a[0,:,:],y0_ensemble[0,:,:]]))
  #plottest.plot_it_ensemble5_fortest (0,[ensemble_member_array_a[0,:,:],y0_ensemble[0,:,:],ensemble_member_array_a[1,:,:],y0_ensemble[1,:,:]] )
  
  for dotime in range(DA_time):
      ensemble_member_array_f = ensemble_member_run_through_model(ensemble_member_array_a,model)# (20,512,512)a >> (20,512,512)f
      ensemble_member_array_a , y0_ensemble= from_xf_to_xa(ensemble_member_array_f, R, alpha, data_fft[dotime+1,:,:], data_PG14[dotime+1,:,:], data_fft[dotime,:,:], model)# (20,512,512)f >> (20,512,512)a , (20,512,512)y0
      
      ###save
      saveout_array[:K,dotime+1,:,:] = ensemble_member_array_a
      saveout_array[K:,dotime+1,:,:] = y0_ensemble
      ###end save 
      #plottest.plot_it_ensemble5_fortest (0,[ensemble_member_array_a[0,:,:],y0_ensemble[0,:,:],ensemble_member_array_a[1,:,:],y0_ensemble[1,:,:]])
      print(dotime)

  #np.save('/wk2/pc/rotation/DA/DA_EnKF_xa_y0.npy',saveout_array)
  ####np.save('/wk2/pc/rotation/DA/DA_noH/DA_EnKF_xa_y0_alpha1normal_64.npy',saveout_array)
  #np.save(datasave_name,saveout_array)















