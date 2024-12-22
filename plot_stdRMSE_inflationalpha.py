import numpy as np

import addtional_function as add_function
import matplotlib.pyplot as plt


L = 12
n_Unet = 32
x = np.linspace(-L / 2, L / 2, n_Unet)
y = np.linspace(-L / 2, L / 2, n_Unet)
X, Y = np.meshgrid(x, y)


small_shape = (n_Unet,n_Unet)
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



if __name__ == '__main__':
  file_name = 'loc5'
  
  # 2. load data
  K = 20
  plt.figure(figsize=(10, 6))
  
  
  
  color , dotime = ['r','b','g','k','m'] , 0
  for inflationAlpha in [0,0.1,0.15,0.2,0.3]: 
    if dotime == 0:
      data = np.load(f'DA_EnKF_xa_y0_alpha1normal_32_loc2_False-inflationAlpha0.1_K20_.npy')# (2K,t,x,y)
    else:
      data = np.load(f'DA_EnKF_xa_y0_alpha1normal_32_loc2_True-inflationAlpha{inflationAlpha}_K20_.npy')# (2K,t,x,y)
    
    data_xa_1000 , data_xa_4000 = data[:K,10,:,:] , data[:K,40,:,:] 
    
    data_fft_small = load_data_fft(small_shape) #(t,x,y)
    #data_PG14_small = load_data_PG14(small_shape) 
    data_fft_1000 , data_fft_4000 = data_fft_small[10,:,:] , data_fft_small[40,:,:]
    
    
    data_RMSE_1000_list , data_RMSE_4000_list = [] , []
    delta1000 , delta4000 = [] , []
    for each_ensemble_member in range(K):
      data_RMSE_1000_list += [ (np.mean((data_xa_1000[each_ensemble_member,:,:]-data_fft_1000)**2))**0.5 ]
      data_RMSE_4000_list += [ (np.mean((data_xa_4000[each_ensemble_member,:,:]-data_fft_4000)**2))**0.5 ]
      delta1000 += [ np.mean(data_xa_1000[each_ensemble_member,:,:]-np.mean(data_xa_1000[:,:,:],axis=0)) ]
      delta4000 += [ np.mean(data_xa_4000[each_ensemble_member,:,:]-np.mean(data_xa_4000[:,:,:],axis=0)) ]
    std1000 = np.std(delta1000)
    std4000 = np.std(delta4000)
    
    n_index1000 = delta1000/std1000
    n_index4000 = delta4000/std4000
  
    
    # Plotting
   
    
    plt.plot(data_RMSE_1000_list, n_index1000, f'{color[dotime]}d')#, label=f'inflationAlpha={inflationAlpha},Time: 1000')
    plt.plot(data_RMSE_4000_list, n_index4000, f'{color[dotime]}o')#, label=f'inflationAlpha={inflationAlpha},Time: 4000')
    
    
    dotime += 1

  # Enhance visualization
  plt.grid(alpha=0.5, linestyle='--', linewidth=0.7)
  plt.title('Normalized Index vs. RMSE', fontsize=16)
  plt.xlabel('RMSE', fontsize=14)
  plt.ylabel('Normalized Index', fontsize=14)
  plt.legend(fontsize=12)
      
  # Add minor ticks
  plt.minorticks_on()
  plt.tick_params(axis='both', which='both', direction='in', length=6)
  plt.tick_params(axis='both', which='minor', length=3)      
  # Show plot
  plt.tight_layout()
  plt.savefig(f'plot/inflationAlpha.png')
  plt.show()



  ## RMSEplot
  def error(data_fft_small , data3D_txy ): # (t,x,y) , (t,x,y)>>  (t)
    return np.mean((data_fft_small-data3D_txy)**2 ,axis=(1,2))**0.5
  
  timeline = np.arange(0,50+1e-5,1)
  
  
  
  plt.figure(figsize=(10, 6)) 
  dotime = 0
  for inflationAlpha in [0,0.1,0.15,0.2,0.3]: 
    if dotime == 0:
      data = np.load(f'DA_EnKF_xa_y0_alpha1normal_32_loc2_False-inflationAlpha0.1_K20_.npy')# (2K,t,x,y)
    else:
      data = np.load(f'DA_EnKF_xa_y0_alpha1normal_32_loc2_True-inflationAlpha{inflationAlpha}_K20_.npy')# (2K,t,x,y)
    
    data_xa = data[:K,:,:,:]
    data_xa_determine = np.mean(data_xa,axis=0)
    
    plt.plot(timeline, error(data_fft_small , data_xa_determine ),color[dotime],label=f'inflationAlpha={inflationAlpha}') 
    #plt.plot(timeline, error(data_fft_small , data_PG14_small   ),'k',label='PG error')
    dotime += 1
  plt.axvline(x=40, color='red', linestyle='--', linewidth=1.5)
  plt.axvline(x=10, color='red', linestyle='--', linewidth=1.5)    
  plt.xlim([0,50])
  plt.ylim([0,0.2])
  plt.legend()
  
  plt.title('inflation Alpha') 
  plt.xlabel('timeline') 
  plt.ylabel('RMSE')
  
  plt.grid(alpha=0.5, linestyle='--', linewidth=0.7)
  plt.tight_layout()
  plt.savefig(f'plot/inflationAlpha-timeline_error_small.png')
  
  plt.show()








