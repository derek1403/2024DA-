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
  

def plot_it_ensemble5 (timestep,images):
    fig, axis = plt.subplots(5, 2, figsize=(8, 12), constrained_layout=True)
    
    levels = np.linspace(-1.5, 1.5, 101)
    titles = ['Input','Predict','Truth - Predict','Truth']
    titles_right = [f'dt = 0.01\nt = {timestep*0.01:.2f}',f't = {timestep*0.01+1:.2f}',
                             f't = {timestep*0.01+1:.2f}',f't = {timestep*0.01+1:.2f}' ]
    
    
    
    for i in range(10):
        row, col = divmod(i, 2)
        im = axis[row, col].contourf(X,Y,images[i][:,:],levels=levels,  cmap='bwr' , extend='both')#, vmin=-1.5, vmax=1.5)  
        axis[row, col].set_xlim(-5, 5)
        axis[row, col].set_ylim(-5, 5)
        
        
    cbar = fig.colorbar(im, ax=axis,extend='both', orientation='vertical', ticks=np.linspace(-1.5, 1.5, 7), fraction=0.03, pad=0.04)
    cbar.set_label("Value")  
    
    
    plt.savefig(f"plot/{file_name}/EnKF_ensemble5-{str(timestep).zfill(4)}.png")
    
    #plt.show()
    plt.close()

def plot_it_ensemble5_fortest (timestep,images):
    fig, axis = plt.subplots(2, 2, figsize=(8, 12), constrained_layout=True)
    
    levels = np.linspace(-1.5, 1.5, 101)    
    for i in range(4):
        row, col = divmod(i, 2)
        print(images[i].shape)
        im = axis[row, col].contourf(X,Y,images[i],levels=levels,  cmap='bwr' , extend='both')#, vmin=-1.5, vmax=1.5)  
        axis[row, col].set_xlim(-5, 5)
        axis[row, col].set_ylim(-5, 5)
        
        
    cbar = fig.colorbar(im, ax=axis,extend='both', orientation='vertical', ticks=np.linspace(-1.5, 1.5, 7), fraction=0.03, pad=0.04)
    cbar.set_label("Value")  
    
    
    #plt.savefig(f"plot/{file_name}/EnKF_ensemble5-{str(timestep).zfill(4)}.png")
    
    plt.show()
    #plt.close()

def plot_it_mean_std (timestep,images):
    fig, axis = plt.subplots(3, 2, figsize=(8, 10), constrained_layout=True)
    
    levels = np.linspace(-1.5, 1.5, 101)
    
    
    for i in range(6):
        row, col = divmod(i, 2)
        im = axis[row, col].contourf(X,Y,images[i][:,:], levels=levels, cmap='bwr', vmin=-1.5, vmax=1.5)
        axis[row, col].set_xlim(-5, 5)
        axis[row, col].set_ylim(-5, 5)
        
        
    cbar = fig.colorbar(im, ax=axis,extend='both', orientation='vertical', ticks=np.linspace(-1.5, 1.5, 7), fraction=0.03, pad=0.04)
    cbar.set_label("Value")  
    
    
    plt.savefig(f"plot/{file_name}/EnKF_mean_std-{str(timestep).zfill(4)}.png")
    
    #plt.show()
    plt.close()


if __name__ == '__main__':
  file_name = 'loc1'
  
  # 2. load data
  K = 20
  #data = np.load('DA_EnKF_xa_y0_alpha1normal_64.npy')
  data = np.load('DA_EnKF_xa_y0_alpha1normal_32_loc.npy')
  data_xa , data_y0 = data[:K,:,:,:] , data[K:,:,:,:]
  
  data_fft_small = load_data_fft(small_shape)
  data_PG14_small = load_data_PG14(small_shape)
  ###ensemble5 plot
  #"""
  for i in range(data.shape[1]):
    images_ensemble5 = []
    for j in range(4):
      images_ensemble5+= [data_xa[j,i,:,:] , data_y0[j,i,:,:]]
    images_ensemble5+= [data_fft_small[i,:,:] , data_PG14_small[i,:,:] ] 
    plot_it_ensemble5(i*100,images_ensemble5)
  #"""
  
  ###mean_std plot
  #"""
  for i in range(data.shape[1]):
    images_mean_std  = [ np.mean(data_xa[:,i,:,:] ,axis=0)   , np.mean(data_y0[:,i,:,:] ,axis=0)    ]
    images_mean_std += [ np.std (data_xa[:,i,:,:] ,axis=0)*20, np.std (data_y0[:,i,:,:] ,axis=0)*20 ]
    
    images_mean_std += [ data_fft_small[i,:,:] , data_PG14_small[i,:,:] ] 
    plot_it_mean_std(i*100,images_mean_std )
  #"""
  
  ###error time plot
  
  
  data_xa_determine = np.mean(data_xa,axis=0)
  data_y0_determine = np.mean(data_y0,axis=0)
  def error(data_fft_small , data3D_txy ): # (t,x,y) , (t,x,y)>>  (t)
    return np.mean((data_fft_small-data3D_txy)**2 ,axis=(1,2))**0.5
  
  timeline = np.arange(0,50+1e-5,1)
  
  
  
  plt.figure(figsize=(8, 6)) 
  plt.plot(timeline, error(data_fft_small , data_xa_determine ),'r',label='xa error') 
  plt.plot(timeline, error(data_fft_small , data_y0_determine ),'b',label='yo error') 
  plt.plot(timeline, error(data_fft_small , data_PG14_small   ),'k',label='PG error')
  plt.xlim([0,50])
  plt.ylim([0,12])
  plt.legend()
  plt.title('Sample Plot') 
  plt.xlabel('timeline') 
  plt.ylabel('error')
  plt.savefig('plot/timeline_error_big.png')
  
  plt.ylim([0,2])
  plt.savefig(f'plot/{file_name}/timeline_error_small.png')
  
  plt.show()
  
  """
  
  ###error2 time plot
  def error2D(data_fft_small , data3D_txy ): #(x,y) , (x,y)+d >> int
    return np.mean((data_fft_small-data3D_txy)**2 )**0.5
    
    
  def error2(data_fft_small , data3D_txy ):
    #(x,y) , (x,y)+d >> int 
    #[int,int,...,int] >> min
    return np.min( [error2D(data_fft_small , data3D_txy+field ) for field in np.linspace(-3,3,30) ] )
    
  data_xa_min_error_mean_timelist , data_xa_min_error_std_timelist = [] , []
  #data_y0_min_error_mean_timelist , data_y0_min_error_std_timelist = [] , []
  
  for timestep in range(51):
    ensemble_list = []
    for each_ensemble in range(K):
      ensemble_list += [ error2(data_fft_small[timestep,:,:] ,data_xa[each_ensemble,timestep,:,:]) ]
    data_xa_min_error_mean_timelist += [ np.mean(ensemble_list) ]
    data_xa_min_error_std_timelist  += [ np.std (ensemble_list) ]
    
    
    
  
  
  plt.figure(figsize=(8, 6)) 
  plt.plot(timeline, data_xa_min_error_mean_timelist           ,'r',label='xa error') 
  
  plt.plot(timeline, error(data_fft_small , data_y0_determine ),'b',label='yo error') 
  plt.plot(timeline, error(data_fft_small , data_PG14_small   ),'k',label='PG error')
  plt.xlim([0,50])
  plt.ylim([0,12])
  plt.legend()
  plt.title('Sample Plot') 
  plt.xlabel('timeline') 
  plt.ylabel('error')
  plt.savefig('plot/timeline_error2_big.png')
  
  plt.ylim([0,2])
  plt.savefig('plot/timeline_error2_small.png')
  
  plt.show()
  """
















