import numpy as np
import cv2
from skimage.io import imshow

#%%
photo = cv2.imread('cat.jpg')
photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
imshow(photo)

#%%
def rotate(img):
  img = np.rot90(img)
  if np.random.randint(0, 2) == 1:
    return np.rot90(img)
  return img

#%%
  
#imshow(rotate(photo))

#%%
  
def mirror(img):
  bound = 0.5
  p = np.random.random()
  if p > bound:
    return img[::-1,:] 
  else:
    return img[:,::-1] 
  
#%%
  
#imshow(mirror(photo))  

#%%
def random_erase(arr, fig_type='square', fill_param='grey', border=10, max_s=1/6):
  #fill = 'grey'
  img = arr.copy()
#  fig_type = 'square'
#  fill_param = 'grey'
#  border = 10 # 0.05 * w or h
#  # calc S
#  k_s = 1 / 6
  w, h, _ = img.shape

  point_c = np.random.randint(0 + border, 255 - border, 2) # x, y
  
  fill = {'black': 0, 
          'white': 255, 
          'grey': 128, 
          'mean':int(np.mean(img)), 
          'median':int(np.median(img))}
  
  #        'rndm':np.random.randint(0, 255, (20, 50, 3))}
  s_of_figure = max_s * np.random.random()
  
  if fig_type == 'circle':
    #r = int(((w * h * s_of_figure) / np.pi) ** 0.5)
    r = int((w * h * s_of_figure ) ** 0.5) // 2    
    
    ind = np.ones((2*r, 1)) * np.arange(1, 2*r + 1).T
    indx = ind #+ point_c[0]
    indy = np.rot90(ind)#ind.T #+ point_c[1]
    point_in = (indx-r)**2 + (indy-r)**2 <= r**2
    
    ind_in = np.argwhere(point_in == True)#, 1, 255)
    
    t_x = indx[ind_in] # + point_c[0]
    t_y = indy[ind_in[:]] + point_c[1]
    
    #ind_in[:] += point_c[0]
    #ind_in[:, 1] += point_c[1]
    
    img[ind_in[:, 0], ind_in[:, 1], :] = 128
    
    imshow(img)
    
#  labels = np.ones((arr.shape[0], ), dtype=np.int32)
#  class_ind = (arr[:, 0]-5)**2 + (arr[:, 1] - 8)**2 <= 36
#  ind = np.where(class_ind == True)[0] 
#  labels[ind[:]] = 0
#  return labels
    
    
  if fig_type == 'square':
    a = int((w * h * s_of_figure ) ** 0.5) // 2    
    #e, b, c, d = point_c[0] - a, point_c[0] + a, point_c[1] - a, point_c[1] + a
    if fill_param != 'rndm':
      img[point_c[0]-a:point_c[0]+a, point_c[1]-a:point_c[1]+a] = fill[fill_param]
    else:
      img[point_c[0]-a:point_c[0]+a, point_c[1]-a:point_c[1]+a] = np.random.randint(0, 255, (a*2, a*2, 3)) # shape
      #ValueError: could not broadcast input array from shape (86,86,3) into shape (75,86,3)
  return img      


#%%
  
imshow(random_erase(photo, fill_param='rndm'))

#%%

def random_gamma(img):
  gamma = 7
  k_g = 1.0/ gamma
  table = (((np.arange(0, 256) / 255.0) ** k_g) * 255).astype("uint8")
  # apply gamma correction using the lookup table
  return cv2.LUT(img, table)

#%% 

#imshow(random_gamma(photo))

#%%

def blur(img, kernel):
  kernel_size = 2 * np.random.randint(1, 4, 2) - 1 
  out = cv2.GaussianBlur(img, tuple(kernel_size), 0)
  return out

#%%

#imshow(blur(photo))
  
#%%

def brightness(img):
  #k = np.random.randint(1, 255) - 255 //2 # 0-1 decrease 1-2 - increase
  k = np.random.randint(1, 255 // 3)
  if np.random.rand() > 0.5:
    k *= -1
  print(k)
  return np.clip(img + k, 0, 255)#.astype(np.uint8)

#%%
  
#imshow(brightness(photo))
    
#test = brightness(photo)
#imshow(test)

#%%
def contrast(img):
  k = np.random.uniform(0.1, 2.) # 0-1 decrease 1-2 - increase
  #print(k)
  return np.clip(np.round(img * k), 0, 255).astype(np.uint8)

#%%
  
#imshow(contrast(photo))
#%%
