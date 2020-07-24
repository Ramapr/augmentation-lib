import numpy as np
import cv2
from skimage.io import imshow, imsave

#%%
photo = cv2.imread('cat.jpg')
photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
imshow(photo)

#%%

def rotate(img, rot='rand'):
  prim_list = ['180', 'mirr', '180mirr']
  if rot not in prim_list and rot!='rand':
    raise Exception("rot must be in ['180','mirr', '180mirr'] or 'rand'")
  if rot=='rand':
    rot = prim_list[np.random.randint(0, len(prim_list))]
  if rot == '180':
    return img[::-1,::-1]
  if rot == 'mirr':
    return img[::-1,:]
  if rot == '180mirr':
    return img[:,::-1]
   
#imshow(rotate(photo))
#%%
 
def mirror(img, p='rand'):
  #bound = 0.5
  if p == 'rand':
    p = np.random.randint(0, 2)
  if p == 0:
    return img[::-1,:] 
  else:
    return img[:,::-1]   
  
#imshow(mirror(photo))  
#%%

def gamma(img, gamma='rand'):
  if gamma == 'rand':
    dark_light = np.random.randint(0,2)
    if dark_light==0:
      gamma = np.random.uniform(0.4, 0.92)
    else:
      gamma = np.random.uniform(1.2, 2)
  k_g = 1.0 / gamma
  table = (((np.arange(0, 256) / 255.0) ** k_g) * 255).astype("uint8")
  return cv2.LUT(img, table)

#imshow(random_gamma(photo))
#%%

#def blur(img, kernel):
#  kernel_size = 2 * np.random.randint(1, 4, 2) - 1 
#  out = cv2.GaussianBlur(img, tuple(kernel_size), 0)
#  return out

def gauss_blur(img, kernel_size='rand'):
  # имитирует расфокус
  shape = np.max(img.shape)
  
  if kernel_size=='rand':
    kernel_size = -1 + 2 * np.random.randint(int(np.round(max(shape//150,2),0)),
                                          int(np.round(max(shape//70,3),0)))
  #print(kernel_size)
  kernel_size = (kernel_size,kernel_size)
  out = cv2.GaussianBlur(img, kernel_size, 0)
  return out

#%%

def motion_blur(img, degree='rand', angle='rand'):
  # имитирует смазанное фото со степенью смаза degree и под углом angle 
  # взято тут и доработанно https://www.programmersought.com/article/70021133946/
  shape = np.max(img.shape)
  if degree=='rand':
    degree = np.random.randint(int(np.round(max(shape//100,2),0)),
                            int(np.round(max(shape//70,4),0)))
  if angle=='rand':
    angle = np.random.randint(0, 365)
    
  print(angle, degree)
  # This generates a matrix of motion blur kernels at any angle. The greater the degree, the higher the blur.
  M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
  motion_blur_kernel = np.diag(np.ones(degree))
  motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

  motion_blur_kernel = motion_blur_kernel / degree
  blurred = cv2.filter2D(img, -1, motion_blur_kernel)

  # convert to uint8
  cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
  blurred = np.array(blurred, dtype=np.uint8)
  return blurred
  

#%%

#imshow(motion_blur(photo))
  
#%%

def brightness(img, br='rand'):
  val = {'decr': 0, 'incr':1}
  
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert it to hsv
  if br == 'rand':
    i = np.random.randint(0, 2)
  else:
    if br not in val.keys():
      raise Exception('br parameter must be in (incr, decr)')
    i = val[br]    
  if i == 0:
    #decrease k = 0.35-1
    k = np.random.uniform(0.3, 1.)
    #print(k)
    hsv[..., 2] = hsv[..., 2] * k
  else:    
    # increase
    k = np.random.randint(1, 50)
    #print(k)
    hsv[..., 2] = np.where((255 - hsv[...,2]) < k, 255, hsv[...,2] + k)
  return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#%%  
  
#imshow(brightness(photo, br='incr'))

#%%
def contrast(img, kont='rand'):
  if kont == 'rand':
    kont = np.random.uniform(0.2, 1.6) # 0-1 decrease 1-2 - increase
  #print(k)
  if .2 <= kont <= 1.6:
    return np.clip(np.round(img * kont), 0, 255).astype(np.uint8)
  else:
    raise Exception('Value out of bounds')

#%%  
  
imshow(contrast(photo))


#%%
    
#def random_erase(arr, fig_type='square', fill_param='grey', border=10, max_s=1/6):
#  #fill = 'grey'
#  img = arr.copy()
##  fig_type = 'square'
##  fill_param = 'grey'
##  border = 10 # 0.05 * w or h
##  # calc S
##  k_s = 1 / 6
#  w, h, _ = img.shape
#
#  point_c = np.random.randint(0 + border, 255 - border, 2) # x, y
#  
#  fill = {'black': 0, 
#          'white': 255, 
#          'grey': 128, 
#          'mean':int(np.mean(img)), 
#          'median':int(np.median(img))}
#  
#  #        'rndm':np.random.randint(0, 255, (20, 50, 3))}
#  s_of_figure = max_s * np.random.random()
#  
#  if fig_type == 'circle':
#    #r = int(((w * h * s_of_figure) / np.pi) ** 0.5)
#    r = int((w * h * s_of_figure ) ** 0.5) // 2    
#    
#    ind = np.ones((2*r, 1)) * np.arange(1, 2*r + 1).T
#    indx = ind #+ point_c[0]
#    indy = np.rot90(ind)#ind.T #+ point_c[1]
#    point_in = (indx-r)**2 + (indy-r)**2 <= r**2
#    
#    ind_in = np.argwhere(point_in == True)#, 1, 255)
#    
#    t_x = indx[ind_in] # + point_c[0]
#    t_y = indy[ind_in[:]] + point_c[1]
#    
#    #ind_in[:] += point_c[0]
#    #ind_in[:, 1] += point_c[1]
#    
#    img[ind_in[:, 0], ind_in[:, 1], :] = 128
#    
#    imshow(img)
#    
##  labels = np.ones((arr.shape[0], ), dtype=np.int32)
##  class_ind = (arr[:, 0]-5)**2 + (arr[:, 1] - 8)**2 <= 36
##  ind = np.where(class_ind == True)[0] 
##  labels[ind[:]] = 0
##  return labels
#    
#    
#  if fig_type == 'square':
#    a = int((w * h * s_of_figure ) ** 0.5) // 2    
#    #e, b, c, d = point_c[0] - a, point_c[0] + a, point_c[1] - a, point_c[1] + a
#    if fill_param != 'rndm':
#      img[point_c[0]-a:point_c[0]+a, point_c[1]-a:point_c[1]+a] = fill[fill_param]
#    else:
#      img[point_c[0]-a:point_c[0]+a, point_c[1]-a:point_c[1]+a] = np.random.randint(0, 255, (a*2, a*2, 3)) # shape
#      #ValueError: could not broadcast input array from shape (86,86,3) into shape (75,86,3)
#  return img      
#%% 
#imshow(random_erase(photo, fill_param='rndm'))
