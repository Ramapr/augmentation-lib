import numpy as np
import cv2
from skimage.io import imshow, imsave

#%%
#photo = cv2.imread('cat.jpg')
#photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
#imshow(photo)

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
#  0.7 - 0.9
# 7 - 40
def brightness(img, k='rand'):
  #val = {'decr': 0, 'incr':1}
  
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert it to hsv
  if k == 'rand':
    v = [- np.random.randint(2, 30), np.random.randint(7, 40)]
    k = v[np.random.randint(0, 2)]
  h, s, v = cv2.split(hsv)
  v = cv2.add(v, k)
  v[v > 255] = 255
  v[v < 0] = 0
  out = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
  return out

#%%  

#imshow(brightness(photo, k=-30))

#%%
def contrast(img, kont='rand'):
  if kont == 'rand':
    kont = np.random.uniform(0.8, 1.2) # 0-1 decrease 1-2 - increase
  return np.clip(np.round(img * kont), 0, 255).astype(np.uint8)

#%%  

#imshow(photo)  
#  
#imshow(contrast(photo, kont=1.2))

#%%