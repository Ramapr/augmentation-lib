import numpy as np
import cv2
from skimage.io import imshow, imsave
from aug_primitives import gamma, rotate, gauss_blur, motion_blur, brightness, contrast, mirror

#%%

photo = cv2.imread('cat.jpg')
photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
imshow(photo)

#%%

def compose(img, aug_list, shuffle=True):
  # замешивает аугментации из list_of_aug
  voc={'gamma':(gamma, img),
       'rotate':(rotate, img),
       'gauss_blur':(gauss_blur, img),
       'motion_blur':(motion_blur, img), 
       'mirror':(mirror, img),
       'contrast':(contrast, img),
       'brightness':(brightness, img)
       }
  if isinstance(aug_list, np.ndarray):
    aug_list = np.array(aug_list)

  if shuffle:
    aug_list = np.random.permutation(aug_list)

  for aug in aug_list:
    funargs = voc[aug][1]
    # voc_out[list_of_tr[args[arg]]] = funargs[2:]
    img = voc[aug][0](funargs)

  return img #out #img

#%%

  
img = compose(photo, ['gamma', 'rotate', 
                      'gauss_blur'],
                      shuffle=True)

imshow(img)
