import numpy as np
import cv2
from skimage.io import imshow, imsave
from aug_primitives import gamma, rotate, gauss_blur, motion_blur, brightness, contrast, mirror

#%%

photo = cv2.imread('cat.jpg')
photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
imshow(photo)

#%%

def compose(img, list_of_aug, shuffle=True):
  # замешивает аугментации из list_of_aug
  if shuffle:
    args = np.random.permutation(np.arange(len(list_of_aug)))
  else:
    args = np.arange(0, len(list_of_aug))
  out = np.zeros((len(list_of_aug), *img.shape), np.uint8)
  for arg in args:
    voc={'gamma':(gamma, img),
         'rotate':(rotate, img),
         'gauss_blur':(gauss_blur, img),
         'motion_blur':(motion_blur, img)}

    funargs = voc[list_of_aug[arg]][1]
    # voc_out[list_of_tr[args[arg]]] = funargs[2:]
    img = voc[list_of_aug[arg]][0](funargs)
    #out[arg, ...] = voc[list_of_aug[arg]][0](funargs)

  return img #out #img

#%%
  
img = compose(photo, ['gamma', 'rotate', 
                      'gauss_blur'],
                      shuffle=True)

imshow(img)
