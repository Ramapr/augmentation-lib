import numpy as np
import pandas as pd
import cv2
from skimage.io import imshow, imsave
from aug_primitives import gamma, rotate, gauss_blur, motion_blur, brightness, contrast, mirror

#%%

def compose(img, aug_list, shuffle=True, log_param=True):
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
    
  if log_param:
    param = {}

  if shuffle:
    aug_list = np.random.permutation(aug_list)
  
  for aug in aug_list:
    func_args = voc[aug][1]
    # voc_out[list_of_tr[args[arg]]] = funargs[2:]
    img, param[aug] = voc[aug][0](func_args)
  
  if log_param:
    return img, param
  else: 
    #print(param)
    return img #out #img

#%%

def gen_rnd_sec(aug, prob): # , n_min=2, n_max=4):  
  if isinstance(aug, np.ndarray):
    aug = np.array(aug)

  if aug.shape[0] != len(prob):
    raise Exception("Augmentation primitivies and probabilities must have equal lenght")
  
  mask = [bool(np.random.choice([0, 1], p=[1 - p, p])) for p in prob]
  #print(mask)  
  aug_cho = aug[mask]
  #print(aug_cho.shape)
  ####  some 
  if 'contrast' in aug_cho and 'brightness' in aug_cho:
    print('cntrs')
    p_cont = prob[np.where(aug == 'contrast')[0][0]]
    p_brig = prob[np.argwhere(aug == 'brightness')[0][0]]
    if p_cont == p_brig:
      del_ind1 = np.where(aug_cho == ('contrast', 'brightness')[np.random.randint(0, 2)])[0][0]
    else:
      if p_cont > p_brig:
        del_ind1 = np.where(aug_cho == 'brightness')[0][0]
      else:
        del_ind1 = np.where(aug_cho == 'contrast')[0][0]
    aug_cho = np.delete(aug_cho, del_ind1)

  if 'gauss_blur' in aug_cho and 'motion_blur' in aug_cho:
    print('blur')
    p_motion = prob[np.where(aug == 'gauss_blur')[0][0]]
    p_blur = prob[np.argwhere(aug == 'motion_blur')[0][0]]
    if p_motion == p_blur:
      del_ind = np.where(aug_cho == ('gauss_blur', 'motion_blur')[np.random.randint(0, 2)])[0][0]
    else:
      if p_motion > p_blur:
        del_ind = np.where(aug_cho == 'gauss_blur')[0][0]
      else:
        del_ind = np.where(aug_cho == 'motion_blur')[0][0]
    aug_cho = np.delete(aug_cho, del_ind)
  
  return aug_cho
    
  

#%%  
aug = np.array(['gamma', 
                'rotate', 
                'gauss_blur', 
                'motion_blur', 
                'mirror',
                'contrast',
                'brightness'
                ])
  
p_ =  [.2, .5, .3, .3, .5, .4, .6] 
#gen_rnd_sec(aug, [.2, .5, .3, .3, .5, .4, .6])

#%%
photo = cv2.imread('cat.jpg')
photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
imshow(photo)


img, p = compose(photo, gen_rnd_sec(aug, p_))
imshow(img)

#%%

######
##
##   Example how to log parameters after parallel execution.
##
######

# test parameter logging


#param = []
#for i in range(5):
#  img, p = compose(photo, gen_rnd_sec(aug, p_))
#  param.append(p)
#  ###db.at[i] = p
#
#dat = pd.DataFrame(data=param, columns=aug)


#%%
