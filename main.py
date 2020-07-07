#import 
import pandas as pd
import numpy as np


def rnd_aug_ind(ind, dict: aug_list):
    """
    для каждого типа аугментации выбираем 
    индексы изображений из класса и перемешиваем
    """
    pass
    max_len = ind.shape[0]
    out = {}

    for aug_name, prob in aug_list.items():
        if prob == 1 or prob == 1. :
            out[aug_name] = ind[:]
        #elif prob == 0:
        else:   
            out[aug_name] = np.permutate(ind)[:int(prob*max_len)]

    return out


def choose_aug(K_expanse, aug_list, data, tag_class):
    """
    K_expanse - 
    aug_list - 
    data - 
    tag_class - df.slice
    """
    pass
    # подумать про соотношение???
    #np.unique(tag_class)
    for _class in np.unique(tag_class):
        pass
        class_ind = np.argwhere(tag_class == _class)[0]
        # индексы класса
        out = rnd_aug_ind(class_ind, aug_list)



