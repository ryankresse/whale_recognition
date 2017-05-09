import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os

def create_sub(preds):
    train_batches = ImageDataGenerator().flow_from_directory(os.getcwd()+'/data/whale/imgs/train', shuffle=False)
    test_batches = ImageDataGenerator().flow_from_directory(os.getcwd()+'/data/whale/imgs/test', shuffle=False)
    file_names = np.array([f[f.find('/') +1:] for f in test_batches.filenames])
    
    idx_class = [(v, k) for k,v in train_batches.class_indices.items()]
    idx_class= sorted(idx_class, key=lambda x: x[0])
    classes = [x[1] for x in idx_class]
    data = np.hstack((file_names[:, np.newaxis], preds))
    columns = ['Image'] + classes
    return pd.DataFrame(data, columns=columns)