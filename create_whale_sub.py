import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os
from IPython.display import FileLink
    
def create_sub(preds, tr_dir, test_dir):
    train_batches = ImageDataGenerator().flow_from_directory(tr_dir, shuffle=False)
    test_batches = ImageDataGenerator().flow_from_directory(test_dir, shuffle=False)
    file_names = np.array([f[f.find('/') +1:] for f in test_batches.filenames])
    
    idx_class = [(v, k) for k,v in train_batches.class_indices.items()]
    idx_class= sorted(idx_class, key=lambda x: x[0])
    classes = [x[1] for x in idx_class]
    data = np.hstack((file_names[:, np.newaxis], preds))
    columns = ['Image'] + classes
    return pd.DataFrame(data, columns=columns)


def create_sub_file(sub, fname):
    sub.to_csv(fname, index=False)
    FileLink(fname)
