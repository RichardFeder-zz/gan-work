"""

Useful for general NN training.

"""

import numpy as np

import os
import sys
import yaml
import ast
import h5py
import pandas as pd
import math

from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_config(yaml_file):

    with open(yaml_file, 'r') as tfile:
        file_str = tfile.read()
        
    return yaml.load(file_str)

def get_xy_data(fmatrix, ypdict, y_key, **kwargs):
        
    x_data = []
    y_data = []
    
    for index, row in fmatrix.iterrows():
        print(str(row['Patient_Dir']))
        for year_str in ypdict.keys():
            if year_str in str(row['Date']):
                path_to_videos = ypdict[year_str]                
        patient_folder_name = str(row['Patient_Dir'])
#         #Just a check
#         os.listdir(path_to_videos + patient_folder_name + '/' + str(row['Folder']))
                
        full_path = path_to_videos + patient_folder_name + '/' + str(row['Folder'])
        
        vid = vtools.load_video(full_path, **kwargs)
        x_data.append(vid.reshape(vid.shape + (1,)))
        
        y_data.append(np.array(row[y_key]))
        
    print(np.array(x_data).shape)
    print(np.array(y_data).shape)
        
    return x_data, y_data

def batch_to(x, y, batch_size):
    
    x = np.array(x)
    y = np.array(y)
    
    x_batch = []
    y_batch = []
    
    up2 = 0
    for batch in np.array_split(x, math.ceil(len(x)/batch_size)):
        x_batch.append(batch)
        y_batch.append(y[up2:up2+len(batch)])
        up2 += len(batch)
        
    return x_batch, y_batch

def shuffle_generator(x, y, batch_size, steps_per_epoch, noise=None):
    bi = 0
    while True:
        if bi in [0, steps_per_epoch]:
            bi = 0
            x, y = shuffle(x, y)
            x_batch, y_batch = batch_to(x, y, batch_size)
            
        if noise is not None:
            yield(x_batch[bi], y_batch[bi] + np.random.normal(0, noise, 
                                                              size=(len(y_batch[bi]),) ))
        else:
            yield(x_batch[bi], y_batch[bi])
        
        bi += 1

if __name__ == '__main__':

    # Get info from config
    conf_file = sys.argv[1]
    config = load_config(conf_file)
    
    # Get locations of train and test set
    fmatrix = pd.read_pickle(config['feature_matrix'])
    fmatrix = fmatrix[0:100]

    patients_training_set, patients_test_set = train_test_split(np.unique(fmatrix['Patient_Dir']), test_size=0.2)
    train = fmatrix.loc[fmatrix['Patient_Dir'].isin(patients_training_set)]
    test = fmatrix.loc[fmatrix['Patient_Dir'].isin(patients_test_set)]
    
    ypdict = config['paths_to_videos']
    ykey = config['target_var']

    # Get the training params
    train_batch_size = int(config['train_config']['train_batch_size'])
    val_batch_size = int(config['train_config']['val_batch_size'])
    ngpus = int(config['train_config']['ngpus'])
    image_shape = ast.literal_eval(config['train_config']['image_shape'])
    frames_per_vid = int(config['train_config']['frames_per_vid'])
    nepochs = int(config['train_config']['nepochs'])
    
    # Load the videos, swap the image_dim for cv2.resize
    print('training set')
    x_train, y_train = get_xy_data(train, ypdict, ykey,
                                   image_dim=(image_shape[1], image_shape[0]), 
                                   img_type='jpg',
                                   normalize='video',
                                   downsample=True,
                                   frames_per_vid=frames_per_vid)
    print('test set')
    x_test, y_test = get_xy_data(test, ypdict, ykey,
                                 image_dim=(image_shape[1], image_shape[0]), 
                                 img_type='jpg',
                                 normalize='video',
                                 downsample=True,
                                 frames_per_vid=frames_per_vid)

    train_steps_per_epoch = int(np.ceil(len(x_train)/(train_batch_size*ngpus)))
    val_steps_per_epoch = int(np.ceil(len(x_test)/(val_batch_size*ngpus)))

    tsg = shuffle_generator(x_train, y_train, 
                            train_batch_size*ngpus, train_steps_per_epoch, 
                            noise=None)
    vsg = shuffle_generator(x_test, y_test, 
                            val_batch_size*ngpus, val_steps_per_epoch)

    # Get the NN config
    nlevels = int(config['net_config']['nlevels'])
    base_features = int(config['net_config']['base_features'])

    # Build the model
    model = get_model(input_shape=(None,) + image_shape + (1,), 
                      nlevels=nlevels, base_features=base_features, nclasses=3)

    #Compile the model
    op_params = {}
    for keyi in config['optimizer_params']:
        op_params[keyi] = float(config['optimizer_params'][keyi])
    optimizer = Adam(**op_params)

    if ngpus == 1:
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        history = model.fit_generator(tsg, validation_data = vsg,
                                      steps_per_epoch=train_steps_per_epoch, 
                                      validation_steps=val_steps_per_epoch,
                                      epochs=nepochs, verbose=True)
        
        # Save the model
        model.save(config['save_model'])
        
    else:
        from keras.utils import multi_gpu_model
        parallel_model = multi_gpu_model(model, gpus=ngpus)

        parallel_model.compile(loss=config['loss'], optimizer=optimizer)

        history = parallel_model.fit_generator(tsg, validation_data = vsg,
                                               steps_per_epoch=train_steps_per_epoch, 
                                               validation_steps=val_steps_per_epoch,
                                               epochs=nepochs, verbose=True)

        parallel_model.save(config['save_model'])

    # Save the history
    with h5py.File(config['save_history']) as hfile:
        for keyi in history.history:
            hfile.create_dataset(keyi, data=history.history[keyi])
