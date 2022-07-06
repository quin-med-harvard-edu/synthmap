#!/usr/bin/env /home/ch215616/abd/miniconda2/envs/tch2_yml/bin/python

"""
Difference to train_miml.py: 
    - resume training possible 
    - define learning rate 


Original repos are located here: 
- https://github.com/thomas-yu-epfl/Model_Informed_Machine_Learning [Keras training code]
- https://github.com/kelvinlayton/T2estimation [EPG fitting ]
- https://drive.google.com/drive/folders/1IoxOtAt-8NiFgbtZh1RDY32Jb_Wd5TGa [data]

"""

import os
import sys
import math 
import argparse 
import datetime
import re

import numpy as np
import tensorflow as tf
from tensorflow import keras 

from get_train_data import load_training_data
import svtools as sv

def load_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default = None, help='name of the training experiment')
    parser.add_argument('--epochs', type=int,default=300, help='epochs to train for')    
    parser.add_argument('--batchsize',type=int,default=2000)
    parser.add_argument('--savedir', type=str, default = None, help='path to directory where to save the results')
    parser.add_argument('--dataset', type=str, default='te10', help='choose dataset to train with: te9 or te10, or a custom path to dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='define learning rate')
    parser.add_argument('--resume_epoch', type=str, default=None, help='specify which epoch to resume from')
    parser.add_argument('--resume_dir', type=str, default=None, help='specify directory where the weights will be found')
    parser.add_argument('--modeltype', type=str, default='default', help='choose architecture type')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    
            
    args = parser.parse_args()
            
    
    return args 
    

def set_architecture(modeltype='default'):
    if modeltype == 'default':
        #Define the network structure
        inputs = tf.keras.Input(shape=(32,))
        x = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(inputs)
        x = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        x=tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        x=tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        x=tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        x=tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        outputs=tf.keras.layers.Dense(60, activation=tf.keras.activations.softmax, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        #outputs=tf.keras.layers.Dense(2, activation=tf.keras.activations.relu, kernel_initializer='normal',bias_initializer=tf.keras.initializers.Constant(0.1))(x)
        #outputs=tf.clip_by_value(x,90.,180.)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)        
        
    elif modeltype == 'dropout1': 

        dropout=0.2
        #Define the network structure
        inputs = tf.keras.Input(shape=(32,))
        x = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(inputs)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        x = tf.keras.layers.Dropout(dropout)(x)        
        x=tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        x = tf.keras.layers.Dropout(dropout)(x)        
        x=tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        x = tf.keras.layers.Dropout(dropout)(x)        
        x=tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        x = tf.keras.layers.Dropout(dropout)(x)        
        x=tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)        
        outputs=tf.keras.layers.Dense(60, activation=tf.keras.activations.softmax, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        #outputs=tf.keras.layers.Dense(2, activation=tf.keras.activations.relu, kernel_initializer='normal',bias_initializer=tf.keras.initializers.Constant(0.1))(x)
        #outputs=tf.clip_by_value(x,90.,180.)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)    
        
    elif modeltype == 'small': 

        dropout=0.2
        #Define the network structure
        inputs = tf.keras.Input(shape=(32,))
        x = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(inputs)
        x=tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        x=tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)        
        outputs=tf.keras.layers.Dense(60, activation=tf.keras.activations.softmax, kernel_initializer='he_uniform',bias_initializer=tf.keras.initializers.Constant(0.01))(x)
        #outputs=tf.keras.layers.Dense(2, activation=tf.keras.activations.relu, kernel_initializer='normal',bias_initializer=tf.keras.initializers.Constant(0.1))(x)
        #outputs=tf.clip_by_value(x,90.,180.)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)                
        
    else:
        sys.exit('architecture not implemented')
    
    return model 

def TE_vector(nT2=60, nStart=10.0,nEnd=2000.0):
    t2times = np.logspace(math.log10(nStart), math.log10(nEnd), num=nT2, endpoint=True, base=10.0)
    #np.logspace(np.log10(nStart), np.log10(nEnd), num=nT2)
    return t2times

def def_loss(t2times,batch_size = 2000):

    arr=t2times
    arr=np.tile(arr, (batch_size, 1))
    arr_tf=tf.constant(arr.astype('float32'), dtype=tf.float32)

    #Implementation of the Wasserstein Distance
    def wasserstein_distance(y_actual,y_pred):
        #np.abs(np.cumsum(gt_distributions[40,:]-dist_array[40,:])

        abs_cdf_difference=tf.math.abs(tf.math.cumsum(y_actual-y_pred,axis=1))

        return tf.reduce_mean(0.5*tf.reduce_sum(tf.math.multiply(-arr_tf[:,:-1]+arr_tf[:,1:],abs_cdf_difference[:,:-1]+abs_cdf_difference[:,1:]),axis=1))

    #Combination loss function used in MIML
    def MSE_wasserstein_combo(y_actual,y_pred):
        wass_loss=wasserstein_distance(y_actual,y_pred)
        MSE= tf.math.reduce_mean(tf.reduce_mean(tf.math.squared_difference(y_pred, y_actual),axis=1))
        return wass_loss+100000.*MSE
    
    losses = {'wasserstein_distance':wasserstein_distance, 'MSE_wasserstein_combo': MSE_wasserstein_combo}
    return losses 

def get_date():
    return datetime.datetime.now().strftime("%Y%d%m-%H%M")    

def check_args(args):
    
    # load default args if certain parameters aren't specified 
    if args.savedir is None:
        args.savedir = '/home/ch215616/code/mwf/synth_unet/train_MIML/trained_weights/'
    if args.name is None: 
        args.name = 'default_e300'
            
    return args 



# -------------------------------
# Resume Training
# -------------------------------
import glob 
import svtools as sv

def choose_weights(args):
    """Find path to latest weights"""
        
    # get all .h5 files 
    all_weights = glob.glob(args.resume_dir + "/*.hdf5")
    assert all_weights, f"No .hdf5 files found. Check trained_weights dir. Current dir {args.resume_dir}"
    
    
    epoch = args.resume_epoch 
    
    # check if we just need latest epoch
    if epoch == 'latest':
        # find latest epoch - perform natural sort 
        
        weights = sv.natural_sort(all_weights)[-1]
        assert os.path.isfile(weights)

        # get the epoch number 
        basename = os.path.basename(weights)
        match = re.match('weights.[0-9]*', basename).group()
        epoch = match.replace("weights.", "")
        assert epoch.isdigit()
        initial_epoch = int(epoch) + 1 
    
    # else return the specified epoch
    else:
        # if string
        if isinstance(epoch, str):
            # extra check for numbers less than 10
            if len(epoch) == 1: 
                e = '0' + epoch 
            else:
                e = epoch
        # if number 
        elif isinstance(epoch, int) or isinstance(epoch, float):                    
            # convert epoch into ckpt 
            e = str(epoch) if epoch > 9 else '0' + str(epoch)

        # check for this specific epoch 
        ckpt = glob.glob(args.resume_dir + "/weights." + e + "-" + "*.hdf5")
        assert ckpt and len(ckpt) == 1, f"Incorrect ckpt. Ckpt found is: "
        # return it 
        assert os.path.isfile(ckpt[0])
        weights = ckpt[0]

        initial_epoch = int(epoch) + 1 

    return weights, initial_epoch
    
def resume_training(args, model):

    # get path to latest weights 
    weights, initial_epoch = choose_weights(args)

    # verbose 
    print(f"Resuming training from epoch:\n{initial_epoch-1}")

    model.load_weights(weights) 
    
    return model, initial_epoch
    

if __name__ == '__main__': 
    
    # init vars
    args = load_args()
    args = check_args(args)
        
    # load training data 
    trainX, trainY, valX, valY = load_training_data(args.dataset)

    # get echo sampling vector (TE times) in lognorm form   
    if args.dataset.startswith('te'):
        TEs = TE_vector(nT2=60, nStart=10.0,nEnd=2000.0)
        
    # get echo sampling vector in lognorm form that corresponds to values of the synth2D dataset 
    # see build_custom_T2range.py for instructions of how this is generated
    else: 
        from IPython import embed;
        import svtools as sv 
        from build_custom_T2range import get_custom_T2range
        
        T2range_custom_60, _ = get_custom_T2range()
        TEs_standard = TE_vector(nT2=60, nStart=10.0,nEnd=2000.0)
        
        msg = f"Loading custom dataset for synth 2D data. \nPlease check T2range_custom_60 values vs TEs_standard. They should be similar. \n Type exit to continue. "
        embed(header=sv.msg(msg))
        
        # export to correct variable name 
        TEs = T2range_custom_60
        
    # define losses function (inc wasserstein distance)
    losses = def_loss(TEs,batch_size=args.batchsize)

    # build model architecture
    model = set_architecture(modeltype=args.modeltype)

    # set optimizer 
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # compile 
    model.compile(optimizer=opt,
                  loss=losses['MSE_wasserstein_combo'],
                  metrics=['mse',losses['wasserstein_distance']])

    # resume training 
    if args.resume_epoch is not None: 
        model, initial_epoch = resume_training(args, model)
        
        ckpt_path=args.resume_dir + '/'

    else: 
        
        # save inputs 
        ckpt_path=args.savedir + '/' + args.name + get_date() + '/'
        os.makedirs(ckpt_path,exist_ok=True)
        sv.save_git_status(ckpt_path)
        sv.save_args(ckpt_path, args)    
        
        initial_epoch=0

    # define callbacks 
    ckpt_callback=tf.keras.callbacks.ModelCheckpoint(ckpt_path+'weights.{epoch:02d}-{val_loss:.2f}.hdf5', 
                                                           monitor='val_loss', 
                                                           verbose=0, 
                                                           save_best_only=True, 
                                                           save_weights_only=True, 
                                                           mode='auto',
                                                           save_freq='epoch')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=ckpt_path+"/logs")
    early_stop_callback = tf.keras.callbacks.EarlyStopping('val_loss', patience=args.patience)    
    callbacks = [ckpt_callback, tensorboard_callback,early_stop_callback]
    
    # Train 
    model.fit(trainX,trainY,epochs=args.epochs, batch_size=args.batchsize, validation_data=(valX,valY),callbacks=callbacks,initial_epoch=initial_epoch) 
