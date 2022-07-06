import os
import random
import numpy as np

import tensorflow as tf
from tensorflow import keras

# local imports 
from models.unet import unet, ivim_loss #, ivim_loss2 unet_ivim, unet_ivim2,  # deprecated
from models.transfer_learn import transfer_learning
from models.keras_tuner import select_best_hyperparameters
#from options.options import TrainArgs # deprecated
from options.options_extra import Args,ArgsExtra  # THIS SHOULD BE RENAMED
from data.train_data import get_data,get_data_ivim, split_data, get_val_data, read_txt # THIS SHOULD BE RENAMED INTO DATAIO
from data.generator import DataGenerator
from util.util import (
    get_date, # not used
    save_history, 
    resume_training, # this should be moved into model definition 
    check_input_args,
    prepare_checkpoint_dir
)

def get_loss(args):
    return {'mse':'mean_squared_error','ivim_loss':ivim_loss}[args.loss]

def set_callbacks(args,checkpoint_path):

    filepath = checkpoint_path+"ckpt-{epoch:02d}-{val_loss:.2f}.h5"
    modelCheckpoint_callback = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',save_best_only=args.save_best_only,mode='min', period=args.save_frequency,save_weights_only=args.save_weights_only)

    # set early stopping 
    early_stop_callback = keras.callbacks.EarlyStopping('val_loss', patience=args.early_stop_patience)    

    # set Tensorboard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=checkpoint_path+"/logs")
    callbacks = [modelCheckpoint_callback, tensorboard_callback,early_stop_callback]
    return callbacks

def dataloaders(args):

    # get data paths and size
    X_paths, Y_paths = get_data(args) if args.input_type == 'mwf' else get_data_ivim(args)

    # Split into train and val
    if args.xval is None: 
        train_X_paths, train_Y_paths, val_X_paths, val_Y_paths = split_data(args,X_paths,Y_paths)
    else: 
        assert args.xval is not None and args.yval is not None, f"Both xval and yval needs to be not null in order to be used"
        
        # leave train paths as is 
        train_X_paths = X_paths 
        train_Y_paths = Y_paths
        
        # add val paths separately 
        assert args.training_data_type is None, f"Please make sure that no datatype is specified when using separate val dataset. This will assure that 'else' statement is used inside 'get_data' function in data/train_data.py/get_data()"
        if not args.xval.endswith(".txt"):
            val_X_paths, val_Y_paths = get_val_data(args) 
        else: 
            assert args.xval.endswith(".txt")
            val_X_paths = read_txt(args.xval)
            val_Y_paths = read_txt(args.yval)
            val_X_paths = sorted(val_X_paths)
            val_Y_paths = sorted(val_Y_paths)
            
        
        

    # Instantiate generator for train and val 
    args.shape = tuple(args.shape)
    train_gen = DataGenerator(args, args.shape, args.input_nc, train_X_paths, train_Y_paths)
    val_gen = DataGenerator(args, args.shape, args.input_nc, val_X_paths, val_Y_paths)

    return train_gen, val_gen

def fix_random_seeds():

    # Random number generation is used in these functions:
        # generator.py 
        # train_data.py 
        # unet.py (random_normal_initializer)

    # Seed value
    seed_value= 0

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)



def check_hostname(args):

    # we are possibly running the algorithm on a machine that does not have access to Rad-Warfield-e2 (e.g. ankara, coffee)
    # Switch the directories from '/home/ch215616/w/' to '/home/ch215616/abd/'
    if not os.path.exists('/home/ch215616/w/'): 
        args.x = args.x.replace('/home/ch215616/w/', '/home/ch215616/abd/')
        if args.y:
            args.y = args.y.replace('/home/ch215616/w/', '/home/ch215616/abd/') 
        if args.custom_checkpoint_dir:
            args.custom_checkpoint_dir = args.custom_checkpoint_dir.replace('/home/ch215616/w/', '/home/ch215616/abd/') 

        if args.mode == 'test':
            sys.exit('please implement check_hostname function for testing')

    return args      

if __name__ == '__main__':

    # read config file
    args = ArgsExtra().parse() if len(sys.argv) > 3 else Args().parse()

    # Fix all random seeds for reproducibility
    fix_random_seeds()

    # check args 
    args = check_hostname(args)
    if not args.x.endswith(".txt"): # only necessary when input args are lists of *.nii.gz files
        args = check_input_args(args)


    # Checkpoint dir
    checkpoint_path = prepare_checkpoint_dir(args)

    # select GPU    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu is not None else 0
        

    # Get data loaders
    train_loader, val_loader = dataloaders(args)

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Initial epoch
    initial_epoch = 0 
    
    # Get custom loss
    loss = get_loss(args)

    # Build a model
    if args.kerastuner:

        # search a range of parameters and return the best model (to be used in full training) 
        model = select_best_hyperparameters(args,checkpoint_path,train_loader, val_loader, loss)

    else:    

        if args.resume_training is not None:
            model, initial_epoch = resume_training(args)
        else:
            model = unet(args.shape,args.input_nc, loss, args.input_type)     # WARNING: this should be the same class as HyperUnet, just without the extension
                    
            if args.transfer_learning:
                model = transfer_learning(args,model)
        
    # Set callbacks 
    callbacks = set_callbacks(args,checkpoint_path)

    # Train 
    history = model.fit(train_loader, epochs=args.epochs, validation_data=val_loader, callbacks=callbacks, initial_epoch=initial_epoch)

    # Save history as .json and .pkl
    save_history(history,checkpoint_path)

