import os
import random
import numpy as np
import sys 
import pathlib 
import nibabel as nb 
import time

import tensorflow as tf
from tensorflow import keras

# local imports 
from models.unet import unet, ivim_loss #, ivim_loss2 unet_ivim, unet_ivim2,  # deprecated
from models.transfer_learn import transfer_learning
from models.keras_tuner import select_best_hyperparameters
#from options.options import TrainArgs # deprecated
from options.options_extra import Args,ArgsExtra   # THIS SHOULD BE RENAMED
from data.train_data import get_data,get_data_ivim, split_data, get_val_data, get_test_data, read_txt, dataids # THIS SHOULD BE RENAMED INTO DATAIO
from data.generator_w_augmentation import DataGenerator_train, DataGenerator_from_ids
from util.util import (save_history,
    get_date, # not used
    resume_training, # this should be moved into model definition 
    check_input_args,
    prepare_checkpoint_dir,
    save_images
)


# UNUSED
# from time import time
  
# def timer(func):
#     # This function shows the execution time of 
#     # the function object passed
#     def wrap_func(*args, **kwargs):
#         t1 = time()
#         result = func(*args, **kwargs)
#         t2 = time()
#         print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
#         return result
#     return wrap_func

def get_loss(args):
    return {'mse':'mean_squared_error','ivim_loss':ivim_loss}[args.loss]

def set_callbacks(args):

    callbacks = []

    # checkpoint callback 
    filepath = args.checkpoint_path +"ckpt-{epoch:02d}-{val_loss:.2f}.h5"
    modelCheckpoint_callback = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',save_best_only=args.save_best_only,mode='min', period=args.save_frequency,save_weights_only=args.save_weights_only)
    callbacks.append(modelCheckpoint_callback)

    # Tensorboard callback 
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=args.checkpoint_path+"/logs")
    callbacks.append(tensorboard_callback)

    # LR schedule callback
    if 'lr_schedule' in args: 
        if args.lr_schedule is not None:
            global schedule 
            schedule=args.lr_schedule
            lr_scheduler_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
            callbacks.append(lr_scheduler_callback)

    if args.early_stop_patience is not None:
        early_stop_callback = keras.callbacks.EarlyStopping('val_loss', patience=args.early_stop_patience)    
        callbacks.append(early_stop_callback)
    
    return callbacks

def lr_scheduler(epoch, lr):
    """
    Fetch a global variable 'schedule' and update learning rate (if the current epoch is listed in the schedule). 

    You must set 'schedule' as a global var in this manner: 
        schedule (dict): key:value pairs that correspond to 'epoch':learning_rate - e.g. {'0':0.01, '10':0.001, '100':0.0001, '300':0.00001}

    Args:
        epoch (int): current epoch. Provided by model.fit automatically 
        lr (float): current learning rate. Provided by model.fit automatically
    Returns: 
        lr (float): new learning rate. 
    """

    # check if current epoch is a key inside 'schedule' 
    if str(epoch) in schedule: 
        # return new learning rate
        lr = schedule[str(epoch)] 
        print(f"ATTENTION: Learning Rate changed to: {lr}")
        return lr 
    else: 
        # return original learning rate 
        return lr  

    """
    ALT version 1: lr is divided by X at specified epochs
        e.g. 
        schedule (dict): key:value pairs that correspond to 'epoch':lr_reduction_rate - e.g. {'10':10, '100':10, '300':10} - will divide lr by 10 for each of the listed epochs - 10,100,300

    # check if current epoch is a key inside 'schedule' 
	if str(epoch) in schedule: 
        # return new learning rate 
		return lr/schedule[str(epoch)] 
	else: 
        # return original learning rate 
		return lr  
    """

    """
    ALT version 2: lr is kept constant for first 10 epochs and then decreased exponentially

	if epoch < 10: 
        return lr
	else: 
        return lr * tf.math.exp(-0.1)

    """

class PredictionCallback2(keras.callbacks.Callback):
    
    # saves ground truth, prediction and input signal image into separate folders
    
    # source https://stackoverflow.com/questions/36864774/python-keras-how-to-access-each-epoch-prediction
    # source: https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L572-L877
    
    def __init__(self,x_y_paths, data_loader, savedir, input_type="val"):    
        super(PredictionCallback2, self).__init__()
        self.x_paths, self.y_paths = x_y_paths
        self.data_loader = data_loader
        self.construct_paths(savedir, input_type)

        
    def construct_paths(self, savedir, input_type):
        
        # construct paths 
        self.savedir = {'x':None, 'y':None, 'pred':None}
        for k in self.savedir.keys(): 
            self.savedir[k] = savedir + "/predictions/" + input_type + "/" + k + "/"
            pathlib.Path(self.savedir[k]).mkdir(parents=True,exist_ok=True)
        
        

    def on_epoch_end(self,epoch,logs):
        
        start_c = time.time()        
        # Run model
        predictions = self.model.predict(self.data_loader)        
        
        ###
        # save predictions 
        ###
        
        savedir = self.savedir['pred'] + "/e" + str(epoch) + "/"
        save_images(savedir, predictions, self.y_paths,verbose=False)         
        
        
        ###
        # save x and y 
        ###

        # load raw images              
        # ... index 0 will fetch the same data since size of self.data_loader is never larger than args.batch_size
        # ... class of self.data_loader : data.generator_w_augmentation.DataGenerator_train -> will generate lists of images
        x,y = self.data_loader[0]        
        
        # verify that length of predictions is not longer than length of image paths supplied 
        num_pred_images = predictions.shape[0] 
        num_image_paths = len(self.y_paths)
        assert num_pred_images == num_image_paths, f"Number of image paths supplied to generator is not equal to number of predicted images. Check the length of list supplied to data generator"
        end_c = time.time()
        print(f"\nComputing validation and test images step: {end_c-start_c:0.2f}s\n")

        # save
        start = time.time()
        x_savedir = self.savedir['x'] + "/e" + str(epoch) + "/"
        y_savedir = self.savedir['y'] + "/e" + str(epoch) + "/"
        save_images(x_savedir, x, self.x_paths,verbose=False)                         
        save_images(y_savedir, y, self.y_paths,verbose=False)   
        end = time.time()
        print(f"\nSaving validation and test images step: {end-start:0.2f}s\n")
        
        

def datapaths_test(args):    
    # fetches paths to test (real) data so that we could save network performance on each epoch on test images (for quick view).
        
    Xtest_paths, Ytest_paths = get_test_data(args) 
    
    return Xtest_paths, Ytest_paths

  
        
def datapaths(args):

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
            
        

    return train_X_paths, train_Y_paths, val_X_paths, val_Y_paths

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
    # Switch the directories from '/home/ch215616/w/' to '/home/ch215616/fs/' to '/home/ch215616/scratch/' to '/scratch/ch215616/w/'
    
    
    alt_dirs = ['/home/ch215616/w/','/scratch/ch215616/w/', '/home/ch215616/fs/', '/home/ch215616/scratch/']
    exist_dirs=[d for d in alt_dirs if os.path.exists(d)]
    assert exist_dirs
    key_dir = exist_dirs[0]
    
    
    if not key_dir == alt_dirs[0]:
        # substitute the dirs in configfile 
        args.configfile = args.configfile.replace(alt_dirs[0], key_dir)
        
        # substitute the dirs in all public props
        keys = vars(args)['_public_props']
        
        for k in keys:
            # check if string 
            if isinstance(args[k], str):
                # check if directory structure 
                if alt_dirs[0] in args[k]:
                    # replace all values
                    #args[k].replace(alt_dirs[0], key_dir)
                    
                    args[k] = args[k].replace(alt_dirs[0], key_dir)
                    
                    if not os.path.exists(args[k]):
                        print(f"Warning: Directory does not exist: {args[k]}")
                    else:
                        print(f"New directory:{args[k]}")
                        
        
        
           

    return args      

def update_userpath(args):

    # get user path 
    up=os.path.expanduser('~')

    # check every variable
    variables = vars(args)['_public_props']
    for v in variables:
        if isinstance(args[v], str):
            if '~' in args[v]: 
                print(v)
                args[v] = args[v].replace('~', up)

    return args
                

if __name__ == '__main__':

    # read config file
    args = ArgsExtra().parse() if len(sys.argv) > 3 else Args().parse()

    # check paths for ~ and expand user
    args = update_userpath(args)

    # Fix all random seeds for reproducibility
    fix_random_seeds()

    # check args 
    args = check_hostname(args)
    if not args.x.endswith(".txt"): # only necessary when input args are lists of *.nii.gz files
        args = check_input_args(args)

    # Checkpoint dir
    args.checkpoint_path = prepare_checkpoint_dir(args)

    # select GPU    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu is not None else 0

    # check valsize 
    assert args.valsize>=args.batchsize, f"Validation size must be bigger or equal to batch size."
    
    # generators for the neural network training
    args.shape = tuple(args.shape)
        

    # get paths to train and val data
    if args.xdir is not None: 
        # get ids 
        print("WARNING: fetching dataids!")
        ids = dataids(args)

        # shuffle (along first axis only)
        np.random.seed(args.seed)
        np.random.shuffle(ids)
        
        # split data into val and train 
        train_ids = ids[:-args.valsize, :]
        val_ids = ids[-args.valsize:, :]

        # limit to 
        if args.limit_input_to is not None:
            train_ids = train_ids[:args.limit_input_to, :]        

        # generators 
        train_gen = DataGenerator_from_ids(args, args.shape, args.input_nc, train_ids)
        val_gen = DataGenerator_from_ids(args, args.shape, args.input_nc, val_ids)
    else:
        train_X_paths, train_Y_paths, val_X_paths, val_Y_paths = datapaths(args)
        assert len(val_X_paths)>=args.batchsize    
        train_gen = DataGenerator_train(args, args.shape, args.input_nc, train_X_paths, train_Y_paths)
        val_gen = DataGenerator_train(args, args.shape, args.input_nc, val_X_paths, val_Y_paths)
    
        # generator for saving predictions on val data at each epoch
        valXp, valYp  = val_X_paths[:args.batchsize], val_Y_paths[:args.batchsize], 
        predict_val_gen = DataGenerator_train(args, args.shape, args.input_nc, valXp, valYp,noaugment=True)
    
        # generator for saving predictions on test data at each epoch
        if args.xtest is not None: 
            test_X_paths, test_Y_paths = datapaths_test(args)    
            testXp, testYp = test_X_paths[:args.batchsize], test_Y_paths[:args.batchsize]
            predict_test_gen = DataGenerator_train(args, args.shape, args.input_nc, testXp, testYp,noaugment=True)
    
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Initial epoch
    initial_epoch = 0 
    
    # Get custom loss
    loss = get_loss(args)

    # Build a model
    if args.kerastuner:

        # search a range of parameters and return the best model (to be used in full training) 
        model = select_best_hyperparameters(args,args.checkpoint_path,train_loader, val_loader, loss)

    else:    
        model = unet(args.shape,args.input_nc, loss, args.input_type,otherparams=args)     # WARNING: this should be the same class as HyperUnet, just without the extension
        if args.resume_training is not None:
            model, initial_epoch = resume_training(args, model)               # update weights and epoch number
        elif args.transfer_learning:
            model = transfer_learning(args,model)

    # Set callbacks 
    callbacks = set_callbacks(args)

    if 'visualize_val' in args:
        if args.visualize_val is not None:
            predict_callback_val = PredictionCallback2((valXp, valYp), predict_val_gen, args.checkpoint_path, "val")
            callbacks.append(predict_callback_val)    
    if 'visualize_train' in args: 
        if args.visualize_train is not None:
            sys.exit('Not implemented')
            predict_callback_train = PredictionCallback2((train_X_paths, train_Y_paths), predict_val_gen, args.checkpoint_path, "val")
            callbacks.append(predict_callback_train)    

    if 'visualize_test' in args: 
        if args.visualize_test is not None:
            predict_callback_test = PredictionCallback2((testXp, testYp), predict_test_gen, args.checkpoint_path, "test")
            callbacks.append(predict_callback_test)

    # Train 
    history = model.fit(train_gen, epochs=args.epochs, validation_data=val_gen, callbacks=callbacks, initial_epoch=initial_epoch)


    # Save history as .json and .pkl
    #save_history(history,args.checkpoint_path)

