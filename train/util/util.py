import os 
import sys 
import json 
import pickle 
import datetime
import glob
import shutil
import re 

import numpy as np 
import nibabel as nb 

from tensorflow.keras.models import load_model
import tensorflow as tf 

import svtools as sv


# -------------------------------
# Resume Training
# -------------------------------

def choose_weights2(args,epoch):
    """Find path to latest weights"""
        
    # get all .h5 files 
    all_weights = glob.glob(args.checkpoint_path + "/*.h5")
    assert all_weights, f"No .h5 files found. Check trained_weights dir. Current dir {args.checkpoint_path}"
    
    # check if we just need latest epoch
    if epoch == 'latest':
        # find latest epoch - perform natural sort 
        
        weights = sv.natural_sort(all_weights)[-1]
        assert os.path.isfile(weights)

        # get the epoch number 
        basename = os.path.basename(weights)
        match = re.match('ckpt-[0-9]*', basename).group()
        epoch = match.replace("ckpt-", "")
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
        ckpt = glob.glob(args.checkpoint_path + "/ckpt-" + e + "-" + "*.h5")
        assert ckpt and len(ckpt) == 1, f"Incorrect ckpt. Ckpt found is: "
        # return it 
        assert os.path.isfile(ckpt[0])
        weights = ckpt[0]

        initial_epoch = int(epoch) + 1 

    return weights, initial_epoch


def resume_training(args, model):
    
    

    if args.resume_training == 'latest': 
        # get path to latest weights 
        weights, initial_epoch = choose_weights2(args, epoch='latest')

        # verbose 
        print(f"Resuming training from epoch:\n{initial_epoch-1}")

    elif str(args.resume_training).isdigit():
        epoch = int(args.resume_training)
        weights,initial_epoch = choose_weights2(args, epoch=epoch)

        # verbose 
        print(f"Resuming training from epoch:\n{initial_epoch-1}")

    else:
        # choose based on specific path 
        weights = args.resume_training 
        assert os.path.isfile(weights), f"File does not exist"
        assert weights.endswith('.h5'), f"Specify full path to pretrained network"
        assert args.resume_epochs is not None, 'Please specify which epoch to resume from'

        # set epoch 
        initial_epoch = int(args.resume_epochs)

        # verbose 
        print(f"Resuming training from:\n{args.resume_training}")
    
    # load model 
    #model = load_model(args.resume_training) # WARNING - must change the input to be full path to previously trained model
    #loss_dummy = "mean_squared_error"
    
    model.load_weights(weights) 
    
    return model, initial_epoch


# -------------------------------
# Input GPU 
# -------------------------------

def select_gpu(gpu_number):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_number], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical (Selected) GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

# -------------------------------
# History
# -------------------------------


def save_history(history, checkpoint_path):
    json.dump(history.history, open(checkpoint_path+'history.json', 'w'))
    model_history = History(history.history, history.epoch, history.params)
    pickle.dump(model_history,open(checkpoint_path+'history.pkl', 'wb'))


# WARNING: UNUSED 
def plot_history(history):
    import matplotlib.pyplot as plt 

    # Plot history: MSE
    plt.plot(history.history['mean_squared_error'], label='MSE (training data)')
    plt.plot(history.history['val_mean_squared_error'], label='MSE (validation data)')
    # plt.title('MSE for Chennai Reservoir Levels')
    plt.ylabel('MSE value')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()

# WARNING: UNUSED 
class History(object):
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params


# -------------------------------
# Input Images
# -------------------------------

# -------------------------------
# Input Arguments
# -------------------------------

def prepare_checkpoint_dir(args):

    # extract name 
    checkpoint_name = os.path.basename(args.configfile.replace('.yaml',''))
    checkpoint_path = checkpoint_name + "-" + args.name + '/' if args.name else checkpoint_name + "/"

    # set path 
    if args.custom_checkpoint_dir is not None:
        assert os.path.exists(args.custom_checkpoint_dir), f"args.custom_checkpoint_dir does not exist: \n{args.custom_checkpoint_dir}"
        checkpoint_path = args.custom_checkpoint_dir + "/" + checkpoint_path
    else:
        checkpoint_path = 'trained_weights/' + checkpoint_path  
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # save source and args  
    datetime = get_date()
    if os.path.exists("/home/ch215616/w/"): # save only on rayan
        if not sv.is_git_repo(os.getcwd()):
            sys.exit(f"\n\nERROR: Not a git repo, please cd into a folder that is part of the git repo and re-run the train procedure")
        sv.save_git_status(checkpoint_path, prefix=datetime + "_") # save git sha

    yamlpath = checkpoint_path + datetime + "_" + checkpoint_name + '.yaml'
    shutil.copyfile(args.configfile, yamlpath) # copy yaml file 
    sv.pickle_dump(checkpoint_path + datetime + "_" + "args_actual.pkl", args.to_dict())
    
    return checkpoint_path


def check_if_e2():

    # check if execution is performed on e2, which requires path substituion for args. 

    import socket
    host = socket.gethostname()

    if not host in ['rayan', 'ankara', 'carlsen', 'istanbul', 'coffee']:
        print('executing on e2')

# check args 
def check_input_args(args):

    if args.mode == 'test':
        
        # check if weights exist
        assert args.trained_weights, f"Please provide path to trained weights"
        assert os.path.isfile(args.trained_weights), f"trained weights path is incorrect"

        # check if savedir is not empty 
        assert args.savedir, f"Savedir cannot be empty. Please specify location"
        
        # check if input folder exists 
        if args.test_dir is not None: 
            args.test_dir = args.test_dir + '/' if not args.test_dir.endswith('/') else args.test_dir
            assert os.path.isdir(args.test_dir), f"Incorrect test input"
        elif args.test_regexp is not None: 
            assert args.test_regexp, f"Incorrect test input. Regexp returned empty set. "
        elif args.test_file is not None: 
            assert os.path.isfile(args.test_file), f"Incorrect test input file. Did you intend to pass a directory with test_dir input variable instead?  :\n{args.test_file}"
        else: 
            sys.exit("No input specified. please specify directory or subject (via regexp) or file")


    elif args.mode == 'train':
	
        # correct dir strings
        if '*' not in args.x:
            args.x = args.x + '/' if not args.x.endswith('/') and not args.x.endswith('.txt') else args.x
            assert os.path.exists(args.x), f"Training path for X does not exist"
        else:
            assert glob.glob(args.x), f"No files found in {args.x}"
        if not args.selfsupervised: 
            if '*' not in args.y:
                args.y = args.y + '/' if not args.y.endswith('/') and not args.y.endswith('.txt') else args.y
                assert os.path.exists(args.y), f"Training path for Y does not exist"
            else:
                assert glob.glob(args.y), f"No files found in {args.y}"
        assert args.valsize>=args.batchsize, f"Validation size must be bigger or equal to batch size."
    
    return args

def get_date():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M")    


# -------------------------------
# Output
# -------------------------------


def save_images_w_epoch(savedir,predictions, input_paths,epoch=None, verbose=True):

    """ Save images as NIFTI files (default)"""
    if not os.path.exists(savedir):
        os.makedirs(savedir,exist_ok=True)
        
    if epoch is not None: 
        savedir = savedir +"/epoch_" + str(epoch) + "/"
        os.makedirs(savedir,exist_ok=True)
                
    if verbose:            
        print(f"Saved files to: {savedir}")

        
    # save images 
    for im,ref in zip(predictions,input_paths):
        # get reference imo
        refo = nb.load(ref)        
        
        # predictions are returned in [x,y,channels] format. Split into [x,y,z,channels] format,where channel corresponds to parameters
        im = np.expand_dims(im,2)
        # update header 
        header = refo.header
        header['dim'][4] = im.shape[-1]

        imo = nb.Nifti1Image(im,refo.affine,header)
        savename = savedir + "/" + os.path.basename(ref).replace('.nii.gz', '_pred.nii.gz')
        nb.save(imo,savename)


def save_images(savedir,predictions, input_paths,verbose=True):

    """ Save images as NIFTI files (default)"""

    if not os.path.exists(savedir):
        os.makedirs(savedir,exist_ok=True)
        if verbose:
            print(f"Created new directory: {savedir}")
    if verbose:            
        print(f"Saved files to: {savedir}")
    
    # save images 
    for im,ref in zip(predictions,input_paths):
        
        # get reference imo
        refo = nb.load(ref)                

        # predictions are returned in [x,y,channels] format. Split into [x,y,z,channels] format,where channel corresponds to parameters
        im = np.expand_dims(im,2)
        # update header 
        header = refo.header
        header['dim'][4] = im.shape[-1]

        imo = nb.Nifti1Image(im,refo.affine,header)
        savename = savedir + os.path.basename(ref).replace('.nii.gz', '_pred.nii.gz')
        nb.save(imo,savename)
