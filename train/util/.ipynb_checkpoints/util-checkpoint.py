import os 
import sys 
import json 
import pickle 
import datetime
import glob
import shutil

import numpy as np 
import nibabel as nb 

from tensorflow.keras.models import load_model
import tensorflow as tf 

import svtools as sv


# -------------------------------
# Resume Training
# -------------------------------

def resume_training(args, model):
    
    # various checks 
    weights = args.resume_training 
    assert os.path.isfile(weights), f"File does not exist"
    assert weights.endswith('.h5'), f"Specify full path to pretrained network"
    assert args.resume_epochs is not None, 'Please specify which epoch to resume from'
    
    # load model 
    #model = load_model(args.resume_training) # WARNING - must change the input to be full path to previously trained model
    loss_dummy = "mean_squared_error"
    model.load_weights(weights) 
    
    # set epoch
    initial_epoch = int(args.resume_epochs)

    # verbose 
    print(f"Resuming training from:\n{args.resume_training}")

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
    if not sv.is_git_repo(os.getcwd()):
        sys.exit(f"\n\nERROR: Not a git repo, please cd into a folder that is part of the git repo and re-run the train procedure")
    sv.save_git_status(checkpoint_path) # save git sha
    shutil.copyfile(args.configfile, checkpoint_path + checkpoint_name + '.yaml') # copy yaml file 
    sv.pickle_dump(checkpoint_path + "args_actual.pkl", args.to_dict())
    
    return checkpoint_path

def _OLD_prepare_checkpoint_dir(args):

    if args.resume_training is not None: 
        checkpoint_path = os.path.dirname(args.resume_training) + "/" 
    else:

        # prepare checkpoint path name 
        noise = f"-noise-{args.noisevariance[0]}to{args.noisevariance[1]}-" if args.noisevariance is not None else ''  # prep noise variance if stored 
        limitinput = f"-limitinput-{args.limit_input_to}-" if args.limit_input_to is not None else ''
        gan = '-GAN' if args.GAN else ''
        transferred = "-transferred" if args.transfer_learning else ''
        checkpoint_path = 'trained_weights/' + 'unet-' + args.name + '-' + get_date() + noise + limitinput + transferred + gan + '/'

    if args.custom_checkpoint_dir is not None:
        assert os.path.exists(args.custom_checkpoint_dir), f"args.custom_checkpoint_dir does not exist: \n{args.custom_checkpoint_dir}"
        checkpoint_path = args.custom_checkpoint_dir + "/" + checkpoint_path

    # save source and args    
    os.makedirs(checkpoint_path, exist_ok=True)
    sv.save_git_status(checkpoint_path)
    sv.save_args(checkpoint_path, args)

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
