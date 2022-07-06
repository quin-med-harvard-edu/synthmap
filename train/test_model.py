import os
import glob 
import sys 
import math 

from tensorflow import keras
from models.unet import unet

# local imports 
from options.options_extra import Args,ArgsExtra  
from util.util import check_input_args, save_images,save_images_w_epoch
from data.test_data import get_data
from data.generator import DataGenerator


import svtools as sv

def choose_weights(args):
    """Find exact path to saved weights based on supplied epoch in the config file"""
    
    # if full path is supplied - just return that (legacy)
    if args.trained_weights.endswith(".h5"):
        return args.trained_weights
    
    # if directory is supplied
    elif os.path.isdir(args.trained_weights):
        
        # basic checks
        assert 'epoch' in vars(args), f"Epoch is not specified for test_model.py"
        epoch = args.epoch 
        
        # get all .h5 files 
        all_weights = glob.glob(args.trained_weights + "/*.h5")
        assert all_weights, f"No .h5 files found. Check trained_weights dir. Current dir {args.trained_weights}"
        
        # check if we just need latest epoch
        if epoch == 'latest':
            # find latest epoch - perform natural sort 
            
            latest_epoch = sv.natural_sort(all_weights)[-1]
            assert os.path.isfile(latest_epoch)
            return latest_epoch
        
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
            ckpt = glob.glob(args.trained_weights + "/ckpt-" + e + "-" + "*.h5")
            assert ckpt and len(ckpt) == 1, f"Incorrect ckpt. Ckpt found is: "
            # return it 
            assert os.path.isfile(ckpt[0])
            return ckpt[0]


if __name__ == '__main__':

    # read config file
    args = ArgsExtra().parse() if len(sys.argv) > 3 else Args().parse()

    # check args 
    #args = check_input_args(args)

    # select GPU    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # get data paths  
    test_X_paths = get_data(args)

    # update batchsize if it is larger than the number of images that we have (else error at .predict)
    #args.batchsize = len(test_X_paths) if len(test_X_paths)<args.batchsize else args.batchsize
    # make sure that len(test_X_paths) % batchsize == 0, else data_loader will only return a shortned vector of args.batchsize in length, or give an error. We fix this by setting batch to highest common denominator between current batch and the length of the test set 
    args.batchsize = math.gcd(len(test_X_paths),args.batchsize) if len(test_X_paths)%args.batchsize !=0 else args.batchsize
    
    # Get data loaders
    args.shape = tuple(args.shape)
    data_loader = DataGenerator(args, args.shape, args.input_nc, test_X_paths, '')    
    
    # Free up RAM in case the model definition were run multiple times
    keras.backend.clear_session()

    # Load model 
    loss_dummy = "mean_squared_error"
    model = unet(args.shape,args.input_nc, loss_dummy, args.input_type, otherparams=args)     # WARNING: this should be the same class as HyperUnet, just without the extension
    weights = choose_weights(args)
    model.load_weights(weights) 
    
    # for item in ds:
    #     xi, yi = item
    #     pi = model.predict_on_batch(xi)
    #     print(xi["group"].shape, pi.shape)    
    
    predictions = model.predict(data_loader)
    assert len(predictions) == len(test_X_paths), f"Number of predictions does not match number of images passed to model for prediction. {len(predictions)} vs {len(test_X_paths)}"


    # Save results 
    if os.path.isdir(args.trained_weights) and 'epoch' in vars(args):
        save_images_w_epoch(args.savedir, predictions, test_X_paths, epoch=args.epoch)
    else:
        save_images(args.savedir, predictions, test_X_paths)
        