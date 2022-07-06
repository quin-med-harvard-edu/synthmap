import os 
import sys 
import argparse

import numpy as np
import tensorflow as tf
import nibabel as nb


from train_miml import set_architecture, TE_vector
from get_test_data import load_test_data

     
def load_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default = None, help='path to saved training weights')
    parser.add_argument('--test_dir', type=str,required=True, help='name of the test dir (clinical, volunteer,acc) or path to custom dir or file')    
    parser.add_argument('--savedir',type=str,required=True, help='name of directory to save the inferrence to')
    parser.add_argument('--type',type=str,choices=['default', 'synthetic'],default='default', help='name of directory to save the inferrence to')
    parser.add_argument('--cutoff_echo_in_ms',type=float,default=80, help='TE value which defines the Myelin Water Fraction ratio calcuation')
    parser.add_argument('--threshold', type=float, default=0, help='define masking threshold')

            
    args = parser.parse_args()
            
    
    return args 

def check_args(args):
    
    # load default args if certain parameters aren't specified 
    if args.weights is None:
        #args.weights = '/home/ch215616/abd/mwf_data/MIML/trained_weights/default_e300/weights.77-104.92.hdf5'
        args.weights = '~/w/mwf_data/MIML/trained_weights/te9/weights.68-102.93.hdf5'
        args.weights = os.path.expanduser(args.weights)

    
    assert os.path.exists(args.weights)
    assert os.path.exists(args.test_dir)
    args.test_dir = args.test_dir + '/' if not args.test_dir.endswith('/') else args.test_dir
    args.savedir = args.savedir + '/' if not args.savedir.endswith('/') else args.savedir
    
    return args 
       


def find_nearest(array, value):
    
    """Get an index of the value in the array that is closest matching to the value specified as a target"""
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def predict_T2_distribution(im,model,TEs): 
    
    """Predict T2 distribution of the entire image
    
    This function vectorizes an image and processes it through a pre-trained MIML network 
    """

    assert im.ndim == 4, "the input image should be (st,sx,sy,sz). If 2D slice, add dummy dimension."
    assert im.shape[0] == 32, "the first dimension should be 32, as per number of echoes"
    
    nT2 = TEs.shape[0]
    
    # flatten image 
    st,sx,sy,sz = im.shape 
    im_v = np.reshape(im,(32,sx*sy*sz))  
    nz = np.nonzero(im_v[0,:]) # get nonzero IDs from S0 image (so that we don't try to estimate T2 dist over zeros) 
    im_v_nz = im_v[:,nz[0]]     # select only non zero voxels 
    im_v_nz = np.transpose(im_v_nz,(1,0))     # swap dimensions. Required for predict 

    assert im_v_nz.shape[-1] == 32, 'data must be a (x,32) numpy.array'
    assert im_v_nz.ndim == 2, 'data must be a (x,32) numpy.array'   
    
    predictions = model.predict(im_v_nz)    
    predictions = np.transpose(predictions,(1,0))
    
    t2dist = np.zeros((nT2,sx,sy,sz))  # create image to hold distributions 
    t2dist_v = np.reshape(t2dist,(nT2,sx*sy*sz)) # reshape accordingly 
    t2dist_v[:,nz[0]] = predictions # fill in with predictions 
    t2dist = np.reshape(t2dist_v, (nT2,sx,sy,sz))
    t2dist = np.transpose(t2dist, (1,2,3,0))
    
    return t2dist

def get_mwf_map(dist_predicted,cutoff_id,t2times):

    """Get Myelin Water Fraction map from a predicted T2 distribution and a certain value of the cutoff T2 (in ms) 
    
    The implementation follows the work of Prasloski et al for multi-component T2 fitting"""
    
    assert 4 >= dist_predicted.ndim >= 2     
    assert dist_predicted.shape[-1] == t2times.shape[-1], "the last dim of the predicted dist has to match default length of t2dist (either 60 or 40)"
    #from IPython import embed; embed()
    #t2times[:cutoff_id]
    
    # calculate mwf 
    if dist_predicted.ndim == 4: 
        mw1 = np.sum(dist_predicted[:,:,:,0:cutoff_id+1],axis=-1)
        tot1= np.sum(dist_predicted[:,:,:,:],axis=-1)
    elif dist_predicted.ndim == 3: 
        mw1 = np.sum(dist_predicted[:,:,0:cutoff_id+1],axis=-1)
        tot1= np.sum(dist_predicted[:,:,:],axis=-1)
    elif dist_predicted.ndim == 2: 
        mw1 = np.sum(dist_predicted[:,0:cutoff_id+1],axis=-1)
        tot1= np.sum(dist_predicted[:,:],axis=-1)
    mwf=np.divide(mw1,tot1, out=np.zeros_like(mw1), where=tot1!=0)  #c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)    
    
    assert mwf.shape == dist_predicted.shape[0:-1], "shape of mwf has to match the shape of dist_predicted (minus the last dimension)"
        
    return mwf 

def save_image(im,path, savedir):
    
    """Save image to filepath"""
    
    # if savedir is specified - save to it, else save to the original file's location
    dirname = os.path.dirname(path) + "/" if savedir is None else savedir + "/"
    
    if path.endswith(".nii.gz"): 
        savename = dirname + os.path.basename(path).replace(".nii.gz", "_pred.nii.gz")
        refo = nb.load(path)
        imo = nb.Nifti1Image(im,affine=refo.affine,header=refo.header)
    elif path.endswith(".nrrd"):
        savename = dirname + os.path.basename(path).replace(".nrrd", "_pred.nii.gz")
        imo = nb.Nifti1Image(im,affine=np.eye(4))
    else: 
        sys.exit(f"saving from this extension is not implemented: {path}")

    nb.save(imo,savename)        
    print(f"Saved: {savename}")
if __name__ == '__main__':

    # init vars
    args = load_args()
    args = check_args(args)
    
    # basic checks 
    assert '2.2' in tf.__version__, "Please change to tf 2.0+ conda environement. `conda activate tch2_yml` "
    assert os.path.exists(args.weights), f"Path does not exist: {args.weights}"
     
    # build model 
    model = set_architecture(type='default')
    
    # load weights 
    model.load_weights(args.weights)
        
    # get echo sampling vector (TE times) in lognorm form   
    TEs = TE_vector(nT2=60, nStart=10.0,nEnd=2000.0)
   
    # find nearest index of the cuttoff echo - used for calculation of myelin water fraction from a given vector of sampled TE values 
    cutoff_te_id, cutoff_te_value = find_nearest(TEs,args.cutoff_echo_in_ms)
    print(f"Nearest matching echo in ms is: {cutoff_te_value}")
    
    # generator to yield test images 
    generator = load_test_data(args.type)        
    
    for i,(im,path) in enumerate(generator(args.test_dir,threshold=args.threshold)):

        # print progress 
        print(i)
        
        # predict distribution of T2s using the model
        t2_distribution = predict_T2_distribution(im,model,TEs)

        # build MWF map
        mwf = get_mwf_map(t2_distribution,cutoff_te_id,TEs)

        # save image 
        save_image(mwf,path,args.savedir)

        


