import os 
import glob 
import random
import re
import copy 
import sys 


def get_data(args):

    # Get lists of X and Y  paths
    if args.ismrm:
        X_paths, Y_paths = ismrm_synthetic_data(args)
    #     elif args.transfer_learning:
    #         X_paths, Y_paths = real_data(args)
    elif args.training_data_type == 'real':
        X_paths, Y_paths = real_data(args)
    elif args.training_data_type == 'synthetic':        
        X_paths, Y_paths = synthetic_data(args)
    else:
        X_paths, Y_paths = any_data(args)


    # check data paths 
    check_data_paths(X_paths, Y_paths)

    # cut training data 
    if args.limit_input_to is not None:
        X_paths, Y_paths = cut_training_data(X_paths,Y_paths,args.limit_input_to)       

    # manual check
    if args.userinput:
        print("USERINPUT = True")
        print("You are now inside train_data.py function")
        print("Type exit to proceed")
        from IPython import embed; embed()
        
        
        
    return X_paths, Y_paths

def any_data(args):
    
    
    if not args.x.endswith(".txt"):
        assert args.x.endswith("*"), f"Please make sure input x ends with *. Currently it is: {args.x}"
        assert args.y.endswith("*"), f"Please make sure input y ends with *. Currently it is: {args.y}"

        # get paths 
        X_paths = glob.glob(args.x+'.nii.gz')
        Y_paths = glob.glob(args.y + '.nii.gz')
        
    else: 
        # load from text file 
        assert args.y.endswith(".txt") # both args.x and args.y should be .txt files 
        X_paths = read_txt(args.x)
        Y_paths = read_txt(args.y)
    
    X_paths = sorted(X_paths)
    Y_paths = sorted(Y_paths)
    assert X_paths, f"No .nii.gz found"
    assert Y_paths, f"No .nii.gz found"    
    assert len(X_paths) == len(Y_paths), f"Number of x and y inputs found in supplied directories does not match. Are there more files with .nii.gz suffix in one directory than another?"    
    
    print("Please check if x and y paths match")
    print("i=0")
    print(os.path.basename(X_paths[0]))
    print(os.path.basename(Y_paths[0]))   
    print("i=end")
    print(os.path.basename(X_paths[len(X_paths)-1]))
    print(os.path.basename(Y_paths[len(Y_paths)-1]))  
    
    return X_paths, Y_paths

def get_val_data(args):
    
    assert "*" in args.xval, f"Please make sure input x ends with *. Currently it is: {args.xval}"
    assert "*" in args.yval, f"Please make sure input y ends with *. Currently it is: {args.yval}"
    #assert args.yval.endswith("*") or args.yval.endswith("*.nii.gz"), f"Please make sure input y ends with *. Currently it is: {args.yval}"

    # get paths 
    X_paths = glob.glob(args.xval+'.nii.gz') if args.xval.endswith("*") else glob.glob(args.xval)
    X_paths = sorted(X_paths)
    Y_paths = glob.glob(args.yval + '.nii.gz') if args.yval.endswith("*") else glob.glob(args.yval)
    Y_paths = sorted(Y_paths)
    assert X_paths, f"No .nii.gz found"
    assert Y_paths, f"No .nii.gz found"    
    assert len(X_paths) == len(Y_paths), f"Number of x and y inputs found in supplied directories does not match. Are there more files with .nii.gz suffix in one directory than another?"
    
    print("Please check if x and y paths match")
    print("i=0")
    print(os.path.basename(X_paths[0]))
    print(os.path.basename(Y_paths[0]))   
    print("i=end")
    print(os.path.basename(X_paths[len(X_paths)-1]))
    print(os.path.basename(Y_paths[len(Y_paths)-1]))  
    if args.userinput:
        input("Press any key to proceed")

    
    return X_paths, Y_paths

def get_test_data(args):
    if not args.xtest.endswith(".txt"):
        assert args.xtest.endswith("*"), f"Please make sure input x ends with *. Currently it is: {args.xtest}"
        assert args.ytest.endswith("*"), f"Please make sure input y ends with *. Currently it is: {args.ytest}"

        # get paths 
        X_paths = glob.glob(args.xtest+'.nii.gz')
        Y_paths = glob.glob(args.ytest+'.nii.gz')
        
    else: 
        # load from text file 
        assert args.ytest.endswith(".txt") # both args.x and args.y should be .txt files 
        X_paths = read_txt(args.xtest)
        Y_paths = read_txt(args.ytest)
    
    X_paths = sorted(X_paths)
    Y_paths = sorted(Y_paths)
    assert X_paths, f"No .nii.gz found"
    assert Y_paths, f"No .nii.gz found"    
    assert len(X_paths) == len(Y_paths), f"Number of x and y inputs found in supplied directories does not match. Are there more files with .nii.gz suffix in one directory than another?"    
    
    print("TEST_DATA_ONLY")
    print("Please check if x and y paths match")
    print("i=0")
    print(os.path.basename(X_paths[0]))
    print(os.path.basename(Y_paths[0]))   
    print("i=end")
    print(os.path.basename(X_paths[len(X_paths)-1]))
    print(os.path.basename(Y_paths[len(Y_paths)-1]))  
    
    return X_paths, Y_paths      

def real_data(args):

    def x2y(args,i, suffix):
        return args.y + os.path.basename(i.replace(".nii.gz", suffix))

    # verify that labels source is given
    assert args.labels_source is not None, f"Please specify --labels_source"

    # get suffix of files based on label source algorithm
    if args.labels_source == 'julia':
        suffix = '_julia_mwf_e14_58ms.nii.gz'
    elif args.labels_source == 'julia_norm':
        suffix = '_julia_mwf_e14_58ms_n.nii.gz'
    elif args.labels_source == 'miml':
        suffix = '_pred.nii.gz'

    # get paths 
    X_paths = glob.glob(args.x+'*.nii.gz')
    Y_paths = [x2y(args,i, suffix) for i in X_paths]

    return X_paths, Y_paths

def synthetic_data(args): 

    # get paths 
    X_paths = glob.glob(args.x+'signals_[0-9]*_s[0-9]*.nii.gz')
    X_paths = sorted(X_paths)
    Y_paths = glob.glob(args.y + 'mwf_[0-9]*_s[0-9]*.nii.gz')
    Y_paths = sorted(Y_paths)

    # verify that each pair of X and Y match (the volume and slice numbers must match)
    for i in range(0,len(X_paths)):
        labelpath = os.path.basename(Y_paths[i])
        inputpath = os.path.basename(X_paths[i])
        assert labelpath[4:]==inputpath[8:], f"labelpath:{labelpath}, inputpath:{inputpath}" # check that image names match 
    
    return X_paths, Y_paths

def ismrm_synthetic_data(args):

    """ Legacy: for ismrm conference submission"""

    # get paths 
    X_paths = glob.glob(args.x+'signals_[0-9]*_s[0-9]*.nii.gz')
    X_paths = sorted(X_paths)
    Y_paths = glob.glob(args.y + 'params_[0-9]*_mwf_s[0-9]*.nii.gz')
    Y_paths = sorted(Y_paths)

    # verify that each pair of X and Y match (the volume and slice numbers must match)
    for i in range(0,len(X_paths)):
        assert Y_paths[i][-10:] == X_paths[i][-10:] # check slices match
        labelpath_ = re.sub('s[0-9]*.nii.gz','',Y_paths[i]) 
        inputpath_ = re.sub('s[0-9]*.nii.gz','',X_paths[i])
        assert os.path.basename(labelpath_)[6:8] == os.path.basename(inputpath_)[7:9] # check volumes match

    return X_paths, Y_paths

def split_data(args,X_paths,Y_paths):

    """ Split img paths into a training and a validation set    """

    v = args.valsize

    random.Random(1337).shuffle(X_paths)
    random.Random(1337).shuffle(Y_paths)
    train_X_paths = X_paths[:-v]
    train_Y_paths = Y_paths[:-v]
    val_X_paths = X_paths[-v:]
    val_Y_paths = Y_paths[-v:]

    # check that none of the lists are empty (since some lists can become empty by accident of split data with incorrect input args) 
    assert train_X_paths and train_Y_paths and val_X_paths and val_Y_paths
    assert len(train_X_paths)>len(val_X_paths), f"train list should be larger than val list"

    return train_X_paths, train_Y_paths, val_X_paths, val_Y_paths

def cut_training_data(X_paths,Y_paths,limit_input_to):

    assert limit_input_to>0 and isinstance(limit_input_to,int) and limit_input_to<len(X_paths), 'must specifiy a positive integer that is smaller than training size'
    random.Random(1337).shuffle(X_paths)
    random.Random(1337).shuffle(Y_paths)
    X_paths = X_paths[:limit_input_to]
    Y_paths = Y_paths[:limit_input_to]

    return X_paths, Y_paths


def check_data_paths(X_paths, Y_paths, args=None):        
    # check lists are not empty 
    assert X_paths, f"No X data found: {X_paths}"

    # check each individual file exists
    assert all([os.path.exists(i) for i in X_paths])


    # repeat the above for Y, but check if y is provided
    if args is not None:
        if args.selfsupervised:
            pass # no need to check Y as they are a copy of X
        else:
            sys.exit('not implemented')
    else:
        assert Y_paths, f"No Y data found: {Y_paths}"
        assert all([os.path.exists(i) for i in Y_paths])


def read_txt(f):
    assert os.path.exists(f)
    with open(f, 'r') as file:
        lines = file.readlines()
    lines = [l.replace('\n', '') for l in lines]
    assert all([os.path.exists(l) for l in lines]), "Some files in the .txt do not exist"    
    return lines