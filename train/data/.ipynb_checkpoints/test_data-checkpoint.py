import glob 

def get_data(args): 
    # check if input exists 
    if args.test_dir is not None: 
        inputpaths = sorted(glob.glob(args.test_dir+"*.nii.gz"))
    elif args.test_regexp is not None: 
        #inputpaths = args.test_regexp
        inputpaths = sorted(glob.glob(args.test_regexp))
    elif args.test_file is not None: 
        inputpaths = args.test_file

    # wrap in a list if only a single file 
    if isinstance(inputpaths, str): 
        inputpaths = [inputpaths] 

    return inputpaths

