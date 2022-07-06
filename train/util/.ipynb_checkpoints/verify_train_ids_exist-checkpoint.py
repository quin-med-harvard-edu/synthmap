import os 
import sys 
import pickle

import numpy as np

import svtools as sv

                  
def getfilename(args,idx):                
    x=args.xdir + "/" + args.xprefix + str(idx[0]) + "_s" + str(idx[1]) + args.xsuffix
    y=args.ydir + "/" + args.yprefix + str(idx[0]) + "_s" + str(idx[1]) + args.ysuffix
                   
    return x,y

if __name__ == '__main__':

    # import dataids function 
    d='/home/ch215616/w/code/mwf/synth_unet/train_unet/'
    sys.path.append('d')
    from data.train_data import dataids
    
    # read input args
    args = sv.init_argparse()
    args.xdir = sys.argv[1]
    args.ydir = sys.argv[2]
    args.xprefix = sys.argv[3]
    args.yprefix = sys.argv[4]
    args.xsuffix = sys.argv[5]
    args.ysuffix = sys.argv[6]
    args.volumes = [int(sys.argv[7]), int(sys.argv[8])]
    args.slices = [int(sys.argv[9]), int(sys.argv[10])]
    args.excludevolumes = sys.argv[11]
    args.excludevolumesrange = [sys.argv[12], sys.argv[13]]
    args.valsize = int(sys.argv[14])
    args.seed = int(sys.argv[15])
    args.limit_input_to = sys.argv[16]
    args.output = sys.argv[17]

    
    if args.excludevolumes.lower() == 'none':
        args.excludevolumes = None 
    else:
        args.excludevolumes = int(args.excludevolumes)
    if args.excludevolumesrange[0].lower() == 'none':
        args.excludevolumesrange = None 
    else: 
        args.excludevolumes[0] = int(args.excludevolumes[0])
        args.excludevolumes[1] = int(args.excludevolumes[1])
    if args.limit_input_to.lower() == 'none':
        args.limit_input_to = None 
    else:
        args.limit_input_to = int(args.limit_input_to)    
    
    #assert args.output.endswith(".txt")
    #f = open(args.output, 'w') 
    
    ids = dataids(args)
    
    
    
    # split data into val and train 
    train_ids = ids[:-args.valsize, :]
    val_ids = ids[-args.valsize:, :]

    # limit to 
    if args.limit_input_to is not None:
        train_ids = train_ids[:args.limit_input_to, :]        
    
    
    #ids = ids[:10000,:]
    L = ids.shape[0] 
    not_existing_ids = np.zeros((ids.shape[0],1))
    #from IPython import embed; embed()    
    for i, _ in enumerate(ids):
        if i % 1000 == 0:
            print(f"{i}/{L}")
        x,y = getfilename(args,ids[i])
        if not os.path.exists(x) or not os.path.exists(y): 
            print(f"No exist: {i}")
            not_existing_ids[i,0] = 1
            # write to file 
            #f.write(str(i[0])+'\n')
            #f.write('\t\t\t\t\t\t\t'+str(i[1])+'\n')
    #f.close()
    from IPython import embed; embed()  
    
    # check not existing volumes
    ids[107669,:]
    ids[107699,:]
    
    ids[354060,:]
    ids[354119,:]

    # check their paths manually 
    x,y = getfilename(args,ids[107669])
    x
    y
    os.path.exists(x)
    os.path.exists(y)
    x,y = getfilename(args,ids[107699])
    x
    y
    os.path.exists(x)
    os.path.exists(y)
    
    x,y = getfilename(args,ids[354060])
    x
    y
    os.path.exists(x)
    os.path.exists(y)
    x,y = getfilename(args,ids[354119])
    x
    y
    os.path.exists(x)
    os.path.exists(y)    
    
    
    
    # save to pickle 
    
    vols_not_existing = np.where(not_existing_ids)[0]
    vols_existing = np.where(not_existing_ids,not_existing_ids==0)[0]
    ids_final = ids[~vols_not_existing,:]
    
    #ids_volumes = ids[:,0]
    #ids_volumes = np.unique(ids_volumes)

    
    
    mydict = {'ids':ids, 'not_existing_ids':not_existing_ids, 'vols_not_existing':vols_not_existing}
    with open(args.output.replace(".txt", ".pkl"), 'w') as f: 
         pickle.dump(mydict, f)


