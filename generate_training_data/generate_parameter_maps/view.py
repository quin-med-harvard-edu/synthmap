""" Helper tools that allow one to view the results of the generated parameter maps immediately after their generation. 

Writes instructions for viewing the generated files with itksnap with one click to a separate .txt file"""


import os 
import svtools as sv
import glob 

def view_itksnap(datadir,remote=False, multiseg=False):  #renamed from 'analyzed_results()'

    """View all of the generated parameters maps in itksnap viewer."""

    # get original segmentation file 
    if not os.path.exists(datadir+"example.nii.gz"): # if segmentation is not in root folder 
        seg = datadir+'../example.nii.gz' if os.path.exists(datadir+'../example.nii.gz') else datadir+'../../example.nii.gz'
    else:
        seg = datadir+"example.nii.gz"

    # get generated files 
    files = glob.glob(datadir+'/*.nii.gz') # includes labels and images 
    ims = [f for f in files if 'labels' not in os.path.basename(f)] # fetch only images, not labels     
    ims = [f for f in ims if '_split_' not in os.path.basename(f)] # remove fslsplits...    
    ims = [f for f in ims if '_params_' not in os.path.basename(f)] # remove fslsplits...    
    ims = [f for f in ims if '_echoes_' not in os.path.basename(f)] # remove fslsplits...    
    ims = [f for f in ims if '_pT2_' not in os.path.basename(f)] # remove fslsplits...        
    ims = [f for f in ims if 'example.nii.gz' not in os.path.basename(f)] # remove example files
    ims = sorted(ims)
    labels = [f for f in files if 'labels' in os.path.basename(f)] # fetch only labels  
    

    #view images in itksnap 
    if multiseg:
        cmd = ['itksnap', '-g', ims.pop(0), '-o']
    else: 
        cmd = ['itksnap', '-g', seg, '-o']
    cmd.extend(ims) # extend with added values
    print("View in itksnap")
    print(' '.join(cmd))    
    if not remote:
        sv.execute(cmd)
    
    return cmd


    # commented out temporarily 
    # #view labels in itksnap 
    # cmd = ['itksnap', '-g', seg, '-o']
    # cmd.extend(labels) # extend with added values
    # if not remote:
    #     sv.execute(cmd)

def view_itksnap_split(datadir,remote=True):
    """ Generated parameter maps have 5 channels - encoding the mean T2 of myelin, mean T2 of IES, mean T2 of CSF, MWF and IEWF maps respectively. 
    
    This short function shows these parameter maps in itksnap AFTER they have been split apart into separate 3D images"""
 
    # get original segmentation file 
    if not os.path.exists(datadir+"example.nii.gz"): # if segmentation is not in root folder 
        seg = datadir+'../example.nii.gz' if os.path.exists(datadir+'../example.nii.gz') else datadir+'../../example.nii.gz'
    else:
        seg = datadir+"example.nii.gz"

    # get generated files 
    ims_split = glob.glob(datadir+'/*split*.nii.gz') # includes labels and images   
    ims_split = sorted(ims_split) # sort by order 

    #view images in itksnap 
    cmd = ['itksnap', '-g', seg, '-o']
    cmd.extend(ims_split) # extend with added values
    print("View splitfiles in itksnap")
    print(' '.join(cmd))
    if not remote:
        sv.execute(cmd)
    
    return cmd


def view_rview(datadir,remote=False):
    """View the first two generated parameters map volumes in RVIEW (IRTK software viewier) to compare deformations."""

    # get generated files 
    files = glob.glob(datadir+'/*.nii.gz') # includes labels and images 
    ims = [f for f in files if 'labels' not in os.path.basename(f)] # fetch only images, not labels     
    ims = [f for f in ims if '_split_' not in os.path.basename(f)] # fetch only images, not labels     
    ims = [f for f in ims if '_params_' not in os.path.basename(f)] # remove fslsplits...    
    ims = [f for f in ims if '_echoes_' not in os.path.basename(f)] # remove fslsplits...    
    ims = [f for f in ims if '_pT2_' not in os.path.basename(f)] # remove fslsplits...        
    ims = [f for f in ims if 'example.nii.gz' not in os.path.basename(f)] # fetch only images, not labels     
    
    # take only the first two images 
    ims = ims[:2]
    #view images in itksnap 
    cmd = ['rview']
    cmd.extend(ims) # extend with added values
    print("View in rview")
    print(' '.join(cmd))
    if not remote:
        sv.execute(cmd)

    return cmd


def fslsplit(datadir,prefix="sim_",suffix=".nii.gz"):
    """ Generated parameter maps have 5 channels - encoding the mean T2 of myelin, mean T2 of IES, mean T2 of CSF, MWF and IEWF maps respectively. 

    This short function splits these parameter maps into separate 3D images"""

    
    suffix = "*"+suffix
    avoid = ["split"]
    
    os.chdir(datadir) # change to current dir (required for fslsplit)
    
    files = glob.glob(datadir+prefix+"[0-9]*"+".nii.gz")
    files = [f for f in files if avoid[0] not in os.path.basename(f)]
    # pick only the first 3 files (don't need to get more at this stage)
    files = files[:3]

    for f in files:
        # fetch each image from datadir and split 
        cmd = ["fslsplit"]
        cmd.append(f)
        _, f_name = os.path.split(f)
        basename = f_name[:-7] + "_split_"
        cmd.append(basename)
        cmd.append("-t")
        print(' '.join(cmd))
        sv.execute(cmd)