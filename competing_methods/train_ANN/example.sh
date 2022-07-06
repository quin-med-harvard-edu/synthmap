# train 
wdir=<pathtodir>
python train_miml.py --name test1 --epochs 600 --batchsize 2000 --savedir $wdir --dataset te9


# test
testdir=<pathtofiledir>
savedir=<pathtodir>
wdir=trained_weights/te9/weights.68-102.93.hdf5 # provided in repo 
cutoff_in_ms=58 # used in our test
python test_miml.py --weights $wdir --test_dir $test_dir --savedir $savedir --cutoff_echo_in_ms $cutoff_in_ms

