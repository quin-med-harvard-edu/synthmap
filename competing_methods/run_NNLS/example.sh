# download DECAES from here:
git clone https://github.com/jondeuce/DECAES.jl


# run 
input=<pathtodir>
out=<pathtodir>
mkdir $out
python run_julia.py --input $input --out $out 
python run_julia.py --input $input --out $out --recalculate_cutoff --cut_off_echo 12

