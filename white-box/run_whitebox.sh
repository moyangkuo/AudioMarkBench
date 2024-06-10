# run whitebox attack for audioseal, with snr 20
python3 whitebox_removal.py --gpu 0 --tau 0.1 --model audioseal --iter 10000 --rescale_snr 20 --dataset librispeech --whitebox_folder wb_audioseal_lib_snr20 

python3 whitebox_forgery.py --gpu 0 --tau 0.1 --model audioseal --iter 10000 --rescale_snr 20 --dataset librispeech --whitebox_folder wb_audioseal_lib_snr20 

