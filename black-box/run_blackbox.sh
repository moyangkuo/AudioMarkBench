# run HSJA signal, example for audioseal
python3 HSJA_signal_audiomarkdata.py --gpu 0 --testset_size 200  --query_budget 10000  --tau 0.15 --norm linf --model audioseal --blackbox_folder audioseal_10k 

# run HSJA spectrogram, exmaple for audioseal 
python3 HSJA_spectrogram_audiomarkdata.py --gpu 0 --testset_size 200  --query_budget 10000  --tau 0.15 --norm linf --model audioseal --blackbox_folder audioseal_10k --attack_type both

# run square attack, example for audioseal
python3 square_audiomarkdata.py --gpu 0 --testset_size 200  --query_budget 10000  --tau 0.15 --model audioseal --blackbox_folder audioseal_10k --attack_type both --eps 0.05

