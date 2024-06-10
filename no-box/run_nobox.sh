# test audioseal
python3 nobox_audioseal_audiomarkdata.py --save_pert --common_perturbation time_stretch --gpu 0 
# test timbre
python3 nobox_timbre_audiomarkdata.py --save_pert --common_perturbation time_stretch --gpu 0
# test wavmark
python3 nobox_wavmark_audiomarkdata.py --save_pert --common_perturbation time_stretch --gpu 0 