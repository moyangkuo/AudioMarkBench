import os
import sys
import argparse
import numpy as np
import torch
import torchaudio 
from tqdm import tqdm
from torch.nn import functional as F

import librosa
import julius
import typing as tp

import tempfile
import uuid
import random
from numpy.typing import NDArray
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_float_samples_to_int16, get_max_abs_amplitude,
)
import warnings
warnings.filterwarnings(
    action='ignore', 
    message='.*MessageFactory class is deprecated.*'
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Audio Watermarking with WavMark")
    parser.add_argument("--testset_size", type=int, default=20000, help="Number of samples from the test set to process")
    parser.add_argument("--encode", action="store_true", help="Run the encoding process before decoding")
    parser.add_argument("--common_perturbation", type=str, default="") 
    '''
    options=["time_stretch", "gaussian_noise", "background_noise", "quantization",
            "soundstream", "opus", "encodec", "lowpass", "highpass",
            "echo", "mp3", "smooth"]
    '''
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")
    parser.add_argument("--save_pert", action="store_true", help="If set, saves the perturbed waveform")
    parser.add_argument("--max_length", type=int, default=5*16000, help="Maximum length of audio samples to process (in time samples)")
    parser.add_argument("--batch_size", type=int, default=100, help="Maximum length of audio samples to process (in time samples)")
    return parser.parse_args()

def encode_audio_files(model,data_dir, output_dir, max_length):
    detector = model[1] # detector
    model = model[0] # generator
    
    total_ba = []
    total_counts = 0
    
    file_list = [file for file in os.listdir(data_dir) if file.endswith('.mp3')]
    for file in tqdm(file_list, desc="Encoding Watermarks"):
        file_path = os.path.join(data_dir, file)
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        '''resample to 16k'''
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            sample_rate = 16000

        # Truncate or pad the waveform to max_length
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        elif waveform.shape[1] < max_length:
            padding = max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        signal = waveform.unsqueeze(0).to(device=next(model.parameters()).device)

        msg = torch.randint(0, 2, (signal.shape[0], 30), device=signal.device,dtype=torch.float32)
        '''rescale the message to -1 and 1 according to the watermark function'''
        msg = msg.unsqueeze(0)  
        msg_rescaled = msg * 2 - 1
        '''watermark function call'''
        watermarked_signal = model.test_forward(signal, msg_rescaled)[0]

        # Compute SNR
        signal_power = torch.mean(signal ** 2)
        noise_power = torch.mean((watermarked_signal - signal) ** 2)
        snr = 10 * torch.log10(signal_power / noise_power)
        
        msg_str =  ''.join([''.join(map(str, map(int, msg.squeeze().tolist())))])
        file_name = f"{file[:-4]}_{msg_str}_{snr:.3f}.wav"
        watermarked_signal = watermarked_signal.cpu().squeeze(0)
        save_path = os.path.join(output_dir, 'watermarked')
        os.makedirs(save_path, exist_ok=True)
        torchaudio.save(os.path.join(save_path, file_name), watermarked_signal, sample_rate)

        bit_acc = get_bitacc(detector, signal, msg)
        total_ba.append(bit_acc.item()) 
        total_counts += 1

    return

def detection_BA(detect_BA, tau=0.8):
    detection_results = [1 if BA > tau else 0 for BA in detect_BA]
    fraction_of_ones = sum(detection_results) / len(detection_results)
    return fraction_of_ones

def get_bitacc(model, signal, message):
    with torch.no_grad():
        msg_decoded = model.test_forward(signal).squeeze()
        message = message * 2 - 1
        msg_decoded = msg_decoded.to(message.device)
    return (msg_decoded >= 0).eq(message >= 0).sum(dim=-1).float()/ message.shape[-1]

def extract_id(filename):
    file_name = filename.split('/')[-1]
    parts = file_name.split('_')
    id_part = '_'.join(parts[:4])
    return id_part

def decode_audio_files(model, output_dir, batch_size):
    total_ba = []
    total_counts = 0
    file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
    progress_bar = tqdm(enumerate(file_list), desc="Decoding Watermarks")

    batch_watermarked_signals = []
    original_msgs = []
    path_list = []
    output_txt_dir = f'txt_audiomarkdata_Timbre/test'
    os.makedirs(output_txt_dir, exist_ok=True)
    output_file = os.path.join(output_txt_dir, "decoding_results_AudioSeal.txt")


    with open(output_file, 'w') as txt_file:
        txt_file.write("Path, Timbre BA\n")
        for id, file in progress_bar:
            try:
                index_str, payload_str = file.rstrip('.wav').split('_')[-3:-1]
                index = int(index_str)
            except ValueError:
                continue

            original_msg_np = np.array(list(map(int, payload_str)))
            original_msg = torch.tensor(original_msg_np, dtype=torch.int32).to(device=next(model.parameters()).device)
            original_msgs.append(original_msg)

            path = os.path.join(output_dir, file)
            path_list.append(extract_id(path))
            watermarked_signal, sr = torchaudio.load(path)
            watermarked_signal = watermarked_signal.to(device=next(model.parameters()).device).unsqueeze(0)
            batch_watermarked_signals.append(watermarked_signal)

            # Extract original payload from filename for BER calculation
            if len(batch_watermarked_signals) == batch_size or id == len(file_list) - 1:
                batch_watermarked_signals = torch.cat(batch_watermarked_signals, dim=0)
                original_msgs = torch.stack(original_msgs)
                bw_acc = get_bitacc(model, batch_watermarked_signals, original_msgs)
                for i in range(len(batch_watermarked_signals)):
                    total_ba.append(bw_acc[i].item())
                    total_counts += 1
                    txt_file.write(f"{path_list[i]}, {bw_acc[i].item()}\n")
                # Reset the batch
                batch_watermarked_signals = []
                original_msgs = []
                path_list = []

                current_average_BA = (sum(total_ba) / total_counts) * 100
                progress_bar.set_description(f"Decoding Watermarks - Avg BA: {current_average_BA:.2f}%")

                # Reset the batch
                batch_watermarked_signals = []
                original_msgs = []
                path_list = []

        for tau_ba in [0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375]:
            detection_fraction = detection_BA(total_ba, tau_ba)
            fnr = 1-detection_fraction
            print(f'FNR (Tau_ba={tau_ba}): {fnr:.3f}\n')

    return



def decode_audio_files_perturb(model, output_dir, common_perturbation, args):
    if common_perturbation == '':
        total_ba = []
        total_visqol_scores = []
        total_counts = 0
        output_dir_pert = f'log_timbre_audiomarkdata_max_5s/TPR'
        os.makedirs(output_dir_pert, exist_ok=True)
        file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
        progress_bar = tqdm(enumerate(file_list), desc=f"Applying {common_perturbation}")
        
        batch_watermarked_signals = []
        batch_visqol_scores = []
        batch_SNRs = []
        original_msgs = []
        path_list = []
        visqol = api_visqol()
        output_txt_dir = f'txt_audiomarkdata_Timbre/test'
        os.makedirs(output_txt_dir, exist_ok=True)
        output_file = os.path.join(output_txt_dir, f"watermarked_acc.txt")

        with open(output_file, 'w') as txt_file:
            txt_file.write("Path, Timbre BA, SNR, Visqol score\n")

            for id, file in progress_bar:
                try:
                    index_str, payload_str = file.rstrip('.wav').split('_')[-3:-1]
                    index = int(index_str)
                except ValueError:
                    continue

                original_msg_np = np.array(list(map(int, payload_str)))
                original_msg = torch.tensor(original_msg_np, dtype=torch.int32).to(device=next(model.parameters()).device)
                original_msgs.append(original_msg)

                path = os.path.join(output_dir, file)
                path_list.append(extract_id(path))
                waveform, sample_rate = torchaudio.load(path)

                waveform_pert = waveform.unsqueeze(0).to(device=next(model.parameters()).device)
                batch_watermarked_signals.append(waveform_pert)
                
                if args.save_pert:
                    torchaudio.save(os.path.join(output_dir_pert, file), waveform_pert.squeeze(0).detach().cpu(), sample_rate)

                # Extract original payload from filename for BER calculation
                if len(batch_watermarked_signals) == args.batch_size or id == len(file_list) - 1:
                    batch_watermarked_signals = torch.cat(batch_watermarked_signals, dim=0)
                    original_msgs = torch.stack(original_msgs)
                    bw_acc = get_bitacc(model, batch_watermarked_signals, original_msgs)
                    for i in range(len(batch_watermarked_signals)):
                        total_ba.append(bw_acc[i].item())
                        total_visqol_scores.append(batch_visqol_scores[i])
                        total_counts += 1
                        txt_file.write(f"{path_list[i]}, {bw_acc[i].item()}, {batch_SNRs[i]:.2f}, {batch_visqol_scores[i]}\n")
                    # Reset the batch
                    batch_watermarked_signals = []
                    batch_visqol_scores = []
                    batch_SNRs = []
                    original_msgs = []
                    path_list = []

                    current_average_BA = (sum(total_ba) / total_counts) * 100
                    current_average_visqol = sum(total_visqol_scores) / total_counts
                    progress_bar.set_description(f"Time Stretch w speed {speed_factor} Avg BA: {current_average_BA:.2f}% - Avg Visqol: {current_average_visqol:.2f}")

            tau_ba = 0.83
            detection_fraction = detection_BA(total_ba, tau_ba)
            print(f'TPR for Timbre (Tau_ber={tau_ba}): {detection_fraction:.3f}\n')

    if common_perturbation == "time_stretch":
        speed_factor_list = [0.7, 0.9, 1.1, 1.3, 1.5]
        for speed_factor in speed_factor_list:
            total_ba = []
            total_visqol_scores = []
            total_counts = 0
            output_dir_pert = f'log_timbre_audiomarkdata_max_5s/common_pert_time_stretch_speed_{speed_factor}'
            os.makedirs(output_dir_pert, exist_ok=True)
            file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            progress_bar = tqdm(enumerate(file_list), desc=f"Applying {common_perturbation}")
            
            batch_watermarked_signals = []
            batch_visqol_scores = []
            batch_SNRs = []
            original_msgs = []
            path_list = []
            visqol = api_visqol()
            output_txt_dir = f'txt_audiomarkdata_Timbre/no_box_attack'
            os.makedirs(output_txt_dir, exist_ok=True)
            output_file = os.path.join(output_txt_dir, f"time_stretch_{speed_factor}.txt")

            with open(output_file, 'w') as txt_file:
                txt_file.write("Path, Timbre BA, SNR, Visqol score\n")

                for id, file in progress_bar:
                    try:
                        index_str, payload_str = file.rstrip('.wav').split('_')[-3:-1]
                        index = int(index_str)
                    except ValueError:
                        continue

                    original_msg_np = np.array(list(map(int, payload_str)))
                    original_msg = torch.tensor(original_msg_np, dtype=torch.int32).to(device=next(model.parameters()).device)
                    original_msgs.append(original_msg)

                    path = os.path.join(output_dir, file)
                    path_list.append(extract_id(path))
                    waveform, sample_rate = torchaudio.load(path)
                    waveform_pert, new_sample_rate = pert_time_stretch(waveform, sample_rate, speed_factor)
                    snr = compute_snr(waveform.squeeze().unsqueeze(0),waveform_pert)
                    batch_SNRs.append(snr)
                    # Compute Visqol score
                    score = visqol.Measure(np.array(waveform.squeeze(), dtype=np.float64), np.array(waveform_pert.squeeze(), dtype=np.float64))
                    batch_visqol_scores.append(score.moslqo)

                    waveform_pert = waveform_pert.unsqueeze(0).to(device=next(model.parameters()).device)
                    batch_watermarked_signals.append(waveform_pert)
                    
                    if args.save_pert:
                        torchaudio.save(os.path.join(output_dir_pert, file), waveform_pert.squeeze(0).detach().cpu(), new_sample_rate)


                    # Extract original payload from filename for BER calculation
                    if len(batch_watermarked_signals) == args.batch_size or id == len(file_list) - 1:
                        batch_watermarked_signals = torch.cat(batch_watermarked_signals, dim=0)
                        original_msgs = torch.stack(original_msgs)
                        bw_acc = get_bitacc(model, batch_watermarked_signals, original_msgs)
                        for i in range(len(batch_watermarked_signals)):
                            total_ba.append(bw_acc[i].item())
                            total_visqol_scores.append(batch_visqol_scores[i])
                            total_counts += 1
                            txt_file.write(f"{path_list[i]}, {bw_acc[i].item()}, {batch_SNRs[i]:.2f}, {batch_visqol_scores[i]}\n")
                        # Reset the batch
                        batch_watermarked_signals = []
                        batch_visqol_scores = []
                        batch_SNRs = []
                        original_msgs = []
                        path_list = []

                        current_average_BA = (sum(total_ba) / total_counts) * 100
                        current_average_visqol = sum(total_visqol_scores) / total_counts
                        progress_bar.set_description(f"Time Stretch w speed {speed_factor} Avg BA: {current_average_BA:.2f}% - Avg Visqol: {current_average_visqol:.2f}")

                tau_ba = 0.83
                detection_fraction = detection_BA(total_ba, tau_ba)
                print(f'TPR for Timbre (Tau_ber={tau_ba}): {detection_fraction:.3f}\n')

    elif common_perturbation in ["gaussian_noise", "background_noise"]:
        snr_values = [40, 30, 20, 10, 5]  # Example SNR values in dB
        for snr in snr_values:
            total_ba = []
            total_visqol_scores = []
            total_counts = 0
            output_dir_pert = f'log_timbre_audiomarkdata_max_5s/common_pert_{common_perturbation}_snr_{snr}'
            os.makedirs(output_dir_pert, exist_ok=True)
            file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            progress_bar = tqdm(enumerate(file_list), desc=f"Applying {common_perturbation}")
            
            batch_watermarked_signals = []
            batch_visqol_scores = []
            batch_SNRs = []
            original_msgs = []
            path_list = []
            visqol = api_visqol()
            output_txt_dir = f'txt_audiomarkdata_Timbre/no_box_attack'
            os.makedirs(output_txt_dir, exist_ok=True)
            if common_perturbation == "gaussian_noise":
                output_file = os.path.join(output_txt_dir, f"gaussian_noise_snr_{snr}.txt")
            elif common_perturbation == "background_noise":
                output_file = os.path.join(output_txt_dir, f"background_noise_snr_{snr}.txt")
            with open(output_file, 'w') as txt_file:
                txt_file.write("Path, Timbre BA, SNR, Visqol score\n")
                for id, file in progress_bar:
                    try:
                        index_str, payload_str = file.rstrip('.wav').split('_')[-3:-1]
                        index = int(index_str)
                    except ValueError:
                        continue

                    original_msg_np = np.array(list(map(int, payload_str)))
                    original_msg = torch.tensor(original_msg_np, dtype=torch.int32).to(device=next(model.parameters()).device)
                    original_msgs.append(original_msg)

                    path = os.path.join(output_dir, file)
                    path_list.append(extract_id(path))
                    waveform, sample_rate = torchaudio.load(path)
                    if common_perturbation == "gaussian_noise":
                        waveform_pert = pert_Gaussian_noise(waveform, snr)
                    elif common_perturbation == "background_noise":
                        waveform_pert = pert_background_noise(waveform, snr)
                    snr = compute_snr(waveform.squeeze().unsqueeze(0),waveform_pert)
                    batch_SNRs.append(snr)
                    # Compute Visqol score
                    score = visqol.Measure(np.array(waveform.squeeze(), dtype=np.float64), np.array(waveform_pert.squeeze(), dtype=np.float64))
                    batch_visqol_scores.append(score.moslqo)
                    
                    waveform_pert = waveform_pert.unsqueeze(0).to(device=next(model.parameters()).device)
                    batch_watermarked_signals.append(waveform_pert)
                    
                    if args.save_pert:
                        torchaudio.save(os.path.join(output_dir_pert, file), waveform_pert.squeeze(0).detach().cpu(), sample_rate)

                    # Extract original payload from filename for BER calculation
                    if len(batch_watermarked_signals) == args.batch_size or id == len(file_list) - 1:
                        batch_watermarked_signals = torch.cat(batch_watermarked_signals, dim=0)
                        original_msgs = torch.stack(original_msgs)

                        bw_acc = get_bitacc(model, batch_watermarked_signals, original_msgs)

                        for i in range(len(batch_watermarked_signals)):
                            total_ba.append(bw_acc[i].item())
                            total_visqol_scores.append(batch_visqol_scores[i])
                            total_counts += 1
                            txt_file.write(f"{path_list[i]}, {bw_acc[i].item()}, {batch_SNRs[i]:.2f}, {batch_visqol_scores[i]}\n")

                        # Reset the batch
                        batch_watermarked_signals = []
                        batch_visqol_scores = []
                        batch_SNRs = []
                        original_msgs = []
                        path_list = []

                        current_average_BA = (sum(total_ba) / total_counts) * 100
                        current_average_visqol = sum(total_visqol_scores) / total_counts
                        progress_bar.set_description(f"{common_perturbation} w SNR {snr} - Avg BA: {current_average_BA:.2f}% - Avg Visqol: {current_average_visqol:.2f}")

                tau_ba = 0.83
                detection_fraction = detection_BA(total_ba, tau_ba)
                print(f'TPR for Timbre (Tau_ber={tau_ba}): {detection_fraction:.3f}\n')

    elif common_perturbation in ["soundstream"]:
        # n_q_values =  [4, 6, 8, 12, 16]  
        n_q_values =  [16]  

        for nq in n_q_values:
            total_ba = []
            total_probs = []
            total_visqol_scores = []
            total_counts = 0
            output_dir_pert = f'log_timbre_audiomarkdata_max_5s/common_pert_{common_perturbation}_nq_{nq}'
            os.makedirs(output_dir_pert, exist_ok=True)
            file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            progress_bar = tqdm(enumerate(file_list), desc=f"Applying {common_perturbation}")
            
            batch_watermarked_signals = []
            batch_visqol_scores = []
            batch_SNRs = []
            original_msgs = []
            path_list = []
            visqol = api_visqol()
            output_txt_dir = f'txt_audiomarkdata_Timbre/no_box_attack'
            os.makedirs(output_txt_dir, exist_ok=True)
            output_file = os.path.join(output_txt_dir, f"soundstream_nq_{nq}.txt")

            with open(output_file, 'w') as txt_file:
                txt_file.write("Path, AudioSeal-B BA, AudioSeal Dection Ratio, SNR, Visqol score\n")
                for id, file in progress_bar:


                    path = os.path.join(output_dir, file)
                    path_list.append(extract_id(path))
                    waveform, sample_rate = torchaudio.load(path)
                    filename = ('_').join(file.split('_')[:-2])
                    waveform_pert_file = [file for file in os.listdir(output_dir_pert) if file.startswith(filename)].pop()
                    waveform_pert, sample_rate = torchaudio.load(os.path.join(output_dir_pert, waveform_pert_file))
                    payload_str = waveform_pert_file.split('_')[4]
                    original_msg_np = np.array(list(map(int, payload_str)))
                    original_msg = torch.tensor(original_msg_np, dtype=torch.int32).to(device=next(model.parameters()).device)
                    original_msgs.append(original_msg)
                    waveform_pert = waveform_pert.squeeze().unsqueeze(0)
                    snr = compute_snr(waveform.squeeze().unsqueeze(0),waveform_pert)
                    batch_SNRs.append(snr)
                    # Compute Visqol score
                    score = visqol.Measure(np.array(waveform.squeeze(), dtype=np.float64), np.array(waveform_pert.squeeze(), dtype=np.float64))
                    batch_visqol_scores.append(score.moslqo)
                    waveform_pert = torch.tensor(waveform_pert.unsqueeze(0)).to(device=next(model.parameters()).device)
                    batch_watermarked_signals.append(waveform_pert)

                    if args.save_pert:
                        torchaudio.save(os.path.join(output_dir_pert, file), waveform_pert.squeeze(0).detach().cpu(), sample_rate)
                    # Extract original payload from filename for BER calculation
                    if len(batch_watermarked_signals) == args.batch_size or id == len(file_list) - 1:
                        batch_watermarked_signals = torch.cat(batch_watermarked_signals, dim=0)
                        original_msgs = torch.stack(original_msgs)

                        bw_acc = get_bitacc(model, batch_watermarked_signals, original_msgs)

                        for i in range(len(batch_watermarked_signals)):
                            total_ba.append(bw_acc[i].item())
                            total_visqol_scores.append(batch_visqol_scores[i])
                            total_counts += 1
                            txt_file.write(f"{path_list[i]}, {bw_acc[i].item()}, {batch_SNRs[i]:.2f}, {batch_visqol_scores[i]}\n")

                        # Reset the batch
                        batch_watermarked_signals = []
                        batch_visqol_scores = []
                        batch_SNRs = []
                        original_msgs = []
                        path_list = []

                        current_average_BA = (sum(total_ba) / total_counts) * 100
                        current_average_visqol = sum(total_visqol_scores) / total_counts
                        progress_bar.set_description(f"{common_perturbation} w nq {nq} Avg BA: {current_average_BA:.2f}% - Avg Visqol: {current_average_visqol:.2f}")

                tau_ba = 0.83
                detection_fraction = detection_BA(total_ba, tau_ba)
                print(f'TPR for Timbre (Tau_ber={tau_ba}): {detection_fraction:.3f}\n')

    elif common_perturbation in ["opus"]:
        bitrate_list = [1, 2, 4, 8 ,16, 31]
        for bitrate in bitrate_list:
            total_ba = []
            total_visqol_scores = []
            total_counts = 0
            output_dir_pert = f'log_timbre_audiomarkdata_max_5s/common_pert_{common_perturbation}_bitrate_{bitrate*16}k'
            os.makedirs(output_dir_pert, exist_ok=True)
            file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            progress_bar = tqdm(enumerate(file_list), desc=f"Applying {common_perturbation}")

            batch_watermarked_signals = []
            batch_visqol_scores = []
            batch_SNRs = []
            original_msgs = []
            path_list = []
            visqol = api_visqol()
            output_txt_dir = f'txt_audiomarkdata_Timbre/no_box_attack'
            os.makedirs(output_txt_dir, exist_ok=True)
            output_file = os.path.join(output_txt_dir, f"{common_perturbation}_bitrate_{bitrate}.txt")

            with open(output_file, 'w') as txt_file:
                txt_file.write("Path, Timbre BA, SNR, Visqol score\n")
                for id, file in progress_bar:
                    try:
                        index_str, payload_str = file.rstrip('.wav').split('_')[-3:-1]
                        index = int(index_str)
                    except ValueError:
                        continue

                    original_msg_np = np.array(list(map(int, payload_str)))
                    original_msg = torch.tensor(original_msg_np, dtype=torch.int32).to(device=next(model.parameters()).device)
                    original_msgs.append(original_msg)

                    path = os.path.join(output_dir, file)
                    path_list.append(extract_id(path))
                    waveform, sample_rate = torchaudio.load(path)
                    waveform_pert = pert_opus(waveform, bitrate = 1000 * bitrate, quality = 1, cache = output_dir_pert)
                    snr = compute_snr(waveform.squeeze().unsqueeze(0),waveform_pert)
                    batch_SNRs.append(snr)
                    # Compute Visqol score
                    score = visqol.Measure(np.array(waveform.squeeze(), dtype=np.float64), np.array(waveform_pert.squeeze(), dtype=np.float64))
                    batch_visqol_scores.append(score.moslqo)
                    waveform_pert = waveform_pert.unsqueeze(0).to(device=next(model.parameters()).device)
                    batch_watermarked_signals.append(waveform_pert)

                    if args.save_pert:
                        torchaudio.save(os.path.join(output_dir_pert, file), waveform_pert.squeeze(0).detach().cpu(), sample_rate)
                    
                    # Extract original payload from filename for BER calculation
                    if len(batch_watermarked_signals) == args.batch_size or id == len(file_list) - 1:
                        batch_watermarked_signals = torch.cat(batch_watermarked_signals, dim=0)
                        original_msgs = torch.stack(original_msgs)

                        bw_acc = get_bitacc(model, batch_watermarked_signals, original_msgs)

                        for i in range(len(batch_watermarked_signals)):
                            total_ba.append(bw_acc[i].item())
                            total_visqol_scores.append(batch_visqol_scores[i])
                            total_counts += 1
                            txt_file.write(f"{path_list[i]}, {bw_acc[i].item()}, {batch_SNRs[i]:.2f}, {batch_visqol_scores[i]}\n")

                        # Reset the batch
                        batch_watermarked_signals = []
                        batch_visqol_scores = []
                        batch_SNRs = []
                        original_msgs = []
                        path_list = []

                        current_average_BA = (sum(total_ba) / total_counts) * 100
                        current_average_visqol = sum(total_visqol_scores) / total_counts
                        progress_bar.set_description(f"{common_perturbation} w bitrate {bitrate*16}k Avg BA: {current_average_BA:.2f}% - Avg Visqol: {current_average_visqol:.2f}")

                for tau_ba in [0.81]:
                    detection_fraction = detection_BA(total_ba, tau_ba)
                    print(f'TPR for Timbre (Tau_ber={tau_ba}): {detection_fraction:.3f}\n')

    elif common_perturbation in ["encodec"]:  # Changed from "opus" to "encodec"
        from transformers import EncodecModel, AutoProcessor
        import warnings
        warnings.filterwarnings("ignore", message=".*Could not find image processor class.*feature_extractor_type.*")
        # Load the ENCODeC model and processor
        model_encodec = EncodecModel.from_pretrained("facebook/encodec_24khz")
        processor_encodec = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        bandwidth_values = [1.5, 3.0, 6.0, 12.0, 24.0]
        for bandwidth in bandwidth_values:
            total_ba = []
            total_visqol_scores = []
            total_counts = 0
            output_dir_pert = f'log_timbre_audiomarkdata_max_5s/common_pert_{common_perturbation}_bandwidth_{bandwidth}'
            os.makedirs(output_dir_pert, exist_ok=True)
            encodec_cache = os.path.join(output_dir_pert, 'perturbed')
            os.makedirs(encodec_cache, exist_ok=True)
            file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            progress_bar = tqdm(enumerate(file_list), desc=f"Applying {common_perturbation}")
            
            batch_watermarked_signals = []
            batch_visqol_scores = []
            batch_SNRs = []
            original_msgs = []
            path_list = []
            output_txt_dir = f'txt_audiomarkdata_Timbre/no_box_attack'
            os.makedirs(output_txt_dir, exist_ok=True)
            output_file = os.path.join(output_txt_dir, f"{common_perturbation}_bandwidth_{bandwidth}.txt")

            for id, file in progress_bar:
                try:
                    index_str, payload_str = file.rstrip('.wav').split('_')[-3:-1]
                    index = int(index_str)
                except ValueError:
                    continue
                path = os.path.join(output_dir, file)
                path_list.append(extract_id(path))
                waveform, sample_rate = torchaudio.load(path)
                waveform_pert = pert_encodec(waveform, 24000, bandwidth, model_encodec, processor_encodec) 
                torchaudio.save(os.path.join(encodec_cache, file), waveform_pert.cpu().detach(), sample_rate)
            '''start visqol evaluation'''
            import subprocess
            subprocess.run(['python', 'encodec_visqol.py', '--origin', output_dir, '--perturb', encodec_cache])
            visqol_scores = np.load(os.path.join(encodec_cache, 'scores.npy'), allow_pickle=True).item()

            with open(output_file, 'w') as txt_file:
                txt_file.write("Path, Timbre BA, SNR, Visqol score\n")
                file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
                progress_bar = tqdm(enumerate(file_list), desc=f"Applying {common_perturbation}")
                for id, file in progress_bar:
                    try:
                        index_str, payload_str = file.rstrip('.wav').split('_')[-3:-1]
                        index = int(index_str)
                    except ValueError:
                        continue

                    original_msg_np = np.array(list(map(int, payload_str)))
                    original_msg = torch.tensor(original_msg_np, dtype=torch.int32).to(device=next(model.parameters()).device)
                    original_msgs.append(original_msg)

                    path = os.path.join(output_dir, file)
                    path_list.append(extract_id(path))
                    waveform, sample_rate = torchaudio.load(path)
                    waveform_pert, sample_rate = torchaudio.load(os.path.join(encodec_cache, file))
                    snr = compute_snr(waveform.squeeze().unsqueeze(0),waveform_pert)
                    batch_SNRs.append(snr)

                    waveform_pert = waveform_pert.unsqueeze(0).to(device=next(model.parameters()).device)
                    batch_watermarked_signals.append(waveform_pert)
                    batch_visqol_scores.append(visqol_scores[file])
                    
                    # Extract original payload from filename for BER calculation
                    if len(batch_watermarked_signals) == args.batch_size or id == len(file_list) - 1:
                        batch_watermarked_signals = torch.cat(batch_watermarked_signals, dim=0)
                        original_msgs = torch.stack(original_msgs)

                        bw_acc = get_bitacc(model, batch_watermarked_signals, original_msgs)

                        for i in range(len(batch_watermarked_signals)):
                            total_ba.append(bw_acc[i].item())
                            total_visqol_scores.append(batch_visqol_scores[i])
                            total_counts += 1
                            txt_file.write(f"{path_list[i]}, {bw_acc[i].item()}, {batch_SNRs[i]:.2f}, {batch_visqol_scores[i]}\n")

                        # Reset the batch
                        batch_watermarked_signals = []
                        batch_visqol_scores = []
                        batch_SNRs = []
                        original_msgs = []
                        path_list = []

                        current_average_BA = (sum(total_ba) / total_counts) * 100
                        current_average_visqol = sum(total_visqol_scores) / total_counts
                        progress_bar.set_description(f"{common_perturbation} w bandwidth {bandwidth} Avg BA: {current_average_BA:.2f}% - Avg Visqol: {current_average_visqol:.2f}")

                for tau_ba in [0.81]:
                    detection_fraction = detection_BA(total_ba, tau_ba)
                    print(f'TPR for Timbre (Tau_ber={tau_ba}): {detection_fraction:.3f}\n')

    elif common_perturbation in ["quantization"]:  
        import math
        quantization_levels = [2**2, 2**3, 2**4, 2**5, 2**6]
        for quantization_bit in quantization_levels:
            total_ba = []
            total_visqol_scores = []
            total_counts = 0
            output_dir_pert = f'log_timbre_audiomarkdata_max_5s/common_pert_{common_perturbation}_quantization_bit_{quantization_bit}'
            os.makedirs(output_dir_pert, exist_ok=True)
            file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            progress_bar = tqdm(enumerate(file_list), desc=f"Applying {common_perturbation}")

            batch_watermarked_signals = []
            batch_visqol_scores = []
            batch_SNRs = []
            original_msgs = []
            path_list = []
            visqol = api_visqol()
            output_txt_dir = f'txt_audiomarkdata_Timbre/no_box_attack'
            os.makedirs(output_txt_dir, exist_ok=True)
            output_file = os.path.join(output_txt_dir, f"{common_perturbation}_quantization_bit_{quantization_bit}.txt")

            with open(output_file, 'w') as txt_file:
                txt_file.write("Path, Timbre BA, SNR, Visqol score\n")
                for id, file in progress_bar:
                    try:
                        index_str, payload_str = file.rstrip('.wav').split('_')[-3:-1]
                        index = int(index_str)
                    except ValueError:
                        continue

                    original_msg_np = np.array(list(map(int, payload_str)))
                    original_msg = torch.tensor(original_msg_np, dtype=torch.int32).to(device=next(model.parameters()).device)
                    original_msgs.append(original_msg)

                    path = os.path.join(output_dir, file)
                    path_list.append(extract_id(path))
                    waveform, sample_rate = torchaudio.load(path)
                    waveform_pert = pert_quantization(waveform,quantization_bit) 
                    snr = compute_snr(waveform.squeeze().unsqueeze(0),waveform_pert)
                    batch_SNRs.append(snr)
                    # Compute Visqol score
                    score = visqol.Measure(np.array(waveform.squeeze(), dtype=np.float64), np.array(waveform_pert.squeeze(), dtype=np.float64))
                    batch_visqol_scores.append(score.moslqo)
                    waveform_pert = waveform_pert.unsqueeze(0).to(device=next(model.parameters()).device)
                    batch_watermarked_signals.append(waveform_pert)
                    if args.save_pert:
                        torchaudio.save(os.path.join(output_dir_pert, file), waveform_pert.squeeze(0).detach().cpu(), sample_rate)
                    # Extract original payload from filename for BER calculation
                    if len(batch_watermarked_signals) == args.batch_size or id == len(file_list) - 1:
                        batch_watermarked_signals = torch.cat(batch_watermarked_signals, dim=0)
                        original_msgs = torch.stack(original_msgs)
                        bw_acc = get_bitacc(model, batch_watermarked_signals, original_msgs)
                        for i in range(len(batch_watermarked_signals)):
                            total_ba.append(bw_acc[i].item())
                            total_visqol_scores.append(batch_visqol_scores[i])
                            total_counts += 1
                            txt_file.write(f"{path_list[i]}, {bw_acc[i].item()}, {batch_SNRs[i]:.2f}, {batch_visqol_scores[i]}\n")
                        # Reset the batch
                        batch_watermarked_signals = []
                        batch_visqol_scores = []
                        batch_SNRs = []
                        original_msgs = []
                        path_list = []

                        current_average_BA = (sum(total_ba) / total_counts) * 100
                        current_average_visqol = sum(total_visqol_scores) / total_counts
                        progress_bar.set_description(f"{common_perturbation} w quantization_bit {quantization_bit} Avg BA: {current_average_BA:.2f}% - Avg Visqol: {current_average_visqol:.2f}")

                tau_ba = 0.83
                detection_fraction = detection_BA(total_ba, tau_ba)
                print(f'TPR for Timbre (Tau_ber={tau_ba}): {detection_fraction:.3f}\n')

    elif common_perturbation in ["highpass", "lowpass"]:
        ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        for ratio in ratio_list:
            total_ba = []
            total_visqol_scores = []
            total_counts = 0
            output_dir_pert = f'log_timbre_audiomarkdata_max_5s/common_pert_{common_perturbation}_ratio_{ratio}'
            os.makedirs(output_dir_pert, exist_ok=True)
            file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            progress_bar = tqdm(enumerate(file_list), desc=f"Applying {common_perturbation}")

            batch_watermarked_signals = []
            batch_visqol_scores = []
            batch_SNRs = []
            original_msgs = []
            path_list = []
            visqol = api_visqol()
            output_txt_dir = f'txt_audiomarkdata_Timbre/no_box_attack'
            os.makedirs(output_txt_dir, exist_ok=True)
            output_file = os.path.join(output_txt_dir, f"{common_perturbation}_ratio_{ratio}.txt")

            with open(output_file, 'w') as txt_file:
                txt_file.write("Path, Timbre BA, SNR, Visqol score\n")
                for id, file in progress_bar:
                    try:
                        index_str, payload_str = file.rstrip('.wav').split('_')[-3:-1]
                        index = int(index_str)
                    except ValueError:
                        continue

                    original_msg_np = np.array(list(map(int, payload_str)))
                    original_msg = torch.tensor(original_msg_np, dtype=torch.int32).to(device=next(model.parameters()).device)
                    original_msgs.append(original_msg)

                    path = os.path.join(output_dir, file)
                    path_list.append(extract_id(path))
                    waveform, sample_rate = torchaudio.load(path)
                    if common_perturbation == 'highpass':
                        waveform_pert = pert_highpass(waveform,ratio,sample_rate) 
                    elif common_perturbation == 'lowpass':
                        waveform_pert = pert_lowpass(waveform,ratio,sample_rate) 
                    snr = compute_snr(waveform.squeeze().unsqueeze(0),waveform_pert)
                    batch_SNRs.append(snr)
                    # Compute Visqol score
                    score = visqol.Measure(np.array(waveform.squeeze(), dtype=np.float64), np.array(waveform_pert.squeeze(), dtype=np.float64))
                    batch_visqol_scores.append(score.moslqo)
                    waveform_pert = waveform_pert.unsqueeze(0).to(device=next(model.parameters()).device)
                    batch_watermarked_signals.append(waveform_pert)

                    if args.save_pert:
                        torchaudio.save(os.path.join(output_dir_pert, file), waveform_pert.squeeze(0).detach().cpu(), sample_rate)
                    # Extract original payload from filename for BER calculation
                    if len(batch_watermarked_signals) == args.batch_size or id == len(file_list) - 1:
                        batch_watermarked_signals = torch.cat(batch_watermarked_signals, dim=0)
                        original_msgs = torch.stack(original_msgs)
                        bw_acc = get_bitacc(model, batch_watermarked_signals, original_msgs)
                        for i in range(len(batch_watermarked_signals)):
                            total_ba.append(bw_acc[i].item())
                            total_visqol_scores.append(batch_visqol_scores[i])
                            total_counts += 1
                            txt_file.write(f"{path_list[i]}, {bw_acc[i].item()}, {batch_SNRs[i]:.2f}, {batch_visqol_scores[i]}\n")
                        # Reset the batch
                        batch_watermarked_signals = []
                        batch_visqol_scores = []
                        batch_SNRs = []
                        original_msgs = []
                        path_list = []

                        current_average_BA = (sum(total_ba) / total_counts) * 100
                        current_average_visqol = sum(total_visqol_scores) / total_counts
                        progress_bar.set_description(f"{common_perturbation} w ratio {ratio} Avg BA: {current_average_BA:.2f}% - Avg Visqol: {current_average_visqol:.2f}")
                tau_ba = 0.83
                detection_fraction = detection_BA(total_ba, tau_ba)
                print(f'TPR for Timbre (Tau_ber={tau_ba}): {detection_fraction:.3f}\n')

    elif common_perturbation in ["smooth"]:
        window_list = [6, 10, 14, 18, 22]
        for window_size in window_list:
            total_ba = []
            total_visqol_scores = []
            total_counts = 0
            output_dir_pert = f'log_timbre_audiomarkdata_max_5s/common_pert_{common_perturbation}_window_size_{window_size}'
            os.makedirs(output_dir_pert, exist_ok=True)
            file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            progress_bar = tqdm(enumerate(file_list), desc=f"Applying {common_perturbation}")

            batch_watermarked_signals = []
            batch_visqol_scores = []
            batch_SNRs = []
            original_msgs = []
            path_list = []
            visqol = api_visqol()
            output_txt_dir = f'txt_audiomarkdata_Timbre/no_box_attack'
            os.makedirs(output_txt_dir, exist_ok=True)
            output_file = os.path.join(output_txt_dir, f"{common_perturbation}_window_size_{window_size}.txt")

            with open(output_file, 'w') as txt_file:
                txt_file.write("Path, Timbre BA, SNR, Visqol score\n")
                for id, file in progress_bar:
                    try:
                        index_str, payload_str = file.rstrip('.wav').split('_')[-3:-1]
                        index = int(index_str)
                    except ValueError:
                        continue

                    original_msg_np = np.array(list(map(int, payload_str)))
                    original_msg = torch.tensor(original_msg_np, dtype=torch.int32).to(device=next(model.parameters()).device)
                    original_msgs.append(original_msg)

                    path = os.path.join(output_dir, file)
                    path_list.append(extract_id(path))
                    waveform, sample_rate = torchaudio.load(path)
                    waveform_pert = pert_smooth(waveform, window_size) 
                    snr = compute_snr(waveform.squeeze().unsqueeze(0),waveform_pert)
                    batch_SNRs.append(snr)
                    # Compute Visqol score
                    score = visqol.Measure(np.array(waveform.squeeze(), dtype=np.float64), np.array(waveform_pert.squeeze(), dtype=np.float64))
                    batch_visqol_scores.append(score.moslqo)
                    waveform_pert = waveform_pert.unsqueeze(0).to(device=next(model.parameters()).device)
                    batch_watermarked_signals.append(waveform_pert)
                    if args.save_pert:
                        torchaudio.save(os.path.join(output_dir_pert, file), waveform_pert.squeeze(0).detach().cpu(), sample_rate)
                    # Extract original payload from filename for BER calculation
                    if len(batch_watermarked_signals) == args.batch_size or id == len(file_list) - 1:
                        batch_watermarked_signals = torch.cat(batch_watermarked_signals, dim=0)
                        original_msgs = torch.stack(original_msgs)
                        bw_acc = get_bitacc(model, batch_watermarked_signals, original_msgs)
                        for i in range(len(batch_watermarked_signals)):
                            total_ba.append(bw_acc[i].item())
                            total_visqol_scores.append(batch_visqol_scores[i])
                            total_counts += 1
                            txt_file.write(f"{path_list[i]}, {bw_acc[i].item()}, {batch_SNRs[i]:.2f}, {batch_visqol_scores[i]}\n")
                        # Reset the batch
                        batch_watermarked_signals = []
                        batch_visqol_scores = []
                        batch_SNRs = []
                        original_msgs = []
                        path_list = []

                        current_average_BA = (sum(total_ba) / total_counts) * 100
                        current_average_visqol = sum(total_visqol_scores) / total_counts
                        progress_bar.set_description(f"{common_perturbation} w window size {window_size} Avg BA: {current_average_BA:.2f}% - Avg Visqol: {current_average_visqol:.2f}")

                tau_ba = 0.83
                detection_fraction = detection_BA(total_ba, tau_ba)
                print(f'TPR for Timbre (Tau_ber={tau_ba}): {detection_fraction:.3f}\n')

    elif common_perturbation in ["echo"]:
        decay_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        for decay in decay_list:
            total_ba = []
            total_visqol_scores = []            
            total_counts = 0
            output_dir_pert = f'log_timbre_audiomarkdata_max_5s/common_pert_{common_perturbation}_decay{decay}'
            os.makedirs(output_dir_pert, exist_ok=True)
            file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            progress_bar = tqdm(enumerate(file_list), desc=f"Applying {common_perturbation}")

            batch_watermarked_signals = []
            batch_visqol_scores = []
            batch_SNRs = []
            original_msgs = []
            path_list = []
            visqol = api_visqol()
            output_txt_dir = f'txt_audiomarkdata_Timbre/no_box_attack'
            os.makedirs(output_txt_dir, exist_ok=True)
            output_file = os.path.join(output_txt_dir, f"{common_perturbation}_decay_{decay}.txt")

            with open(output_file, 'w') as txt_file:
                txt_file.write("Path, Timbre BA, SNR, Visqol score\n")
                for id, file in progress_bar:
                    try:
                        index_str, payload_str = file.rstrip('.wav').split('_')[-3:-1]
                        index = int(index_str)
                    except ValueError:
                        continue

                    original_msg_np = np.array(list(map(int, payload_str)))
                    original_msg = torch.tensor(original_msg_np, dtype=torch.int32).to(device=next(model.parameters()).device)
                    original_msgs.append(original_msg)

                    path = os.path.join(output_dir, file)
                    path_list.append(extract_id(path))
                    waveform, sample_rate = torchaudio.load(path)
                    waveform_pert = pert_echo(waveform, duration=decay)
                    snr = compute_snr(waveform.squeeze().unsqueeze(0),waveform_pert)
                    batch_SNRs.append(snr)
                    # Compute Visqol score
                    score = visqol.Measure(np.array(waveform.squeeze(), dtype=np.float64), np.array(waveform_pert.squeeze(), dtype=np.float64))
                    batch_visqol_scores.append(score.moslqo)
                    waveform_pert = waveform_pert.unsqueeze(0).to(device=next(model.parameters()).device)
                    batch_watermarked_signals.append(waveform_pert)
                    if args.save_pert:
                        torchaudio.save(os.path.join(output_dir_pert, file), waveform_pert.squeeze(0).detach().cpu(), sample_rate)
                    # Extract original payload from filename for BER calculation
                    if len(batch_watermarked_signals) == args.batch_size or id == len(file_list) - 1:
                        batch_watermarked_signals = torch.cat(batch_watermarked_signals, dim=0)
                        original_msgs = torch.stack(original_msgs)
                        bw_acc = get_bitacc(model, batch_watermarked_signals, original_msgs)
                        for i in range(len(batch_watermarked_signals)):
                            total_ba.append(bw_acc[i].item())
                            total_visqol_scores.append(batch_visqol_scores[i])
                            total_counts += 1
                            txt_file.write(f"{path_list[i]}, {bw_acc[i].item()}, {batch_SNRs[i]:.2f}, {batch_visqol_scores[i]}\n")
                        # Reset the batch
                        batch_watermarked_signals = []
                        batch_visqol_scores = []
                        batch_SNRs = []
                        original_msgs = []
                        path_list = []

                        current_average_BA = (sum(total_ba) / total_counts) * 100
                        current_average_visqol = sum(total_visqol_scores) / total_counts
                        progress_bar.set_description(f"{common_perturbation} decay {decay}s Avg BA: {current_average_BA:.2f}% - Avg Visqol: {current_average_visqol:.2f}")

                tau_ba = 0.83
                detection_fraction = detection_BA(total_ba, tau_ba)
                print(f'TPR for Timbre (Tau_ber={tau_ba}): {detection_fraction:.3f}\n')

    elif common_perturbation in ["mp3"]:
        bitrate_list = [8, 16, 24, 32, 40]
        for bitrate in bitrate_list:
            total_ba = []
            total_visqol_scores = []
            total_counts = 0
            output_dir_pert = f'log_timbre_audiomarkdata_max_5s/common_pert_{common_perturbation}_bitrate_{bitrate}'
            os.makedirs(output_dir_pert, exist_ok=True)
            file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            progress_bar = tqdm(enumerate(file_list), desc=f"Applying {common_perturbation}")

            batch_watermarked_signals = []
            batch_visqol_scores = []
            batch_SNRs = []
            original_msgs = []
            path_list = []
            visqol = api_visqol()
            output_txt_dir = f'txt_audiomarkdata_Timbre/no_box_attack'
            os.makedirs(output_txt_dir, exist_ok=True)
            output_file = os.path.join(output_txt_dir, f"{common_perturbation}_bitrate_{bitrate}.txt")

            with open(output_file, 'w') as txt_file:
                txt_file.write("Path, Timbre BA, SNR, Visqol score\n")
                for id, file in progress_bar:
                    try:
                        index_str, payload_str = file.rstrip('.wav').split('_')[-3:-1]
                        index = int(index_str)
                    except ValueError:
                        continue

                    original_msg_np = np.array(list(map(int, payload_str)))
                    original_msg = torch.tensor(original_msg_np, dtype=torch.int32).to(device=next(model.parameters()).device)
                    original_msgs.append(original_msg)

                    path = os.path.join(output_dir, file)
                    path_list.append(extract_id(path))
                    waveform, sample_rate = torchaudio.load(path)
                    waveform_pert = pert_mp3(waveform, bitrate)
                    snr = compute_snr(waveform.squeeze().unsqueeze(0),waveform_pert)
                    batch_SNRs.append(snr)
                    # Compute Visqol score
                    score = visqol.Measure(np.array(waveform.squeeze(), dtype=np.float64), np.array(waveform_pert.squeeze(), dtype=np.float64))
                    batch_visqol_scores.append(score.moslqo)
                    waveform_pert = waveform_pert.unsqueeze(0).to(device=next(model.parameters()).device)
                    batch_watermarked_signals.append(waveform_pert)
                    if args.save_pert:
                        torchaudio.save(os.path.join(output_dir_pert, file), waveform_pert.squeeze(0).detach().cpu(), sample_rate)              
                    # Extract original payload from filename for BER calculation
                    if len(batch_watermarked_signals) == args.batch_size or id == len(file_list) - 1:
                        batch_watermarked_signals = torch.cat(batch_watermarked_signals, dim=0)
                        original_msgs = torch.stack(original_msgs)
                        bw_acc = get_bitacc(model, batch_watermarked_signals, original_msgs)
                        for i in range(len(batch_watermarked_signals)):
                            total_ba.append(bw_acc[i].item())
                            total_visqol_scores.append(batch_visqol_scores[i])
                            total_counts += 1
                            txt_file.write(f"{path_list[i]}, {bw_acc[i].item()}, {batch_SNRs[i]:.2f}, {batch_visqol_scores[i]}\n")
                        # Reset the batch
                        batch_watermarked_signals = []
                        batch_visqol_scores = []
                        batch_SNRs = []
                        original_msgs = []
                        path_list = []

                        current_average_BA = (sum(total_ba) / total_counts) * 100
                        current_average_visqol = sum(total_visqol_scores) / total_counts
                        progress_bar.set_description(f"{common_perturbation} w bitrate {bitrate} Avg BA: {current_average_BA:.2f}% - Avg Visqol: {current_average_visqol:.2f}")

                tau_ba = 0.83
                detection_fraction = detection_BA(total_ba, tau_ba)
                print(f'TPR for Timbre (Tau_ber={tau_ba}): {detection_fraction:.3f}\n')

    else: 
        raise NotImplementedError

def compute_snr(signal, noisy_signal):
    # Compute the power of the original signal
    signal_power = torch.mean(signal ** 2)
    
    # Compute the power of the noise
    noise = noisy_signal - signal
    noise_power = torch.mean(noise ** 2)
    # print(f'mean_noise_power: {noise_power}')
    
    # Compute the Signal-to-Noise Ratio (SNR)
    snr = 10 * torch.log10(signal_power / noise_power)
    
    return snr.item()

def compute_scale_factor(signal, noisy_signal, target_snr_db):
    snr = compute_snr(signal, noisy_signal)
    if snr < target_snr_db:
        scale_factor = 10 ** ((target_snr_db - snr) / 10)
    else: 
        scale_factor = 1
    return scale_factor

def pert_time_stretch(waveform, sample_rate, speed_factor):
    waveform_np = waveform.numpy()
    if waveform_np.shape[0] == 1:
        waveform_np = waveform_np.squeeze()
    
    waveform_stretched = librosa.effects.time_stretch(waveform_np, rate=speed_factor)
    time_stretched_waveform = torch.from_numpy(waveform_stretched).unsqueeze(0).float()
    if time_stretched_waveform.shape[1] < waveform.shape[1]:
        time_stretched_waveform = F.pad(time_stretched_waveform, (0, waveform.shape[1] - time_stretched_waveform.shape[1]))
    elif time_stretched_waveform.shape[1] > waveform.shape[1]:
        time_stretched_waveform = time_stretched_waveform[:, :waveform.shape[1]]
    
    return time_stretched_waveform, sample_rate

def pert_Gaussian_noise(waveform, snr_db):
    # Calculate signal power
    signal_power = torch.mean(waveform**2).to(device=waveform.device)

    # Calculate noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate noise with calculated noise power
    noise = torch.randn(waveform.size()) * torch.sqrt(noise_power)
    waveform_noisy = waveform + noise
    snr = compute_snr(waveform, waveform_noisy)
    # print(f'!snr:{snr}')
    # assert 0==1
    return waveform_noisy

# Download the sample noise file
from torchaudio.utils import download_asset
SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")

def pert_background_noise(waveform, snr_db):
    noise, _ = torchaudio.load(SAMPLE_NOISE)
    # Resize the noise to match the length of the waveform
    if noise.size(1) > waveform.size(1):
        noise = noise[:, :waveform.size(1)]
    else:
        repeat_times = waveform.size(1) // noise.size(1) + 1
        noise = noise.repeat(1, repeat_times)
        noise = noise[:, :waveform.size(1)]
    
    # Calculate signal power and noise power
    signal_power = torch.mean(waveform**2)
    noise_power = torch.mean(noise**2)

    # Calculate the scaling factor for the noise
    snr_linear = 10 ** (snr_db / 10)
    scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))

    # Scale and add the noise
    noisy_waveform = waveform + noise * scaling_factor
    return noisy_waveform

def api_visqol():
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    from visqol.pb2 import similarity_result_pb2
    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = 16000
    config.options.use_speech_scoring = True
    svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    return api

"""path: str, 
waveform_tc: numpy.ndarray[numpy.int16], 
sample_rate: int, 
bitrate: int = -1000, 
signal_type: int = 0, 
encoder_complexity: int = 10) -> None:
The waveform must be a numpy array of np.int16 type, and shape [samples (time axis), channels]. 
Recommended sample rate is 48000. You can specify the bitrate in bits/s, as well as
encoder_complexity (in range [0, 10] inclusive, the higher the better quality at given bitrate, 
but more CPU usage, 10 is recommended). Finally, there is signal_type option, that can help to
improve quality for specific audio, types (0 = AUTO (default), 1 = MUSIC, 2 = SPEECH)."""
import opuspy
def pert_opus(waveform: torch.tensor, bitrate: int, quality: int, cache: str) -> None:
    waveform_scaled = waveform * 32768
    waveform_scaled = waveform_scaled.reshape(-1,1).numpy()
    cache_file = os.path.join(cache, "temp.opus")
    opuspy.write(cache_file, waveform_scaled, sample_rate = 16000, 
                bitrate = bitrate, signal_type = 0, encoder_complexity = quality)
    pert_waveform, sampling_rate = opuspy.read(cache_file) # NOTE that the sample rate is always 48000
    os.remove(cache_file)
    resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
    pert_waveform = torch.tensor(pert_waveform, dtype=torch.float32).reshape(1,-1)
    pert_waveform /= 32768
    return resampler(pert_waveform)

def pert_encodec(waveform, sample_rate, bandwidth, model_encodec, processor):
    # Process the waveform to match the model's expected input
    # print(waveform.shape)
    model_encodec = model_encodec.to('cuda')
    waveform = waveform.squeeze().numpy()  # Assuming waveform is a PyTorch tensor
    inputs = processor(raw_audio=waveform, sampling_rate=sample_rate, return_tensors="pt").to('cuda')

    # Encode and decode the audio sample using ENCODeC
    encoder_outputs = model_encodec.encode(inputs["input_values"], inputs["padding_mask"], bandwidth)
    audio_values = model_encodec.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
    # The output 'audio_values' is the perturbed waveform
    return torch.tensor(audio_values.detach().cpu()).squeeze().unsqueeze(0)

def pert_encodec_author(waveform, sample_rate, bandwidth, model_encodec):
    # Process the waveform to match the model's expected input
    waveform = julius.resample_frac(waveform, 16000, sample_rate)
    waveform = waveform.unsqueeze(0).to(device=next(model_encodec.parameters()).device)
    model_encodec.set_num_codebooks()
    code, scale = model_encodec.encode(waveform, bandwidth)
    audio_encodec = model_encodec.decode(code, scale)
    audio_encodec = julius.resample_frac(waveform, sample_rate, 16000)
    # The output 'audio_values' is the perturbed waveform
    return torch.tensor(audio_encodec).squeeze().unsqueeze(0)

def pert_quantization(waveform, quantization_bit):
    # Normalize the waveform to the range of the quantization levels
    min_val, max_val = waveform.min(), waveform.max()
    normalized_waveform = (waveform - min_val) / (max_val - min_val)

    # Quantize the normalized waveform
    quantized_waveform = torch.round(normalized_waveform * (quantization_bit - 1))

    # Rescale the quantized waveform back to the original range
    rescaled_waveform = (quantized_waveform / (quantization_bit - 1)) * (max_val - min_val) + min_val

    return rescaled_waveform

def audio_effect_return(
    tensor: torch.Tensor, mask: tp.Optional[torch.Tensor]
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the mask if it was in the input otherwise only the output tensor"""
    if mask is None:
        return tensor
    else:
        return tensor, mask

def pert_highpass(
    waveform: torch.Tensor,
    cutoff_ratio: float,
    sample_rate: int = 16000,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    return audio_effect_return(
        tensor=julius.highpass_filter(waveform, cutoff=cutoff_ratio),
        mask=mask,
    )
    
def pert_lowpass(
    waveform: torch.Tensor,
    cutoff_ratio: float,
    sample_rate: int = 16000,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    return audio_effect_return(
        tensor=julius.lowpass_filter(waveform, cutoff=cutoff_ratio),
        mask=mask,
    )

def pert_smooth(
    waveform: torch.Tensor,
    window_size: int = 5,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    waveform = waveform.unsqueeze(0)
    window_size = int(window_size)
    # Create a uniform smoothing kernel
    kernel = torch.ones(1, 1, window_size).type(waveform.type()) / window_size
    kernel = kernel.to(waveform.device)

    smoothed = julius.fft_conv1d(waveform, kernel)
    # Ensure tensor size is not changed
    tmp = torch.zeros_like(waveform)
    tmp[..., : smoothed.shape[-1]] = smoothed
    smoothed = tmp

    return audio_effect_return(tensor=smoothed, mask=mask).squeeze().unsqueeze(0)

def pert_echo(
    tensor: torch.Tensor,
    # volume_range: tuple = (0.1, 0.5),
    # duration_range: tuple = (0.1, 0.5),
    volume: float = 0.4,
    duration: float = 0.1,
    sample_rate: int = 16000,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Attenuating the audio volume by a factor of 0.4, delaying it by 100ms,
    and then overlaying it with the original.

    :param tensor: 3D Tensor representing the audio signal [bsz, channels, frames]
    :param echo_volume: volume of the echo signal
    :param sample_rate: Sample rate of the audio signal.
    :return: Audio signal with reverb.
    """
    tensor = tensor.unsqueeze(0)
    # Create a simple impulse response
    # Duration of the impulse response in seconds
    # duration = torch.FloatTensor(1).uniform_(*duration_range)
    duration = torch.Tensor([duration])
    # volume = torch.FloatTensor(1).uniform_(*volume_range)
    volume = torch.Tensor([volume])
    n_samples = int(sample_rate * duration)
    impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)

    # Define a few reflections with decreasing amplitude
    impulse_response[0] = 1.0  # Direct sound

    impulse_response[
        int(sample_rate * duration) - 1
    ] = volume  # First reflection after 100ms

    # Add batch and channel dimensions to the impulse response
    impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

    # Convolve the audio signal with the impulse response
    reverbed_signal = julius.fft_conv1d(tensor, impulse_response)

    # Normalize to the original amplitude range for stability
    reverbed_signal = (
        reverbed_signal
        / torch.max(torch.abs(reverbed_signal))
        * torch.max(torch.abs(tensor))
    )

    # Ensure tensor size is not changed
    tmp = torch.zeros_like(tensor)
    tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal
    reverbed_signal = tmp
    reverbed_signal = reverbed_signal.squeeze(0)
    return audio_effect_return(tensor=reverbed_signal, mask=mask)

def pert_mp3(waveform, bitrate, sample_rate=16000):
    mp3_compressor = Mp3Compression(
    min_bitrate=bitrate,  # Set the minimum bitrate
    max_bitrate=bitrate,  # Set the maximum bitrate
    backend="pydub",  # Choose the backend
    p=1.0)  # Set the probability to 1 to always apply the effect
    waveform = waveform.detach().cpu().numpy()
    mp3_compressor.randomize_parameters(waveform, sample_rate)
    waveform_pert = mp3_compressor.apply(waveform, sample_rate)
    return torch.tensor(waveform_pert)


class Mp3Compression(BaseWaveformTransform):
    """Compress the audio using an MP3 encoder to lower the audio quality.
    This may help machine learning models deal with compressed, low-quality audio.

    This transform depends on either lameenc or pydub/ffmpeg.

    Note that bitrates below 32 kbps are only supported for low sample rates (up to 24000 Hz).

    Note: When using the lameenc backend, the output may be slightly longer than the input due
    to the fact that the LAME encoder inserts some silence at the beginning of the audio.

    Warning: This transform writes to disk, so it may be slow. Ideally, the work should be done
    in memory. Contributions are welcome.
    """

    supports_multichannel = True

    SUPPORTED_BITRATES = [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        192,
        224,
        256,
        320,
    ]

    def __init__(
        self,
        min_bitrate: int = 8,
        max_bitrate: int = 64,
        backend: str = "pydub",
        p: float = 0.5,
    ):
        """
        :param min_bitrate: Minimum bitrate in kbps
        :param max_bitrate: Maximum bitrate in kbps
        :param backend: "pydub" or "lameenc".
            Pydub may use ffmpeg under the hood.
                Pros: Seems to avoid introducing latency in the output.
                Cons: Slower than lameenc.
            lameenc:
                Pros: You can set the quality parameter in addition to bitrate.
                Cons: Seems to introduce some silence at the start of the audio.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if min_bitrate < self.SUPPORTED_BITRATES[0]:
            raise ValueError(
                "min_bitrate must be greater than or equal to"
                f" {self.SUPPORTED_BITRATES[0]}"
            )
        if max_bitrate > self.SUPPORTED_BITRATES[-1]:
            raise ValueError(
                "max_bitrate must be less than or equal to"
                f" {self.SUPPORTED_BITRATES[-1]}"
            )
        if max_bitrate < min_bitrate:
            raise ValueError("max_bitrate must be >= min_bitrate")

        is_any_supported_bitrate_in_range = any(
            min_bitrate <= bitrate <= max_bitrate for bitrate in self.SUPPORTED_BITRATES
        )
        if not is_any_supported_bitrate_in_range:
            raise ValueError(
                "There is no supported bitrate in the range between the specified"
                " min_bitrate and max_bitrate. The supported bitrates are:"
                f" {self.SUPPORTED_BITRATES}"
            )

        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        if backend not in ("pydub", "lameenc"):
            raise ValueError('backend must be set to either "pydub" or "lameenc"')
        self.backend = backend
        self.post_gain_factor = None

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            bitrate_choices = [
                bitrate
                for bitrate in self.SUPPORTED_BITRATES
                if self.min_bitrate <= bitrate <= self.max_bitrate
            ]
            self.parameters["bitrate"] = random.choice(bitrate_choices)

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        if self.backend == "lameenc":
            return self.apply_lameenc(samples, sample_rate)
        elif self.backend == "pydub":
            return self.apply_pydub(samples, sample_rate)
        else:
            raise Exception("Backend {} not recognized".format(self.backend))

    def maybe_pre_gain(self, samples):
        """
        If the audio is too loud, gain it down to avoid distortion in the audio file to
        be encoded.
        """
        greatest_abs_sample = get_max_abs_amplitude(samples)
        if greatest_abs_sample > 1.0:
            self.post_gain_factor = greatest_abs_sample
            samples = samples * (1.0 / greatest_abs_sample)
        else:
            self.post_gain_factor = None
        return samples

    def maybe_post_gain(self, samples):
        """If the audio was pre-gained down earlier, post-gain it up to compensate here."""
        if self.post_gain_factor is not None:
            samples = samples * self.post_gain_factor
        return samples

    def apply_lameenc(self, samples: NDArray[np.float32], sample_rate: int):
        try:
            import lameenc
        except ImportError:
            print(
                (
                    "Failed to import the lame encoder. Maybe it is not installed? "
                    "To install the optional lameenc dependency of audiomentations,"
                    " do `pip install audiomentations[extras]` or simply"
                    " `pip install lameenc`"
                ),
                file=sys.stderr,
            )
            raise

        assert samples.dtype == np.float32

        samples = self.maybe_pre_gain(samples)

        int_samples = convert_float_samples_to_int16(samples).T

        num_channels = 1 if samples.ndim == 1 else samples.shape[0]

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(self.parameters["bitrate"])
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(num_channels)
        encoder.set_quality(7)  # 2 = highest, 7 = fastest
        encoder.silence()

        mp3_data = encoder.encode(int_samples.tobytes())
        mp3_data += encoder.flush()

        # Write a temporary MP3 file that will then be decoded
        tmp_dir = tempfile.gettempdir()
        tmp_file_path = os.path.join(
            tmp_dir, "tmp_compressed_{}.mp3".format(str(uuid.uuid4())[0:12])
        )
        with open(tmp_file_path, "wb") as f:
            f.write(mp3_data)

        degraded_samples, _ = librosa.load(tmp_file_path, sr=sample_rate, mono=False)

        os.unlink(tmp_file_path)

        degraded_samples = self.maybe_post_gain(degraded_samples)

        if num_channels == 1:
            if int_samples.ndim == 1 and degraded_samples.ndim == 2:
                degraded_samples = degraded_samples.flatten()
            elif int_samples.ndim == 2 and degraded_samples.ndim == 1:
                degraded_samples = degraded_samples.reshape((1, -1))

        return degraded_samples

    def apply_pydub(self, samples: NDArray[np.float32], sample_rate: int):
        try:
            import pydub
        except ImportError:
            print(
                (
                    "Failed to import pydub. Maybe it is not installed? "
                    "To install the optional pydub dependency of audiomentations,"
                    " do `pip install audiomentations[extras]` or simply"
                    " `pip install pydub`"
                ),
                file=sys.stderr,
            )
            raise

        assert samples.dtype == np.float32

        samples = self.maybe_pre_gain(samples)

        int_samples = convert_float_samples_to_int16(samples).T
        num_channels = 1 if samples.ndim == 1 else samples.shape[0]
        audio_segment = pydub.AudioSegment(
            int_samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=int_samples.dtype.itemsize,
            channels=num_channels,
        )

        tmp_dir = tempfile.gettempdir()
        tmp_file_path = os.path.join(
            tmp_dir, "tmp_compressed_{}.mp3".format(str(uuid.uuid4())[0:12])
        )

        bitrate_string = "{}k".format(self.parameters["bitrate"])
        file_handle = audio_segment.export(tmp_file_path, bitrate=bitrate_string)
        file_handle.close()

        degraded_samples, _ = librosa.load(tmp_file_path, sr=sample_rate, mono=False)

        os.unlink(tmp_file_path)

        degraded_samples = self.maybe_post_gain(degraded_samples)

        if num_channels == 1:
            if int_samples.ndim == 1 and degraded_samples.ndim == 2:
                degraded_samples = degraded_samples.flatten()
            elif int_samples.ndim == 2 and degraded_samples.ndim == 1:
                degraded_samples = degraded_samples.reshape((1, -1))

        return degraded_samples
    
def adjust_padding_ss_model(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Conv1d):
            # Calculate the new padding value
            # This formula assumes stride = 1 and dilation = 1 for simplicity
            new_padding = (module.kernel_size[0] - 1) // 2
            # Set the new padding
            # module.padding = (new_padding,)
            module.padding = 'valid'
        elif isinstance(module, torch.nn.ConvTranspose1d):
            # You might also want to adjust padding for transposed convolutions if necessary
            # module.padding = 'valid'
            pass
        else:
            # Recursively adjust padding for child modules
            adjust_padding_ss_model(module)

def main():
    args = parse_arguments()

    np.random.seed(42)
    torch.manual_seed(42)

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    import yaml
    from timbre.model.conv2_mel_modules import Encoder, Decoder
    process_config = yaml.load(open("timbre/config/process.yaml", "r"), Loader=yaml.FullLoader)

    model_config = yaml.load(open("timbre/config/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open("timbre/config/train.yaml", "r"), Loader=yaml.FullLoader)
    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"]
    nlayers_encoder = model_config["layer"]["nlayers_encoder"]
    nlayers_decoder = model_config["layer"]["nlayers_decoder"]
    attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]
    attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
    generator = Encoder(process_config, model_config, 30, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
    detector = Decoder(process_config, model_config, 30, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    checkpoint = torch.load('timbre/results/ckpt/pth/compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar')
    generator.load_state_dict(checkpoint['encoder'])
    detector.load_state_dict(checkpoint['decoder'], strict=False)
    generator.eval()
    detector.eval()
    # decoder.robust = False

    data_dir = "audiomarkdata/sample_20k" #NOTE: Change this to the path of the Common Voice dataset


    output_dir = f'audiomarkdata_timbre_max_5s'
    dataset_dir = os.path.join(output_dir, 'watermarked')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    if args.encode:
        models = [generator, detector]
        encode_audio_files(models, data_dir, output_dir, args.max_length)

    if args.common_perturbation != '':
        decode_audio_files_perturb(detector, dataset_dir, args.common_perturbation, args)
    else:
        decode_audio_files(detector, dataset_dir, args.batch_size)

if __name__ == "__main__":
    main()
