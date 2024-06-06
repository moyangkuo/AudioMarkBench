import os
import argparse
import numpy as np
import torch
import torchaudio
# import audioseal
from audioseal import AudioSeal
# import wavmark
import wavmark
from wavmark.utils import wm_add_util


# import timbre
import yaml
from timbre.model.conv2_mel_modules import Decoder


from tqdm import tqdm
from art.estimators.classification import PyTorchClassifier
import torch.optim as optim
import torch.nn as nn
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Audio Watermarking with WavMark")
    parser.add_argument("--testset_size", type=int, default=200, help="Number of samples from the test set to process")
    parser.add_argument("--encode", action="store_true", help="Run the encoding process before decoding")
    parser.add_argument("--length", type=int, default=5*16000, help="Length of the audio samples")

    parser.add_argument("--gpu", type=int, default=7, help="GPU device index to use")
    
    parser.add_argument("--iter", type=int, default=1000, help="Number of iterations for the attack")
    parser.add_argument("--pert_boundary", type=float, default=0.001, help="Perturbation boundary for the attack")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the attack")
    parser.add_argument("--rescale_snr", type=float, default=60, help="rescaled SNR after applying the perturbation")
    
    parser.add_argument("--whitebox_folder", type=str, default="whitebox_debug", help="Folder to save the whitebox attack results")
    
    parser.add_argument("--tau", type=float, default=0.1, help="Threshold for the detector")

    parser.add_argument("--attack_bitstring", action="store_true", default=False, help="If set, perturb the bitstring instead of the detection probability")
    parser.add_argument("--model", type=str, default='audioseal', choices=['audioseal','wavmark', 'timbre'], help="Model to be attacked")
    parser.add_argument("--dataset", type=str, default="audiomark", help="Dataset to use for the attack")

    print("Arguments: ", parser.parse_args())
    return parser.parse_args()
    

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


class WatermarkDetectorWrapper():
    def __init__(self, model, message, on_bitstring, model_type, threshold, device):
        self.model = model
        self._device = device
        self.message = message.to(self._device)
        self.on_bitstring = on_bitstring
        self.model.to(self._device)
        self.model_type = model_type
        self.threshold = threshold
        if model_type == 'timbre':
            self.bwacc = self.bwacc_timbre
            self.get_loss = self.loss_timbre
        elif model_type == 'audioseal':
            self.bwacc = self.bwacc_audioseal
            self.get_loss = self.loss_audioseal
        elif model_type == 'wavmark':
            wavmark_start_bit = wm_add_util.fix_pattern[0:16]  # 16 bits of watermark
            start_bit = torch.tensor(wavmark_start_bit, dtype=torch.float32)
            self.reversed_start_bit = 1 - start_bit
            self.reversed_start_bit = self.reversed_start_bit.repeat(20, 1)          
            self.reversed_start_bit = self.reversed_start_bit.to(self._device)
            self.total_detect_points = torch.arange(0, 800 * 80, 800)
            self.bwacc = self.bwacc_wavmark
            self.get_loss = self.loss_wavmark


    def loss_audioseal(self, signal):
        results, messages = self.model(signal)
        reversed_message = 1 - self.message
        loss = nn.CrossEntropyLoss()
        cross_entropy_loss = loss(messages.squeeze(), reversed_message)
        class_1_probs = results[:, 1, :]
        penalty = torch.relu(class_1_probs)
        total_penalty = torch.sum(penalty)
        return total_penalty + cross_entropy_loss
    
    def loss_timbre(self, signal):
        payload = self.model.test_forward(signal) # signal: [1,1,80000]
        message = (1 - self.message) * 2 - 1
        payload = payload.to(message.device)
        loss = nn.CrossEntropyLoss()
        cross_entropy_loss = loss(payload.squeeze(), message)
        return cross_entropy_loss

    def loss_wavmark(self, signal):
        signal = signal.squeeze()
        select_indices = torch.randint(0, 80, (20,))
        detect_points = self.total_detect_points[select_indices]
        slices = torch.stack([signal[..., p:p + 16000] for p in detect_points]).to(self._device)
        batch_messages = self.model.decode(slices)
        decoded_start_bits = batch_messages[:, 0:16]
        decoded_messages = batch_messages[:, 16:]
        loss_fn = nn.BCEWithLogitsLoss()
        start_bit_loss = loss_fn(decoded_start_bits, self.reversed_start_bit)
        reversed_msg = 1 - self.message
        msg_loss = loss_fn(decoded_messages, reversed_msg.repeat(20, 1))
        return start_bit_loss + msg_loss 
    

    def bwacc_audioseal(self, signal):
        result, msg_decoded = self.model.detect_watermark(signal)
        if self.on_bitstring:
            if msg_decoded is None:
                return torch.zeros(1)
            else: 
                bitacc = 1 - torch.sum(torch.abs(self.message - msg_decoded)) / self.message.numel()
                return bitacc
        else:
            return result
        
    def bwacc_wavmark(self, signal):
        signal = signal.squeeze().detach().cpu()
        # payload, info = wavmark.decode_watermark(self.model, signal)
        payload, info = wavmark.decode_watermark(self.model, signal)

        if payload is None:
            return 0
        else: 
            payload = torch.tensor(payload).to(self.message.device)
            bitacc = 1 - torch.sum(torch.abs(self.message - payload)) / self.message.numel()
            return bitacc.item()

    def bwacc_timbre(self, signal):  #signal is tensor on gpu
        payload = self.model.test_forward(signal) # signal: [1,1,80000]
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bitacc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return bitacc


def get_scale_factor(signal, noise, required_SNR):
    snr = 10*torch.log10(torch.mean(signal**2)/torch.mean(noise**2))
    if snr < required_SNR:
        scale_factor = 10 ** ((required_SNR - snr) / 10)
    else: 
        scale_factor = 1
    return scale_factor


def whitebox_attack(detector, watermarked_signal, args):
    start_time = time.time()
    bwacc = detector.bwacc(watermarked_signal)
    best_bwacc = bwacc
    best_adv_signal = watermarked_signal
    # Initialize tensor_pert
    tensor_pert = torch.zeros_like(watermarked_signal, requires_grad=True)
    # Freeze detector and watermarked_signal
    watermarked_signal.requires_grad = False
    # Define optimizer
    optimizer = optim.Adam([tensor_pert], lr=args.lr)
    # Projected Gradient Descent
    for _ in range(args.iter):
        detector.model.train()
        optimizer.zero_grad()
        watermarked_signal_with_noise = watermarked_signal + tensor_pert 
        loss = detector.get_loss(watermarked_signal_with_noise)
        # bwacc = detector.bwacc(watermarked_signal_with_noise)
        # snr = 10*torch.log10(torch.mean(watermarked_signal**2)/torch.mean(tensor_pert**2))
        # print(f'best BWACC: {best_bwacc:.3f}, BWACC: {bwacc:.3f}, SNR: {snr:.1f}')
        # Backpropagation
        loss.backward()
        optimizer.step()
        # tensor_pert.data = torch.clamp(tensor_pert.data, -args.pert_boundary, args.pert_boundary)
        scale_factor = get_scale_factor(watermarked_signal, tensor_pert, args.rescale_snr)
        if scale_factor > 1:
            tensor_pert.data /= scale_factor
        # Evaluation
        detector.model.eval()
        with torch.no_grad():
            watermarked_signal_with_noise = watermarked_signal + tensor_pert 
            bwacc = detector.bwacc(watermarked_signal_with_noise)
            snr = 10*torch.log10(torch.mean(watermarked_signal**2)/torch.mean(tensor_pert**2))
            if bwacc < best_bwacc:
                best_bwacc = bwacc
                best_adv_signal = watermarked_signal_with_noise
            if best_bwacc <= args.tau:
                break
    if best_bwacc > args.tau:
        print(f'Attack failed, the best bwacc is {best_bwacc}')
    print(f'Attack time: {time.time() - start_time}')
    return best_adv_signal


def decode_audio_files_perturb_whitebox(model, output_dir, args, device):
    if args.dataset == 'audiomark':
        watermarked_files = os.listdir(os.path.join(output_dir, 'watermarked_200'))
    else:
        watermarked_files = [f for f in os.listdir(os.path.join(output_dir,'watermarked'))]
        watermarked_files = sorted(watermarked_files)[:args.testset_size]
    progress_bar = tqdm(enumerate(watermarked_files), desc="Decoding Watermarks under whitebox attack")
    save_path = os.path.join(output_dir, args.whitebox_folder)
    os.makedirs(save_path, exist_ok=True)   
    visqol = api_visqol()
    for file_num, watermarked_file in progress_bar:
        idx = '_'.join(watermarked_file.split('_')[:-2]) # idx_bitstring_snr
        waveform, sample_rate = torchaudio.load(os.path.join(output_dir, 'watermarked', watermarked_file))

        '''waveform.shape = [1, 80000]'''
        waveform = waveform.to(device=device)
        waveform = waveform.unsqueeze(0)

        original_payload_str = watermarked_file.split('_')[-2]
        original_payload = torch.tensor(list(map(int, original_payload_str)), dtype=torch.float32, device=device)

        detector = WatermarkDetectorWrapper(model, original_payload, args.attack_bitstring, args.model, args.tau, device)
        adv_signal = whitebox_attack(detector, waveform, args)

        '''save to log file'''
        filename=os.path.join(save_path, f'whitebox.csv')
        log = open(filename, 'a' if os.path.exists(filename) else 'w')
        log.write('idx, query, acc, snr, visqol\n')
        acc = detector.bwacc(adv_signal)
        snr = 10*torch.log10(torch.sum(waveform**2)/torch.sum((adv_signal - waveform)**2))
        visqol_score = visqol.Measure(np.array(waveform.squeeze().detach().cpu(), dtype=np.float64), np.array(adv_signal.squeeze().detach().cpu(), dtype=np.float64)).moslqo
        print(f'idx: {idx}, query: {args.iter}, acc: {acc:.3f}, snr: {snr:.1f}, visqol: {visqol_score:.3f}')
        log.write(f'{idx}, {args.iter}, {acc}, {snr}, {visqol_score}\n')
        if file_num % 5 == 0:
            torchaudio.save(os.path.join(save_path, 
                f"{idx}_tau{args.tau}_query{args.iter}_snr{snr:.1f}_acc{acc:.1f}_visqol{visqol_score:.1f}.wav"),
                adv_signal.squeeze(0).detach().cpu(), sample_rate)
             
def main():
    args = parse_arguments()

    np.random.seed(42)
    torch.manual_seed(42)

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    if args.model == 'audioseal':
        model = AudioSeal.load_detector("audioseal_detector_16bits").to(device=device)
        output_dir = 'audiomark_audioseal_max_5s'if args.dataset == 'audiomark' else 'LibriSpeech_audioseal_max_5s'
        os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'wavmark':
        model = wavmark.load_model().to(device)
        output_dir = 'audiomark_wavmark_max_5s' if args.dataset == 'audiomark' else 'LibriSpeech_wavmark_max_5s'
        os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'timbre':
        process_config = yaml.load(open("timbre/config/process.yaml", "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open("timbre/config/model.yaml", "r"), Loader=yaml.FullLoader)
        win_dim = process_config["audio"]["win_len"]
        embedding_dim = model_config["dim"]["embedding"]
        nlayers_decoder = model_config["layer"]["nlayers_decoder"]
        attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
        msg_length = 30
        detector = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
        checkpoint = torch.load('/home/mg585/audio_watermark/timbre/results/ckpt/pth/compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar')
        detector.load_state_dict(checkpoint['decoder'], strict=False)
        detector.eval()
        model = detector
        output_dir = 'audiomark_timbre_max_5s' if args.dataset == 'audiomark' else 'LibriSpeech_timbre_max_5s'
        os.makedirs(output_dir, exist_ok=True)

    decode_audio_files_perturb_whitebox(model, output_dir, args, device)
if __name__ == "__main__":
    main()