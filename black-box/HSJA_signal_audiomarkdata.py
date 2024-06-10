import os
import argparse
import numpy as np
import torch
import torchaudio
from audioseal import AudioSeal
import wavmark
import yaml
from timbre.model.conv2_mel_modules import Decoder
from tqdm import tqdm
import fnmatch
import torch.nn.functional as F
from art.estimators.classification import PyTorchClassifier
from hop_skip_jump import HopSkipJump


def parse_arguments():
    parser = argparse.ArgumentParser(description="Audio Watermarking with WavMark")
    parser.add_argument("--testset_size", type=int, default=200, help="Number of samples from the test set to process")
    parser.add_argument("--encode", action="store_true", help="Run the encoding process before decoding")
    # parser.add_argument("--encode", default=True, help="Run the encoding process before decoding")

    parser.add_argument("--length", type=int, default=5*16000, help="Length of the audio samples")

    parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")
    
    parser.add_argument("--query_budget", type=int, default=10000, help="Query budget for the attack")
    parser.add_argument("--blackbox_folder", type=str, default="HSJ_signal", help="Folder to save the blackbox attack results")
   
    parser.add_argument("--max_iter", type=int, default=1, help="Maximum number of iterations for the attack")
    parser.add_argument("--max_eval", type=int, default=1000, help="Maximum number of evaluations for estimating gradient")
    parser.add_argument("--init_eval", type=int, default=100, help="Initial number of evaluations for estimating gradient")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the attack")
    parser.add_argument("--tau", type=float, default=0, help="Threshold for the detector")
    parser.add_argument("--norm", type=str, default='linf', help="Norm for the attack")
    parser.add_argument("--attack_bitstring", action="store_true", help="If set, perturb the bitstring instead of the detection probability")
    parser.add_argument("--model", type=str, default='', choices=['audioseal','wavmark', 'timbre'], help="Model to be attacked")

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


class logger():
    def __init__(self, filename, raw, idx):
        self.visqol = api_visqol()
        self.raw = raw
        self.best_l2 = 1000
        self.best_linf = 1000
        self.best_acc = 1
        self.best_snr = -1000
        self.best_visqol = -1
        self.idx = idx
        file_exists = os.path.exists(filename)
        self.log = open(filename, 'a' if file_exists else 'w')
        if not file_exists:
            # Write the title only if the file does not exist
            self.log.write('idx, query, l2, linf, acc, snr, visqol, best_l2, best_linf, best_acc, best_snr, best visqol\n')

    def evaluate(self, signal, query, acc):
        if_better = False
        l2 = np.linalg.norm(signal - self.raw)
        linf = np.max(np.abs(signal - self.raw))
        snr = 10 * np.log10(np.sum(np.square(self.raw))/np.sum(np.square(signal - self.raw)))
        visqol_score = self.visqol.Measure(np.array(self.raw.squeeze(), dtype=np.float64), np.array(signal.squeeze(), dtype=np.float64)).moslqo
        if l2 < self.best_l2:
            self.best_l2 = l2
        if linf < self.best_linf:
            self.best_linf = linf
        if acc < self.best_acc:
            self.best_acc = acc
        if visqol_score > self.best_visqol:
            self.best_visqol = visqol_score 
            if_better = True
        if snr >= self.best_snr:
            self.best_snr = snr
        self.log.write(f'{self.idx}, {query}, {l2}, {linf}, {acc}, {snr}, {visqol_score}, {self.best_l2}, {self.best_linf}, {self.best_acc}, {self.best_snr}, {self.best_visqol}\n')
        return if_better
    
    
# Define a wrapper for the watermark detector to fit into the HopSkipJump interface
class WatermarkDetectorWrapper(PyTorchClassifier):
    def __init__(self, model, message, detector_type, on_bitstring, th, model_type, device):
        super(WatermarkDetectorWrapper, self).__init__(model=model,
        input_shape=(1,), nb_classes=2, channels_first=True,loss=None)
        self._device = device
        self.message = message.to(self._device)
        self.detector_type = detector_type
        self.th = th
        self.on_bitstring = on_bitstring
        self.model.to(self._device)
        if model_type == 'timbre':
            self.predict = self.predict_timbre
            self.bwacc = self.bwacc_timbre
        elif model_type == 'wavmark':
            self.predict = self.predict_wavmark
            self.bwacc = self.bwacc_wavmark
        elif model_type == 'audioseal':
            self.predict = self.predict_audioseal
            self.bwacc = self.bwacc_audioseal

    def predict_audioseal(self, signal, batch_size=1):# signal is np.array
        signal = torch.tensor(signal, dtype=torch.float).to(self._device)
        result, msg_decoded = self.model.detect_watermark(signal.unsqueeze(0))
        if self.on_bitstring:
            return self.our_conversion_logic(msg_decoded)
        else:
            return self.our_conversion_logic_binary(result)

    def bwacc_audioseal(self, signal): #signal is tensor on gpu
        result, msg_decoded = self.model.detect_watermark(signal.unsqueeze(0))
        if self.on_bitstring:
            if msg_decoded is None:
                return 0
            else: 
                msg_decoded = torch.tensor(msg_decoded, dtype=torch.int).to(self.message.device)
                bit_acc = 1-torch.sum(torch.abs(msg_decoded-self.message))/self.message.shape[0]
                return bit_acc.item()
        else:
            return result
        
    def predict_wavmark(self, signal, batch_size=1): # signal is np.array
        signal = torch.tensor(signal, dtype=torch.float).squeeze(0)
        payload, info = wavmark.decode_watermark(self.model, signal) # signal: [,80000]
        return self.our_conversion_logic(payload)

    def bwacc_wavmark(self, signal):  #signal is tensor on gpu
        signal = signal.squeeze(0).detach().cpu()
        payload, info = wavmark.decode_watermark(self.model, signal) # signal: [,80000]
        if payload is None:
            return 0
        else: 
            payload = torch.tensor(payload).to(self.message.device)
            bit_acc = 1-torch.sum(torch.abs(payload-self.message))/self.message.shape[0]
            return bit_acc.item()
        
    def predict_timbre(self, signal, batch_size=1): # signal is np.array
        signal = torch.tensor(signal, dtype=torch.float).to(self._device)
        payload = self.model.test_forward(signal.unsqueeze(0)) # signal: [1,80000]
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bit_acc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        if self.detector_type=='double-tailed':
            class_idx = torch.logical_or((bit_acc>=self.th), (bit_acc<=(1-self.th)))
        if self.detector_type=='single-tailed':
            class_idx = (bit_acc>=self.th)
        return np.array([[0,1]]) if class_idx else np.array([[1,0]])

    def bwacc_timbre(self, signal):  #signal is tensor on gpu
        payload = self.model.test_forward(signal.unsqueeze(0)) # signal: [1,1,80000]
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bit_acc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return bit_acc

          
    def our_conversion_logic(self, payload):
        if payload is None:
            return np.array([[1,0]])# NOTE if result is None, then it is classified as 0, not detected
        else: 
            payload = torch.tensor(payload).to(self.message.device)
            bit_acc = 1-torch.sum(torch.abs(payload-self.message))/self.message.shape[0]
            if self.detector_type=='double-tailed':
                class_idx = torch.logical_or((bit_acc>=self.th), (bit_acc<=(1-self.th))).long()
            if self.detector_type=='single-tailed':
                class_idx = (bit_acc>=self.th).long()
            return np.array([[0,1]]) if class_idx else np.array([[1,0]])

    def our_conversion_logic_binary(self, bit_acc):
        if bit_acc is None:
            return np.array([[1,0]])# NOTE if result is None, then it is classified as 0, not detected
        else: 
            if self.detector_type=='double-tailed':
                class_idx = torch.logical_or((bit_acc>=self.th), (bit_acc<=(1-self.th)))
            if self.detector_type=='single-tailed':
                class_idx = (bit_acc>=self.th)
            return np.array([[0,1]]) if class_idx else np.array([[1,0]])


def initial_ad_samples(model, signal, query, tau):
    signal_power = torch.mean(signal ** 2)
    for noise_level in [30,25,20,17.5,15,12.5,10,7.5,5,2.5,0]:
        noise_power = signal_power / (10 ** (noise_level / 10))
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(signal) * noise_std
        adv_signal = signal + noise
        query = query + 1
        acc = model.bwacc(adv_signal)
        if acc <= tau:
            break
    snr = 10 * torch.log10(torch.mean(signal ** 2) / torch.mean(noise ** 2))
    print(f'detection probability: {acc}, snr: {snr:.3f}')
    return adv_signal, query



def decode_audio_files_perturb_blackbox(model, output_dir, args, device):
    watermarked_files = os.listdir(os.path.join(output_dir, 'watermarked_200')) # NOTE: this is the subsampled dataset of audiomarkdataBench
    progress_bar = tqdm(watermarked_files, desc="Decoding Watermarks under blackbox attack")
    save_path = os.path.join(output_dir, args.blackbox_folder)
    os.makedirs(save_path, exist_ok=True)   
    for watermarked_file in progress_bar:
        idx = '_'.join(watermarked_file.split('_')[:-2]) # idx_bitstring_snr
        waveform, sample_rate = torchaudio.load(os.path.join(output_dir, 'watermarked', watermarked_file))
        '''waveform.shape = [1, 80000]'''
        waveform = waveform.to(device=next(model.parameters()).device)
        original_payload_str = watermarked_file.split('_')[-2]
        original_payload = torch.tensor(list(map(int, original_payload_str)), dtype=torch.int)
        detector = WatermarkDetectorWrapper(model, original_payload, 'single-tailed', args.attack_bitstring, args.tau, args.model, device)
        adv_signal, num_queries = initial_ad_samples(detector, waveform, np.zeros((1)), args.tau)
        # Initialize the HopSkipJump attack; 
        attack = HopSkipJump(classifier=detector, targeted=False, norm=args.norm, max_iter = args.max_iter,
                        max_eval=args.max_eval, init_eval=args.init_eval, batch_size=args.batch_size)
        waveform = waveform.detach().cpu().numpy()
        adv_signal = adv_signal.detach().cpu().numpy()
        log = logger(filename=os.path.join(save_path, f'{args.model}_tau{args.tau}.csv'), raw=waveform, idx=idx)
        while num_queries <= args.query_budget and num_queries >= 0:
            adv_signal, num_queries = attack.generate(x=waveform, x_adv_init = adv_signal, num_queries_ls = num_queries, resume=True)
            print(f'num_queries: {num_queries}')
            acc = detector.bwacc(torch.tensor(adv_signal, dtype=torch.float).to(device))
            if log.evaluate(signal=adv_signal, query=num_queries[0], acc=acc):
                print(f'idx: {idx}, query: {num_queries[0]}, acc: {acc:.3f}, snr: {log.best_snr:.1f}, visqol: {log.best_visqol:.3f}')
        torchaudio.save(os.path.join(save_path, 
        f"{idx}_tau{args.tau}_query{num_queries[0]}_snr{log.best_snr:.1f}_visqol{log.best_visqol:.2f}_acc{acc:.3f}.wav"),
        torch.tensor(adv_signal), sample_rate)


def main():
    args = parse_arguments()

    if args.norm == 'l2':
        args.norm = 2
    else:
        args.norm = np.inf
    
    np.random.seed(42)
    torch.manual_seed(42)

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')


    if args.model == 'audioseal':
        model = AudioSeal.load_detector("audioseal_detector_16bits").to(device=device)
        output_dir = 'audiomarkdata_audioseal_max_5s'
        os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'wavmark':
        model = wavmark.load_model().to(device)
        output_dir = 'audiomarkdata_wavmark_max_5s'
        os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'timbre':
        process_config = yaml.load(open("timbre/config/process.yaml", "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open("timbre/config/model.yaml", "r"), Loader=yaml.FullLoader)
        win_dim = process_config["audio"]["win_len"]
        embedding_dim = model_config["dim"]["embedding"]
        nlayers_decoder = model_config["layer"]["nlayers_decoder"]
        attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
        detector = Decoder(process_config, model_config, 30, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
        checkpoint = torch.load('timbre/results/ckpt/pth/compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar')
        detector.load_state_dict(checkpoint['decoder'], strict=False)
        detector.eval()
        model = detector
        output_dir = 'audiomarkdata_timbre_max_5s'
        os.makedirs(output_dir, exist_ok=True)

    decode_audio_files_perturb_blackbox(model, output_dir, args, device)

if __name__ == "__main__":
    main()
