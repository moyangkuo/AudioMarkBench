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
import torchaudio.transforms as T
import time
from art.estimators.classification import PyTorchClassifier


def parse_arguments():
    parser = argparse.ArgumentParser(description="Audio Watermarking with WavMark")
    parser.add_argument("--testset_size", type=int, default=200, help="Number of samples from the test set to process")
    parser.add_argument("--encode", action="store_true", help="Run the encoding process before decoding")
    parser.add_argument("--length", type=int, default=5*16000, help="Length of the audio samples")

    parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")
    parser.add_argument("--save_pert", action="store_true", help="If set, saves the perturbed waveform")
    
    parser.add_argument("--query_budget", type=int, default=10000, help="Query budget for the attack")
    parser.add_argument("--blackbox_folder", type=str, default="blackbox_square", help="Folder to save the blackbox attack results")

    parser.add_argument("--eps", type=float, default=0, help="Epsilon for the attack")
    parser.add_argument("--p", type=float, default=0.05, help="probability")
    
    parser.add_argument("--tau", type=float, default=0, help="Threshold for the detector")
    parser.add_argument("--snr", type=list, default=[0,2.5, 5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30], help="Signal-to-noise ratio for the attack")
    parser.add_argument("--norm", type=str, default='linf', help="Norm for the attack")
    parser.add_argument("--attack_bitstring", action="store_true", help="If set, perturb the bitstring instead of the detection probability")
    parser.add_argument("--attack_type", type=str, default='both', choices=['amplitude','phase','both'], help="Type of attack")
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

np.set_printoptions(precision=5, suppress=True)
def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p


def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
              max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def square_attack_linf(model, x, eps, n_iters, p_init, args):
    """ The Linf square attack """
    np.random.seed(0)  # important to leave it here as well
    # min_val, max_val = 0, 1 if x.max() <= 1 else 255
    min_val, max_val = x.min(), x.max()

    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]
    # x, y = x[corr_classified], y[corr_classified]

    # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    x_best = np.clip(x + init_delta, min_val, max_val)

    # logits = model.predict(x_best)
    # loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    # margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    loss = model.get_detection_result(x_best)
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    progress_bar = tqdm(range(n_iters), desc='Linf square attack')
    for i_iter in progress_bar:
        idx_to_fool = (loss >= 0)
        # x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        # loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        loss_min_curr = loss[idx_to_fool]

        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])

        x_new = np.clip(x_curr + deltas, min_val, max_val)

        # logits = model.predict(x_new)
        # loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        # margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')
        loss = model.get_detection_result(x_new)

        idx_improved = loss < loss_min_curr
        # loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        # margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        # acc = (margin_min > 0.0).sum() / n_ex_total
        # acc_corr = (margin_min > 0.0).mean()
        # mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(n_queries[margin_min <= 0])
        # avg_margin_min = np.mean(margin_min)
        best_loss = np.minimum(loss, loss_min_curr)
        acc = best_loss.mean()
        time_total = time.time() - time_start
        curr_norms_image = np.max(np.abs(x_new - x))
        curr_norms_image_best = np.max(np.abs(x_best - x))
        progress_bar.set_description(f'iter: {i_iter+1}, acc: {acc:.2f}, time: {time_total:.1f}, max_pert: {curr_norms_image:.2f}, max_pert_best: {curr_norms_image_best:.2f}')
        # print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
        #     format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], eps, time_total))

        if acc <= args.tau:
            break

    return n_queries, x_best


def square_attack_l2(model, x, eps, n_iters, p_init, args):
    

    """ The L2 square attack """
    np.random.seed(0)

    min_val, max_val = x.min(), x.max()
    c, h, w = x.shape[1:]
    
    n_features = c * h * w

    ### initialization
    delta_init = np.zeros(x.shape)
    s = h // 5
    print('Initial square side={} for bumps'.format(s))
    sp_init = (h - s * 5) // 2
    center_h = sp_init + 0
    for counter in range(h // s):
        center_w = sp_init + 0
        for counter2 in range(w // s):
            delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += meta_pseudo_gaussian_pert(s).reshape(
                [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
            center_w += s
        center_h += s

    x_best = np.clip(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * eps, min_val, max_val)

    loss = model.get_detection_result(x_best)

    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    s_init = int(np.sqrt(p_init * n_features / c))
    progress_bar = tqdm(range(n_iters))
    for i_iter in progress_bar:
        idx_to_fool = (loss >= 0.0)

        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        loss_min_curr = loss[idx_to_fool]
        delta_curr = x_best_curr - x_curr
        p = p_selection(p_init, i_iter, n_iters)
        s = max(int(round(np.sqrt(p * n_features / c))), 3)

        if s % 2 == 0:
            s += 1

        s2 = s + 0
        ### window_1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = np.zeros(x_curr.shape)
        new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

        ### window_2
        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = np.zeros(x_curr.shape)
        new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0
        # norms_window_2 = np.sqrt(
        #     np.sum(delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] ** 2, axis=(-2, -1),
        #            keepdims=True))

        ### compute total norm available
        curr_norms_window = np.sqrt(
            np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
        curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
        mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = np.sqrt(np.sum((delta_curr * mask_2) ** 2, axis=(2, 3), keepdims=True))

        ### create the updates
        new_deltas = np.ones([x_curr.shape[0], c, s, s])
        new_deltas = new_deltas * meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
        new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
        old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
        new_deltas += old_deltas
        new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
            np.maximum(eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
        delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
        delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

        hps_str = 's={}->{}'.format(s_init, s)
        x_new = x_curr + delta_curr / np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True)) * eps
        x_new = np.clip(x_new, min_val, max_val)

        loss = model.get_detection_result(x_new)

        idx_improved = loss < loss_min_curr

        idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        best_loss = np.minimum(loss, loss_min_curr)
        acc = best_loss.mean()
        time_total = time.time() - time_start
        
        curr_norms_image = np.sqrt(np.sum((x_new - x) ** 2))
        curr_norms_image_best = np.sqrt(np.sum((x_best - x) ** 2))
        progress_bar.set_description(f'iter: {i_iter+1}, acc: {acc:.2f}, time: {time_total:.1f}, max_pert: {curr_norms_image:.2f}, max_pert_best: {curr_norms_image_best:.2f}')

        if acc <= args.tau:
            curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
            print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))
            break

    curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
    print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))

    return n_queries, x_best


class WatermarkDetectorWrapper(PyTorchClassifier):
    def __init__(self, model, message, detector_type, on_bitstring, transform, th, input_size, model_type, device):
        super(WatermarkDetectorWrapper, self).__init__(model=model,
        input_shape=input_size, nb_classes=2, channels_first=True,loss=None)
        self._device = device
        self.message = message.to(self._device)
        self.detector_type = detector_type
        self.th = th
        self.on_bitstring = on_bitstring
        self.transform = transform
        self.model.to(self._device)
        if model_type == 'timbre':
            self.get_detection_result = self.get_detection_result_timbre
            self.bwacc = self.bwacc_timbre
        elif model_type == 'wavmark':
            self.get_detection_result = self.get_detection_result_wavmark
            self.bwacc = self.bwacc_wavmark
        elif model_type == 'audioseal':
            self.get_detection_result = self.get_detection_result_audioseal
            self.bwacc = self.bwacc_audioseal

    def get_detection_result_audioseal(self, spectrogram):
        spectrogram = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram)
        result, msg_decoded = self.model.detect_watermark(signal.unsqueeze(0))
        if self.on_bitstring:
            if msg_decoded is None:
                return np.array([0])
            else: 
                bitacc = 1-torch.sum(torch.abs(self.message - msg_decoded)) / len(self.message)
                return np.array([bitacc.item()])
        else:
            return np.array([result])

    def bwacc_audioseal(self, signal):
        result, msg_decoded = self.model.detect_watermark(signal.unsqueeze(0))
        if self.on_bitstring:
            if msg_decoded is None:
                return np.array([0])
            else: 
                bitacc = 1-torch.sum(torch.abs(self.message - msg_decoded)) / len(self.message)
                return np.array([bitacc.item()])
        else:
            return np.array([result])

    def get_detection_result_wavmark(self, spectrogram):
        spectrogram = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram).squeeze(0).detach().cpu()
        payload, info = wavmark.decode_watermark(self.model, signal)
        if payload is None:
            return np.array([0])
        else: 
            payload = torch.tensor(payload).to(self.message.device)
            bit_acc = 1-torch.sum(torch.abs(payload-self.message))/self.message.shape[0]
            return np.array([bit_acc.item()])
        
    def bwacc_wavmark(self, signal):
        signal = signal.squeeze(0).detach().cpu()
        payload, info = wavmark.decode_watermark(self.model, signal)
        if payload is None:
            return np.array([0])
        else: 
            payload = torch.tensor(payload).to(self.message.device)
            bit_acc = 1-torch.sum(torch.abs(payload-self.message))/self.message.shape[0]
            return np.array([bit_acc.item()])

    def get_detection_result_timbre(self, spectrogram, batch_size=1): # signal is np.array
        spectrogram = torch.tensor(spectrogram).to(device=self._device)
        signal = self.transform.spectrogram2signal(spectrogram)
        payload = self.model.test_forward(signal.unsqueeze(0)) # signal: [1,1,80000]
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bitacc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return np.array([bitacc.item()])

    def bwacc_timbre(self, signal):  #signal is tensor on gpu
        payload = self.model.test_forward(signal.unsqueeze(0)) # signal: [1,1,80000]
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bit_acc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return np.array([bit_acc.item()])
    

class signal22spectrogram:
    def __init__(self, signal, low_frequency, high_frequency, device, attack_type):
        self.signal = signal
        self.attack_type = attack_type
        self.sig2spec = T.Spectrogram(n_fft=400, power=None).to(device)
        self.spec2sig = T.InverseSpectrogram(n_fft=400).to(device)
        self.spectrogram = self.sig2spec(signal)
        self.amplitude = torch.abs(self.spectrogram)
        self.phase = torch.angle(self.spectrogram)
        self.lf = low_frequency
        self.hf = high_frequency
        self.attack_shape = self.spectrogram[...,low_frequency:high_frequency,:].shape
        self.length = signal.shape[-1]
        '''attack boundary'''
    def signal2spectrogram(self, signal): # [1,length]->[2 or 1,freq,length]
        spectro_complex = self.sig2spec(signal)
        spectro_amplitude = torch.abs(spectro_complex)
        spectro_phase = torch.angle(spectro_complex)
        if self.attack_type == 'amplitude':
            return spectro_amplitude[...,self.lf:self.hf,:]
        elif self.attack_type == 'phase':
            return spectro_phase[...,self.lf:self.hf,:]
        elif self.attack_type == 'both':
            return torch.cat([spectro_amplitude, spectro_phase], dim=0)[...,self.lf:self.hf,:]

    def spectrogram2signal(self, spectrogram): # [2,freq,length]->[1,length]
        spectrogram = spectrogram.squeeze()
        if self.attack_type == 'both':
            padding_amp = self.amplitude
            padding_amp[...,self.lf:self.hf,:] = spectrogram[0]
            padding_phase = self.phase
            padding_phase[...,self.lf:self.hf,:] = spectrogram[1]
            spectro_complex = padding_amp * torch.exp(1j * padding_phase)
        elif self.attack_type == 'amplitude':
            padding_amp = self.amplitude
            padding_amp[...,self.lf:self.hf,:] = spectrogram
            spectro_complex = padding_amp * torch.exp(1j * self.phase)
        elif self.attack_type == 'phase':
            padding_phase = self.phase
            padding_phase[...,self.lf:self.hf,:] = spectrogram
            spectro_complex = self.amplitude * torch.exp(1j * padding_phase)
        signal = self.spec2sig(spectro_complex, self.length)
        return signal
    

def decode_audio_files_perturb_blackbox(model, output_dir, args, device):
    attack = square_attack_l2 if args.norm == 2 else square_attack_linf
    watermarked_files = os.listdir(os.path.join(output_dir, 'watermarked_200')) # NOTE: this is the subsampled dataset of AudioMarkBench
    progress_bar = tqdm(watermarked_files, desc="Decoding Watermarks under blackbox attack")
    save_path = os.path.join(output_dir, args.blackbox_folder)
    os.makedirs(save_path, exist_ok=True)   
    visqol = api_visqol()
    for watermarked_file in progress_bar:
        parts = watermarked_file.split('_')
        idx = '_'.join(watermarked_file.split('_')[:-2]) # idx_bitstring_snr
        waveform, sample_rate = torchaudio.load(os.path.join(output_dir, 'watermarked', watermarked_file))

        '''waveform.shape = [1, 80000]'''
        waveform = waveform.to(device=device)
        original_payload_str = watermarked_file.split('_')[-2]
        original_payload = torch.tensor(list(map(int, original_payload_str)), dtype=torch.int)
        transform = signal22spectrogram(waveform, 0, 201, device,args.attack_type)
        detector = WatermarkDetectorWrapper(model, original_payload, 'single-tailed', args.attack_bitstring, 
            transform, args.tau, model_type=args.model, input_size = transform.attack_shape, device=device)
        
        '''save to log file'''
        filename=os.path.join(save_path, f'square_spectrogram.csv')
        log = open(filename, 'a' if os.path.exists(filename) else 'w')
        log.write('idx, query, acc, snr, visqol\n')
        
        '''apply the attack to the watermarked signal and save the perturbed signal to the save_path'''
        watermarked_spectrogram = transform.signal2spectrogram(waveform).unsqueeze(0).detach().cpu().numpy()
        n_queries, adv_spectrogram = attack(detector, watermarked_spectrogram, args.eps, args.query_budget, args.p, args)
        adv_signal = transform.spectrogram2signal(torch.tensor(adv_spectrogram).to(device))
        acc = detector.bwacc(adv_signal).item()
        snr = 10*torch.log10(torch.sum(waveform**2)/torch.sum((adv_signal - waveform)**2))
        visqol_score = visqol.Measure(np.array(waveform.squeeze().detach().cpu(), dtype=np.float64), np.array(adv_signal.squeeze().detach().cpu(), dtype=np.float64)).moslqo
        print(f'idx: {idx}, query: {n_queries.item()}, acc: {acc:.3f}, snr: {snr:.1f}, visqol: {visqol_score:.3f}')
        log.write(f'{idx}, {n_queries.item()}, {acc}, {snr}, {visqol_score}\n')
        torchaudio.save(os.path.join(save_path, 
            f"{idx}_tau{args.tau}_query{n_queries.item()}_snr{snr:.1f}_acc{acc:.1f}_visqol{visqol_score:.1f}.wav"),
            adv_signal.detach().cpu(), sample_rate)

                    

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
        output_dir = 'audiomark_audioseal_max_5s'
        os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'wavmark':
        model = wavmark.load_model().to(device)
        output_dir = 'audiomark_wavmark_max_5s'
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
        output_dir = 'audiomark_timbre_max_5s'
        os.makedirs(output_dir, exist_ok=True)

    decode_audio_files_perturb_blackbox(model, output_dir, args, device)
if __name__ == "__main__":
    main()