import os
import argparse
import numpy as np
import torchaudio
from tqdm import tqdm

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=str, required=True)
    parser.add_argument("--perturb", type=str, required=True)
    args = parser.parse_args()
    api = api_visqol()
    scores = dict()
    for file in tqdm(os.listdir(args.perturb), desc="Calculating Visqol"):
        if file.endswith(".wav"):
            waveform = torchaudio.load(os.path.join(args.origin, file))[0]
            perturb_waveform = torchaudio.load(os.path.join(args.perturb, file))[0]
            waveform = np.array(waveform.squeeze(), dtype=np.float64)
            perturb_waveform = np.array(perturb_waveform.squeeze(), dtype=np.float64)
            score = api.Measure(waveform, perturb_waveform)
            scores[file] = score.moslqo
    socres = np.array(scores, dtype=object)
    np.save(os.path.join(args.perturb, "scores.npy"), scores)

