import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import os

import audio
from glow import WaveGlow
import hparams as hp
from text import text_to_sequence
from data_utils import FastSpeechDataset, collate_fn, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_data(data, figsize=(12, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')
    plt.savefig(os.path.join("img", "model_test.jpg"))


def get_waveglow():
    waveglow_path = os.path.join('waveglow_256channels.pt')
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().half()
    for k in waveglow.convinv:
        k.float()

    return waveglow



def synthesis_waveglow(mel, waveglow, num, alpha=1.0):
    wav = waveglow.infer(mel, sigma=0.666)
    print("Wav Have Been Synthesized.")

    if not os.path.exists("results"):
        os.mkdir("results")
    audio.save_wav(wav[0].data.cpu().numpy(), os.path.join(
        "results", str(num) + ".wav"))


if __name__ == "__main__":
    # Test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed(hp.seed)
    model = WaveGlow().cuda()
    checkpoint = torch.load('outdir/TTSglow_200000')
    model.load_state_dict(checkpoint['model'].state_dict())

    dataset = FastSpeechDataset()
    testing_loader = DataLoader(dataset, 
                                batch_size = 1,
                                shuffle=False,
                                collate_fn=collate_fn,
                                drop_last=True,
                                num_workers=4)
    model = model.eval()

    for i, data_of_batch in enumerate(testing_loader):
        src_seq = data_of_batch["texts"]
        src_pos = data_of_batch["pos"]

        src_seq = torch.from_numpy(src_seq).long().to(device)
        src_pos = torch.from_numpy(src_pos).long().to(device)

        mel = model.inference(src_seq, src_pos, sigma=1.0, alpha=1.0)
        mel_path = os.path.join(
            "results", "{}_synthesis.pt".format(i))
        torch.save(mel, mel_path)
        print(mel_path)



        ''' glow = get_waveglow()
        synthesis_waveglow(mel, glow, i, alpha=1.0)
        print("Synthesized by Waveglow.")'''
        
