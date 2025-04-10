import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import SynthesizerTrn

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

def getText(text, hps):
    textNorm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        textNorm = commons.intersperse(textNorm, 0)
    textNorm = torch.LongTensor(textNorm).unsqueeze(0)
    return textNorm 

hps = utils.get_hparams_from_file("configs/ljs_base.json")

generatorNet = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,    
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
).cuda()

_ = utils.load_checkpoint("./logs/ljs_base/G_141000.pth", generatorNet, None)

with open("transcripts.txt", "r+") as textFile:
    i = 0
    for line in textFile.readlines():
        if line != "":
            i += 1
            text = getText(line.strip(), hps).cuda().unsqueeze(0)
            textLengths = torch.LongTensor([text.size(0)]).cuda()
            audio = generatorNet.infer(text, textLengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)[0][0, 0].data.cpu().float().numpy()
            print(audio)
