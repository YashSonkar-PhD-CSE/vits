import warnings
from numba.core.errors import NumbaWarning

warnings.simplefilter('ignore', category=NumbaWarning)
# import IPython.display as ipd
import torch
from torch.utils.data import DataLoader
import os
import commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from mel_processing import spectrogram_torch
import torchaudio


hps = utils.get_hparams_from_file("/ssd_scratch/cvit/yash/vits/configs/wtimit_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers = hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

ckptFile = os.listdir("/ssd_scratch/cvit/yash/vits/logs/wt_custom")
ckptFile = [s for s in ckptFile if s.startswith("G_") and s.endswith(".pth")]
ckptFile = sorted(ckptFile)
print("Using checkpoint", ckptFile)
_ = utils.load_checkpoint(f"/ssd_scratch/cvit/yash/vits/logs/wt_custom/{ckptFile[-1]}", net_g, None)

# For every audio provided, get the spectrogram and sid

audios = {}
with open("/ssd_scratch/cvit/yash/vits/convertData.txt", "r+") as dataFile:
    for line in dataFile:
        if not line or line.strip() == "":
            continue
        parts = line.split("|")
        path = parts[0]
        src_sid = parts[1]
        tgt_sid = parts[2]
        
        audios[path] = (src_sid, tgt_sid)
i = 0
l = len(audios)    
for audioPath, (src_sid, tgt_sid) in audios.items():
    print(f"Progress: {i}/{l} ({i / l * 100 :.2f}%)", end="\r")
    filename = audioPath.split("/")[-1]
    #audio, sr = utils.load_wav_to_torch(audioPath)
    #audioNorm = audio / 1.33 # 1.33 = Max Wav value
    #audioNorm = audioNorm.unsqueeze(0)
    #spec = spectrogram_torch(
    #    audioNorm,
    #    hps.data.filter_length,
    #    hps.data.sampling_rate,
    #    hps.data.hop_length,
    #    hps.data.win_length,
    #    center=False
    #)
    try:
        spec = torch.load(audioPath.replace(".wav", ".spec.pt")).unsqueeze(0)
    except:
        print(f"Skipping file {filename}", flush=True)
        continue
    specLen = torch.LongTensor([spec.size(1)])
    sid1 = torch.LongTensor([int(src_sid)])
    sid2 = torch.LongTensor([int(tgt_sid)])
    torch.cuda.empty_cache()
    convertedAudio = net_g.voice_conversion(
        spec.cuda(non_blocking=True), 
        specLen.cuda(non_blocking=True),
        sid_src = sid1.cuda(non_blocking=True),
        sid_tgt = sid2.cuda(non_blocking=True),
    )[0][0, 0].data.cpu().float()
    torchaudio.save(
        f"/ssd_scratch/cvit/yash/vits/converted/{filename}",
        convertedAudio.unsqueeze(0),
        sample_rate=16_000,
    )
    i += 1
