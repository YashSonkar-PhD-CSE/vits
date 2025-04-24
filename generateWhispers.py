import os
import torch
import torchaudio

from models import SynthesizerTrn
from text import symbols
import utils
import commons 
from text import text_to_sequence

# Text files path:
ROOT_TEXT_DIR = "/ssd_scratch/cvit/yash/LibriSpeech/text"

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

speakers = [2 * i for i in range(38)]

hps = utils.get_hparams_from_file("./configs/wtimit_192.json")


net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("vits_mod_decoder.pth", net_g, None)

def generateWhisperForSpeaker(text: torch.Tensor, textLengths: torch.LongTensor, sid: int):
    """
    Generate whisper for a given text and speaker id.
    :param text: Text to generate whisper for.
    :param sid: Speaker id.
    :return: Whisper for the given text and speaker id.
    """
    # Generate whisper
    with torch.no_grad():
        text = text.cuda().unsqueeze(0)
        textLengths = textLengths.cuda()
        sid = torch.LongTensor([sid]).cuda()
        audio = net_g.infer(text, textLengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8)[0][0, 0].data().cpu().float().numpy()
        return audio

def saveAudio(filename: str, audio: torch.Tensor):
    """
    Save audio to a file.
    :param filename: Filename to save the audio to.
    :param audio: Audio to save.
    """
    # Save audio
    torchaudio.save(filename, audio.unsqueeze(0), hps.data.sampling_rate)
    
def generateWhispersForText(txtFilePath: str):
    txt = ""
    with open(txtFilePath, "r") as f:
        txt = f.read().strip()
    if txt.strip() == "":
        print("Enpty string in file ", txtFilePath)
        return
    txtTensor = get_text(txt)
    txtLengthTensor = torch.LongTensor([txtTensor.size(0)])
    os.makedirs("/ssd_scratch/cvit/yash/LibriSpeech/genWav", exist_ok=True)
    for i in speakers:
        os.makedirs(f"/ssd_scratch/cvit/yash/LibriSpeech/genWav/{i}")
    # Generate whisper for each speaker
    for i in speakers:
        audio = generateWhisperForSpeaker(txt, i)
        saveAudio(
            os.path.join("/ssd_scratch/cvit/yash/LibriSpeech/genWav", str(i), txtFilePath.split("/")[-1].split(".txt")[0] + f"_{i}.wav"), 
            audio
        )
        print(f"Generated whisper for speaker {i}", end="\r")
        
def main():
    for file in os.listdir(ROOT_TEXT_DIR):
        if file.endswith(".txt"):
            generateWhispersForText(os.path.join(ROOT_TEXT_DIR, file))
            print(f"Generated whispers for {file}")

if __name__ == "__main__":
    main()