import math
import os
import random
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import wandb
import commons
from mel_processing import spectrogram_torch
import monotonic_align
from text import cleaned_text_to_sequence, text_to_sequence
import utils
from models import (
  SynthesizerTrn,
)
from text.symbols import newSymbols


torch.backends.cudnn.benchmark = True
globalStep = 0

class TextWhisperDataset(torch.utils.data.Dataset):
    def __init__(self, audioPathsSidText, hparams):
        # self.audioPathsSidText = utils.load_filepaths_and_text(audioPathsSidText)
        self.textCleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate
        
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        # random.shuffle(self.audiopaths_sid_text)
        # self._filter()
        
    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        
    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)
    
    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return (text, spec, wav, sid)

    def get_audio(self, filename):
        audio, sampling_rate = utils.load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid
    
class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid

def main():
    assert torch.cuda.is_available()
    nGPUs = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_ADDR'] = '12350'
    os.environ['MASTER_PORT'] = '8008'
    
    hps: utils.HParams = utils.get_hparams()
    mp.spawn(run, nprocs = nGPUs, args=(nGPUs, hps, ))
    
def run(rank: int, nGPUs: int, hps: utils.HParams):
    global globalStep
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        # writer = SummaryWriter(log_dir=hps.model_dir)
        # writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    torch.cuda.set_device(rank)
    device = torch.device('cuda')
    # trainDataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    # trainSampler = DistributedBucketSampler(
    #     trainDataset,
    #     hps.train.batch_size,
    #     [32,300,400,500,600,700,800,900,1000],
    #     num_replicas=nGPUs,
    #     rank=rank,
    #     shuffle=True
    # )
    # if rank == 0:
    #     eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    #     eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
    #         batch_size=hps.train.batch_size, pin_memory=True,
    #         drop_last=False, collate_fn=collate_fn)
    
    netG = SynthesizerTrn(
      len(newSymbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      **hps.model).cuda(rank)
    netG.load_state_dict(torch.load('./G_661000.pth', weights_only=True, map_location='cuda')['model'])
    # Load state dict of generator into netG
    ds = TextWhisperDataset('', hps.data)
    audio = "./data/speaker01_english_nonnative_effort1_1_1.wav"
    text = "./data/speaker01_english_nonnative_effort1_1_1.txt"
    sid = ds.get_sid(0) # TODO: Modify sid
    spec, wav = ds.get_audio(audio)
    # text = ds.get_text("ðɪs ˈiːbʊk ɪz fɚðə jˈuːs ʌv ˈɛnɪwˌʌn ˈɛnɪwˌɛɹ æt nˈoʊ kˈɔst ænd wɪð ˈɔːlmoʊst nˈoʊ ɹᵻstɹˈɪkʃənz wʌtsˌoʊˈɛvɚ.")
    text = ds.get_text("baɪ ˈædəm smˈɪθ")
    print(spec.shape, wav.shape, text.shape)
    batch = [(text, spec, wav, sid)]
    textPadded, textLengths, specPadded, specLengths, wavPadded, wavLengths, sid = TextAudioSpeakerCollate()(batch)
    
    textEncoder = netG.enc_p
    for param in textEncoder.parameters():
        param.requires_grad = False
    
    dp = netG.dp.to(device)
    for param in dp.parameters():
        param.requires_grad = False
    whisperEmbedder = netG.enc_q.to(device)
    flow = netG.flow.to(device)
    
    # Pipeline:
    # Use textEncoder to generate targets for whisperEmbedder
    # Pass cleaned text to textEncoder to get text embeddings
    # Pass whisper signals to whisper embedder and then pass the output to the flow
    # Minimize the loss
    # TODO: Check dimensionality at multiple steps
    # X, XLengths = Text, TextLengths
    with torch.autocast('cuda', enabled=hps.train.fp16_run):
        textPadded, textLengths = textPadded.to(device), textLengths.to(device)
        specPadded, specLengths = specPadded.to(device), specLengths.to(device)
        print(textPadded.shape, textLengths.shape, specPadded.shape, specLengths.shape, sid.shape)
        sid = sid.to(device)
        textEncoder = textEncoder.to(device)
        x, mP, logsP, xMask = textEncoder(textPadded, textLengths)
        # # Y, YLengths = Audio spec and spec lengths
        # # G = speaker embeddings, can be none
        # # TODO: Try setting g to None and see if it breaks anything
        gEmbedder = netG.emb_g.to(device)
        # g = gEmbedder(sid).unsqueeze(-1)
        g = None
        z, mQ, logsQ, yMask = whisperEmbedder(specPadded, specLengths, g = g)
        zP = flow(z, yMask, g = g)
        print("Text Outs:")
        print(x.shape, mP.shape, logsP.shape, xMask.shape)
        print("Embed output")
        print(z.shape, mQ.shape, logsQ.shape, yMask.shape)
        print("Flow Out")
        print(zP.shape)
        
        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logsP) # [b, d, t]
            negCent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logsP, [1], keepdim=True) # [b, 1, t_s]
            negCent2 = torch.matmul(-0.5 * (zP ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            negCent3 = torch.matmul(zP.transpose(1, 2), (mP * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            negCent4 = torch.sum(-0.5 * (mP ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
            negCent = negCent1 + negCent2 + negCent3 + negCent4

            attn_mask = torch.unsqueeze(xMask, 2) * torch.unsqueeze(yMask, -1)
            attn = monotonic_align.maximum_path(negCent, attn_mask.squeeze(1)).unsqueeze(1).detach()
        
        w = attn.sum(2)
        lLength = dp(x, xMask, w, g = g) / torch.sum(xMask)
        mP = torch.matmul(attn.squeeze(1), mP.transpose(1, 2)).transpose(1, 2)
        logsP = torch.matmul(attn.squeeze(1), logsP.transpose(1, 2)).transpose(1, 2)
        print(mP.shape, logsP.shape)
    # print(zP.shape, x.shape, mP.shape, logsP.shape, xMask.shape)
    # print("FROZEN")
    # print(textEncoder)
    print("DONE")
        
if __name__ == "__main__":
    main()
        