import random
from models import SynthesizerTrn
import torch
import nemo.collections.asr as nemo_asr
from text.symbols import symbols
import utils

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
projLayer = torch.nn.Linear(in_features=192, out_features=256)
hps: utils.HParams = utils.get_hparams()
rank = 0
netG = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      **hps.model).cuda(rank)
netG.load_state_dict(torch.load('./G_661000.pth', weights_only=True, map_location='cuda')['model'])
embG = netG.emb_g

# Freezing things
for param in speaker_model.parameters():
    param.requires_grad = False
for param in netG.parameters():
    param.requires_grad = False
for param in embG.parameters():
    param.requires_grad = False

optim = torch.optim.AdamW(projLayer.parameters(), lr = 1e-4)
criterion = torch.nn.MSELoss()

numEpochs = 50
data = []
shuffle = True
device = torch.device('cuda')
with open('./filelists/wtimit_train2.txt.cleaned', 'r+') as dataFile:
    data = dataFile.readlines()
for epoch in range(numEpochs):
    trainLoss = 0
    if shuffle:
        random.shuffle(data)
    for line in data:
        wavPath, sid, cT = line.split("|")
        sid = torch.LongTensor([int(sid)]).to(device)
        emb = speaker_model.get_embedding(wavPath)
        proj = projLayer(emb)
        loss = criterion(emb, proj)
        trainLoss += loss.item()
        loss.backward()
        optim.step()
    print(trainLoss)
    torch.save(projLayer.state_dict, f'projector_{epoch}.pt')
    