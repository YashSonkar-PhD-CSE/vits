import os
import librosa
import wandb
import commons
from data_utils import DistributedBucketSampler
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from models import MultiPeriodDiscriminator, PosteriorEncoder, Generator
import torch
from torch.utils.tensorboard import SummaryWriter
import utils
import numpy as np
# Input Waveform: (1, 1, N)
# HuBERT Units: 
#   Soft: (1, N', 256)
#   Discrete: (N')
#   N' = number of HuBERT units
#   N = number of frames in the input waveform
#   N' = N / (16 * 20)
#   Discrete to be used
# Posterior Encoder input: (1, filterLen // 2 + 1, N'')
# Posterior Encoder output: (1, interChannels, N'')
# N'' may or may not be equal to N' (post enc can work on any value of N'')
# Decoder: Generator
# Decoder input: (1, interChannels, N'')
# Decoder output: (1, 1, N)

lr = 2e-4
betas = (0.8, 0.99)
eps = 1e-9
lrDecay = 0.999875
numTrainEpochs = 20000
fp16Run = True
logInterval = 200
evalInterval = 1000
logDir = 'logs/wts_embed_mod'

globalStep = 0
epochStr = 1

hubert = torch.hub.load(
    "bshall/hubert:main",
    f"hubert_soft",
    trust_repo=True,
).cuda()
enc = PosteriorEncoder(
    in_channels=256,
    out_channels=32,
    hidden_channels=1,
    kernel_size=5,
    dilation_rate=1,
    n_layers=16,
    gin_channels=0,
)
dec = Generator(
    initial_channel=32,
    resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[8, 8, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 4, 4],
    gin_channels=192
)

disc = MultiPeriodDiscriminator(
    use_spectral_norm=False,
).cuda()

class Model(torch.nn.Module):
    def __init__(
        self,
        hubert: torch.nn.Module,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
    ):
        super(Model, self).__init__()
        self.hubert = hubert.eval() # (1, N', 256)
        self.embedder = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 256),
        )
        self.encoder = encoder # (1, N', 256)
        
        self.decoder = decoder # (1, 1, N'')
    
    def forward(self, x: torch.Tensor):
        hX = self.hubert.units(x).clone() # (B, N', 256)
        emX = self.embedder(hX) # (B, N', 256)
        emXLengths = torch.tensor([emXi.size(-1) for emXi in emX]).cuda() # (B)
        emX = emX.transpose(1, 2) # (B, 256, N')
        encX, mQ, logsQ, yMask = self.encoder(emX, emXLengths)
        # zSlice, idsSlice = commons.rand_slice_segments(encX, emXLengths, 8192)
        decX = self.decoder(encX) # (B, 1, N'')
        return decX
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str = 'train',
        dataPath: str = '/ssd_scratch/cvit/yash/wTIMIT/converted_wavs/normal/'
    ):
        self.split = split
        self.dataPath = dataPath
        self.filelist = []
        self.speakerMap = {0: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6, 14: 7, 16: 8, 18: 9, 101: 10, 103: 11, 105: 12, 107: 13, 109: 14, 111: 15, 116: 16, 118: 17, 120: 18, 122: 19, 124: 20, 126: 21, 128: 22, 130: 23, 1: 24, 3: 25, 5: 26, 7: 27, 9: 28, 11: 29, 13: 30, 15: 31, 17: 32, 19: 33, 102: 34, 104: 35, 106: 36, 108: 37, 110: 38, 112: 39, 117: 40, 119: 41, 121: 42, 123: 43, 125: 44, 127: 45, 129: 46, 131: 47}
        for speaker in os.listdir(dataPath):
            if split == 'val' and int(speaker[1:]) < 121:
                continue
            
            for file in os.listdir(os.path.join(dataPath, speaker, "wavs")):
                if file.endswith(".wav"):
                    self.filelist.append(os.path.join(dataPath, speaker, "wavs", file))
        self.maxWavValue = 1.33
        self.samplingRate = 16000
        self.filterLengths = 1024
        self.hopLength = 256
        self.winLength = 1024
            
    def __len__(self):
        return len(self.filelist)      
    
    def __getitem__(self, index):
        filePath = self.filelist[index]
        audio, sr = librosa.load(filePath, sr=self.samplingRate, dtype=np.float32)
        audio = torch.FloatTensor(audio)
        audio = audio / self.maxWavValue
        audio = audio.unsqueeze(0)
        sid = self.speakerMap[int(filePath.split("/")[-3][1:])]
        sid = torch.LongTensor([sid])
        return audio, sid

class DataCollate():
    def __init__(self):
        """Don't need to initialize anything for now"""
        pass
    
    def __call__(self, batch):
        # Any entry in batch is a tuple of (audio, sid)
        maxAudioLen = max([b[0].size(-1) for b in batch])
        audioLens = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        
        audioPadded = torch.FloatTensor(len(batch), 1, maxAudioLen)
        audioPadded.zero_()
        
        for i in range(len(batch)):
            row = batch[i]
            audio = row[0]
            sid[i] = row[1]
            audioPadded[i, :, :audio.size(1)] = audio
            audioLens[i] = audio.size(1)
        return audioPadded, audioLens, sid

def run():
    model = Model(hubert, enc, dec).cuda()
    
    optimG = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
    )

    optimD = torch.optim.AdamW(
        disc.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
    )

    schedulerG = torch.optim.lr_scheduler.ExponentialLR(
        optimG,
        gamma = lrDecay,
        last_epoch = epochStr - 2
    )
    schedulerD = torch.optim.lr_scheduler.ExponentialLR(
        optimD,
        gamma = lrDecay,
        last_epoch = epochStr - 2
    )

    scaler = torch.amp.GradScaler(device='cuda', enabled=fp16Run)
    logger = utils.get_logger(logDir)
    writer = SummaryWriter(logDir)
    valWriter = SummaryWriter(logDir + "/val")
    
    trainDS = CustomDataset()
    collate_fn = DataCollate()
    trainLoader = torch.utils.data.DataLoader(
        trainDS,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )
    valDS = CustomDataset(split='val')
    valLoader = torch.utils.data.DataLoader(
        valDS,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    for epoch in range(epochStr, numTrainEpochs + 1):
        train(epochStr, [model, disc], [optimG, optimD], [schedulerG, schedulerD], scaler, [trainLoader, valLoader], logger, [writer, valWriter])
        schedulerG.step()
        schedulerD.step()
    
def train(epoch, models, optims, schedulers, scaler, loaders, logger, writers):
    modelG, modelD = models
    optimG, optimD = optims
    schedulerG, schedulerD = schedulers
    
    trainLoader, valLoader = loaders
    if writers is not None:
        writer, valWriter = writers
    
    wandb.login(key="4abbb21b2d83424beaac33db691b8736ef01b7ed")
    wandb.init(
        project = "Speech Experiments",
        name = "ContExtractionMod",
    )
    
    global globalStep
    
    modelG.train()
    modelG.hubert.eval()
    modelD.train()

    for batchIdx, (y, yLengths, speakers) in enumerate(trainLoader):
        y, yLenths = y.cuda(non_blocking=True), yLengths.cuda(non_blocking=True)
        speakers = speakers.cuda(non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled = fp16Run):
            yHat = modelG(y)
            # y = commons.slice_segments(y, idsSlice * 256, 8192) # slice 
            # Discriminator
            yDHatR, yDHatG, _, _ = modelD(y, yHat.detach())
            with torch.amp.autocast('cuda', enabled=False):
                lossDisc, lossesDiscR, lossesDiscG = discriminator_loss(yDHatR, yDHatG)
                lossDiscAll = lossDisc
        optimD.zero_grad()
        scaler.scale(lossDiscAll).backward()
        scaler.unscale_(optimD)
        gradNormD = commons.clip_grad_value_(modelD.parameters(), None)  
        scaler.step(optimD)
        
        with torch.amp.autocast('cuda', enabled = fp16Run):
            # Generator
            yHat = modelG(y)
            lossDur = torch.floatTensor([0.0]).cuda()
            for pred in range(y.shape[0]):
                lossDur += (yHat[i].size()[-1] - y.size()[-1]) ** 2
            lossDur /= y.shape[0]
            if y.shape[-1] < yHat.shape[-1]:
                pass # Pad y to make length equal
            elif y.shape[-1] > yHat.shape[-1]:
                pass # Pad yHat to make length equal
            print(y.shape, yHat.shape)
            yDHatR, yDHatG, fMapR, fMapG = modelD(y, yHat)
            with torch.amp.autocast('cuda', enabled=False):
                lossGen, lossesGen = generator_loss(yDHatG)
                lossFM = feature_loss(fMapR, fMapG)
                lossGenAll = lossGen + lossFM + lossDur

        optimG.zero_grad()
        scaler.scale(lossGenAll).backward()
        scaler.unscale_(optimG)
        gradNormG = commons.clip_grad_value_(modelG.parameters(), None)  
        scaler.step(optimG)
        scaler.update()
        
        if globalStep % logInterval == 0:
            lr = optimG.param_groups[0]['lr']   
            losses = [lossDisc, lossGen, lossFM]
            logger.info("Train Epoch: {} [{:0f}%]".format(epoch, 100. * batchIdx / len(trainLoader)))
            logger.info([x.item() for x in losses] + [globalStep, lr])
            
            wandb.log({
                "lossDisc": lossDisc.item(),
                "lossGen": lossGen.item(),
                "lossFM": lossFM.item(),
                "globalStep": globalStep,
                "gradNormG": gradNormG,
                "gradNormD": gradNormD,
                "lr": lr,
            })
            
            scalarDict = {
                "loss/g/total": lossGenAll,
                "loss/d/total": lossDiscAll,
                "learning_rate": lr,
                "grad_norm_g": gradNormG,
                "grad_norm_d": gradNormD,
                "loss/g/fm": lossFM,
            }
            
            scalarDict.update({"loss/g/{}".format(i): v for i, v in enumerate(lossesGen)})
            scalarDict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(lossesDiscR)})
            scalarDict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(lossesDiscG)})
            imageDict = {}
            
            utils.summarize(
                writer = writer,
                global_step=globalStep,
                images = imageDict,
                scalars=scalarDict
            )
        
        if globalStep % evalInterval == 0:
            evaluate(modelG, valLoader, valWriter)
            utils.save_checkpoint(modelG, optimD, lr, epoch, os.path.join(logDir, "G_{}.pth".format(globalStep)))   
            utils.save_checkpoint(modelD, optimD, lr, epoch, os.path.join(logDir, "D_{}.pth".format(globalStep)))   
        globalStep += 1
        logger.info('----> Epoch: {}'.format(epoch))
        
    def evaluate(model, loader, writer):
        pass
    
if __name__ == "__main__":
    run()
