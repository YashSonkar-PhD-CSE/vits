# # from data_utils import DistributedBucketSampler, TextAudioSpeakerLoader, TextAudioSpeakerCollate
# # from torch.utils.data import DataLoader
# # import utils 

# # hps = utils.get_hparams_from_file("./configs/wtimit_192.json")
# # ds = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
# # collate_fn = TextAudioSpeakerCollate(hps.data)
# # sampler = DistributedBucketSampler(
# #       ds,
# #       hps.train.batch_size,
# #       [32,300,400,500,600,700,800,900,1000],
# #       num_replicas=1,
# #       rank=0,
# #       shuffle=True)
# # train_loader = DataLoader(ds, num_workers=8, shuffle=False, pin_memory=True,
# #     collate_fn=collate_fn, batch_sampler=sampler)
# # for batch in train_loader:
# #     print(len(batch), batch[-2].shape, batch[0])
# #     break

# sps = ["s000" , "s002" , "s004" , "s006" , "s008" , "s010" , "s012" , "s014" , "s016" , "s018" , "s101" , "s103" , "s105" , "s107" , "s109" , "s111" , "s116" , "s118" , "s120" , "s122" , "s124" , "s126" , "s128" , "s130", "s001" , "s003" , "s005" , "s007" , "s009" , "s011" , "s013" , "s015" , "s017" , "s019" , "s102" , "s104" , "s106" , "s108" , "s110" , "s112" , "s117" , "s119" , "s121" , "s123" , "s125" , "s127" , "s129" , "s131"]
# sps = [int(s[1:]) for s in sps]
# map = {}
# for i in range(len(sps)):
#     map[sps[i]] = i 
# print(map)

from speech_reconstruction import CustomDataset, DataCollate
from torch.utils.data import DataLoader
import torch

ds = CustomDataset()
print(len(ds), ds[0])
from speech_reconstruction import get_model

model = get_model().cuda()
model.eval()

dataset = CustomDataset()
collater = DataCollate()
dataLoader = DataLoader(dataset, collate_fn=collater, num_workers=1, shuffle=False, batch_size=1, pin_memory=True, drop_last=False)
data_list = list(dataLoader)

with torch.no_grad():
    y, y_lengths, sid_src = [x.cuda() for x in data_list[0]]
    # print(spec.shape, spec_lengths)
    sid_tgt1 = torch.LongTensor([38]).cuda()
    sid_tgt2 = torch.LongTensor([53]).cuda()
    sid_tgt3 = torch.LongTensor([54]).cuda()
    audio1 = model.infer(y, sid=sid_tgt1)[0,0].data.cpu().float().numpy()
    audio2 = model.infer(y, sid=sid_tgt2)[0,0].data.cpu().float().numpy()
    audio3 = model.infer(y, sid=sid_tgt3)[0,0].data.cpu().float().numpy()
    audio4 = model.infer(y, sid=sid_src)[0,0].data.cpu().float().numpy()
    print(y.shape, audio1.shape, audio2.shape, audio3.shape)    