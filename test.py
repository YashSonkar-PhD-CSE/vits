from data_utils import DistributedBucketSampler, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from torch.utils.data import DataLoader
import utils 

hps = utils.get_hparams_from_file("./configs/wtimit_192.json")
ds = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
collate_fn = TextAudioSpeakerCollate(hps.data)
sampler = DistributedBucketSampler(
      ds,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=1,
      rank=0,
      shuffle=True)
train_loader = DataLoader(ds, num_workers=8, shuffle=False, pin_memory=True,
    collate_fn=collate_fn, batch_sampler=sampler)
for batch in train_loader:
    print(len(batch), batch[-2].shape, batch[0])
    break