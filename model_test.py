import os
import numpy as np
from PIL import Image
from torch import nn
from dataset.videomakeup import (
    VideoMakeupDataset,
    VideoMakeupTrainAugmentation,
    VideoMakeupValidAugmentation
)

from dataset.augmentation import (
    TrainFrameSampler,
    ValidFrameSampler
)
from dataset.youtubevis import (
    YouTubeVISAugmentation
)
from model import MattingNetwork
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_loss import makeup_loss, segmentation_loss


dset = VideoMakeupDataset(
                videomakeup_dir='/Users/abhishek/Desktop/PetProjects/gunjan/makeup/data/aligned_data_test',
                size=256,
                seq_length=15,
                seq_sampler=TrainFrameSampler(),
                transform=VideoMakeupTrainAugmentation(256))

dataloader_train = DataLoader(
            dataset=dset,
            batch_size=1,
            num_workers=4,
            pin_memory=True)

model = MattingNetwork("mobilenetv3", pretrained_backbone=True)
# model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

# for no_mkp, true_mkp in dataloader_train:

    # print(no_mkp.shape, true_mkp.shape)
    # pred_mkp = model(no_mkp, downsample_ratio=1)[0]
    # break

no_mkp, true_mkp = dset[0]
no_mkp = no_mkp.unsqueeze(0)
true_mkp = true_mkp.unsqueeze(0)
# print(no_mkp.shape, true_mkp.shape)

pred_mkp = model(no_mkp, downsample_ratio=1)[0]
print(pred_mkp.shape)
mkp_loss = makeup_loss(pred_mkp, true_mkp)
print(mkp_loss['total'])

seg_dset = VideoMakeupDataset(
                videomakeup_dir='/Users/abhishek/Desktop/PetProjects/gunjan/makeup/data/aligned_data_test',
                size=256,
                seq_length=15,
                seq_sampler=TrainFrameSampler(),
                is_segmentation=True,
                transform=YouTubeVISAugmentation(256))

no_mkp, seg = seg_dset[0]
no_mkp = no_mkp.unsqueeze(0)
seg = seg.unsqueeze(0)
# print(no_mkp.shape, seg.shape)

pred_seg = model(no_mkp, segmentation_pass=True)[0]
print(pred_seg.shape)
seg_loss = segmentation_loss(pred_seg, seg)
print(seg_loss)