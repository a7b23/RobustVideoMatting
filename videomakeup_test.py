import os
import numpy as np
from PIL import Image
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
from torch.utils.data import DataLoader
from tqdm import tqdm


dset = VideoMakeupDataset(
                videomakeup_dir='/Users/abhishek/Desktop/PetProjects/gunjan/makeup/data/aligned_data_test',
                size=256,
                seq_length=15,
                seq_sampler=TrainFrameSampler(),
                transform=VideoMakeupTrainAugmentation(256))

dataloader_train = DataLoader(
            dataset=dset,
            batch_size=2,
            num_workers=4,
            pin_memory=True)

for no_mkp, true_mkp in tqdm(dataloader_train, disable=False, dynamic_ncols=True):
    print(no_mkp.shape, true_mkp.shape)
    no_mkp_data = no_mkp.cpu().data.numpy()
    true_mkp_data = true_mkp.cpu().data.numpy()
    break

out_dir = "./makeup_images_vis"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for idx in range(len(no_mkp_data)):
    out_sub_dir = os.path.join(out_dir, str(idx))
    if not os.path.exists(out_sub_dir):
        os.makedirs(out_sub_dir)
    for t in range(len(no_mkp_data[idx])):
        out_fname = os.path.join(out_sub_dir, str(t) + ".png")
        img1 = np.transpose(np.array(no_mkp_data[idx][t]), (1, 2, 0))*255.0
        img2 = np.transpose(np.array(true_mkp_data[idx][t]), (1, 2, 0))*255.0
        img = np.concatenate((img1, img2), axis=1)
        img = Image.fromarray(img.astype(np.uint8))
        img.save(out_fname)

seg_dset = VideoMakeupDataset(
                videomakeup_dir='/Users/abhishek/Desktop/PetProjects/gunjan/makeup/data/aligned_data_test',
                size=256,
                seq_length=15,
                seq_sampler=TrainFrameSampler(),
                is_segmentation=True,
                transform=YouTubeVISAugmentation(256))

dataloader_seg = DataLoader(
            dataset=seg_dset,
            batch_size=2,
            num_workers=4,
            pin_memory=True)

for no_mkp, seg in tqdm(dataloader_seg, disable=False, dynamic_ncols=True):
    print(no_mkp.shape, seg.shape)
    no_mkp_data = no_mkp.cpu().data.numpy()
    seg_data = seg.cpu().data.numpy()
    break

out_dir = "./seg_images_vis"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for idx in range(len(no_mkp_data)):
    out_sub_dir = os.path.join(out_dir, str(idx))
    if not os.path.exists(out_sub_dir):
        os.makedirs(out_sub_dir)
    for t in range(len(no_mkp_data[idx])):
        out_fname = os.path.join(out_sub_dir, str(t) + ".png")
        img1 = np.transpose(np.array(no_mkp_data[idx][t]), (1, 2, 0))*255.0
        img2 = np.repeat(np.transpose(np.array(seg_data[idx][t]), (1, 2, 0))*255.0, 3, axis=-1)
        img = np.concatenate((img1, img2), axis=1)
        img = Image.fromarray(img.astype(np.uint8))
        img.save(out_fname)