import os
import random
from torch.utils.data import Dataset
from PIL import Image

# from .augmentation import MotionAugmentation


class VideoMakeupDataset(Dataset):
    def __init__(self,
                 videomakeup_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 is_segmentation=False,
                 transform=None):
        
        self.videomakeup_dir = videomakeup_dir
        self.videomakeup_clips = sorted(os.listdir(videomakeup_dir))
        self.videomakeup_frames = [sorted(os.listdir(os.path.join(videomakeup_dir, clip))) 
                                  for clip in self.videomakeup_clips]
        self.videomakeup_idx = [(clip_idx, frame_idx) 
                               for clip_idx in range(len(self.videomakeup_clips)) 
                               for frame_idx in range(0, len(self.videomakeup_frames[clip_idx]), seq_length)]
        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform
        self.is_segmentation = is_segmentation

    def __len__(self):
        return len(self.videomakeup_idx)
    
    def __getitem__(self, idx):

        no_mkps, true_mkps, segs = self._get_videomakeup(idx)
        
        # if self.transform is not None:
        #     return self.transform(fgrs, phas, bgrs)
        if not self.is_segmentation:
            return no_mkps, true_mkps
        else:
            return no_mkps, segs
    
    def _get_videomakeup(self, idx):
        clip_idx, frame_idx = self.videomakeup_idx[idx]
        clip = self.videomakeup_clips[clip_idx]
        frame_count = len(self.videomakeup_frames[clip_idx])
        no_mkps, true_mkps, segs = [], [], []
        for i in self.seq_sampler(self.seq_length):
            frame = self.videomakeup_frames[clip_idx][(frame_idx + i) % frame_count]
            with Image.open(os.path.join(self.videomakeup_dir, clip, frame)) as img:
                img = img.convert('RGB')
                w, h = img.size
                imgs_total = int(w/h)
                w2 = int(w / imgs_total)
                A = img.crop((w2, 0, 2*w2, h)) #nomk
                B = img.crop((2*w2, 0, 3*w2, h)) #gt
                D = img.crop((3*w2, 0, w, h)) #lip mask of nomk
            no_mkps.append(A)
            true_mkps.append(B)
            segs.append(D)
        return no_mkps, true_mkps, segs

    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img
