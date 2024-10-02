import math
import os

import torch
import torchvision
import numpy as np
import cv2

def read_rgb(rgb_file):
    bgr = cv2.imread(rgb_file)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    rgb = torchvision.transforms.functional.to_tensor(rgb)
    return rgb

def get_normalize(mean, std):
    normalize = torchvision.transforms.Normalize(mean=mean, std=std)
    unnormalize = torchvision.transforms.Normalize(mean=np.divide(-mean, std), std=(1.0 / std))
    return normalize, unnormalize

def get_neg1pos1_normalize():
    normalize, unnormalize = get_normalize(np.full(3, 0.5), np.full(3, 0.5))
    return normalize, unnormalize

class PairedImageDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, opt, split):
        super(PairedImageDataset, self).__init__()
        self.dir_AB = os.path.join(opt.output_dir, opt.scene, "ganerf", opt.exp_name, "generator_dataset_200000", split)  # get the image directory
        self.nerf_path = os.path.join(self.dir_AB, "nerf")
        self.real_path = os.path.join(self.dir_AB, "real")
        n_images = len(os.listdir(self.nerf_path))
        assert n_images == len(os.listdir(self.real_path))
        self.nerf_paths = ["{}.png".format(i) for i in range(n_images)]
        self.real_paths = ["{}.png".format(i) for i in range(n_images)]

        self.nerf_rgbs = list()
        self.real_rgbs = list()
        print("Start loading")
        for i, (nerf_rgb_path, real_rgb_path) in enumerate(zip(self.nerf_paths, self.real_paths)):
            img = read_rgb(os.path.join(self.nerf_path, nerf_rgb_path))
            self.nerf_rgbs.append(img)
            self.real_rgbs.append(read_rgb(os.path.join(self.real_path, real_rgb_path)))
        print("Finish loading")
        self.nerf_rgbs = torch.stack(self.nerf_rgbs, 0)
        self.real_rgbs = torch.stack(self.real_rgbs, 0)
        self.hflip = False if split in ["test", "val"] else opt.hflip
        self.data_aug_active = True
        self.patch_size = opt.patch_size
        self.load_full_image = False
        self.full_image_size = (img.shape[-2], img.shape[-1])
        self.normalize, self.unnormalize = get_neg1pos1_normalize()

    def apply_color_jitter(self, rgb, jitter_params):
        fn_idx, brightness, contrast, saturation, hue = jitter_params
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                rgb = torchvision.transforms.functional.adjust_brightness(rgb, brightness)
            elif fn_id == 1 and contrast is not None:
                rgb = torchvision.transforms.functional.adjust_contrast(rgb, contrast)
            elif fn_id == 2 and saturation is not None:
                rgb = torchvision.transforms.functional.adjust_saturation(rgb, saturation)
            elif fn_id == 3 and hue is not None:
                rgb = torchvision.transforms.functional.adjust_hue(rgb, hue)
        return rgb

    def get_color_jitter_params(self):
        centered_at_1 = (1.0 - self.color_jitter, 1.0 + self.color_jitter)
        centered_at_0 = (-self.color_jitter, self.color_jitter)
        jitter_params = torchvision.transforms.ColorJitter.get_params(
            brightness=centered_at_1, contrast=centered_at_1, saturation=centered_at_1, hue=centered_at_0
        )
        return jitter_params

    def activate_augmentations(self, value):
        self.data_aug_active = value

    def __getitem__(self, index):
        nerf_rgb = self.nerf_rgbs[index]
        real_rgb = self.real_rgbs[index]
        _, h, w = real_rgb.shape

        if self.load_full_image:
            nerf_rgb = torchvision.transforms.functional.resize(nerf_rgb, self.full_image_size)
            real_rgb = torchvision.transforms.functional.resize(real_rgb, self.full_image_size)
        else:
            # crop
            cropable_w = w - self.patch_size
            cropable_h = h - self.patch_size
            top = math.floor(cropable_h * np.random.random())
            left = math.floor(cropable_w * np.random.random())
            nerf_rgb = torchvision.transforms.functional.crop(nerf_rgb, top, left, self.patch_size, self.patch_size)
            real_rgb = torchvision.transforms.functional.crop(real_rgb, top, left, self.patch_size, self.patch_size)

        # horizontal flip
        if self.hflip and self.data_aug_active:
            apply_hflip = np.random.random() > 0.5
            if apply_hflip:
                nerf_rgb = torchvision.transforms.functional.hflip(nerf_rgb)
                real_rgb = torchvision.transforms.functional.hflip(real_rgb)

        # normalize
        nerf_rgb = self.normalize(nerf_rgb)
        real_rgb = self.normalize(real_rgb)

        return {"A": nerf_rgb, "B": real_rgb, "A_paths": self.nerf_paths[index], "B_paths": self.real_paths[index]}

    def __len__(self):
        return len(self.real_paths)

class SimpleTestDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, path):
        super(SimpleTestDataset, self).__init__()
        self.dir_A = path  # get the image directory
        self.nerf_paths = sorted(os.listdir(self.dir_A))
        self.nerf_rgbs = list()
        print("Start loading")
        for i, nerf_rgb_path in enumerate(self.nerf_paths):
            self.nerf_rgbs.append(read_rgb(os.path.join(self.dir_A, nerf_rgb_path)))
        print("Finish loading")
        self.nerf_rgbs = torch.stack(self.nerf_rgbs, 0)
        self.normalize, self.unnormalize = get_neg1pos1_normalize()

    def __getitem__(self, index):
        nerf_rgb = self.nerf_rgbs[index]

        # normalize
        nerf_rgb = self.normalize(nerf_rgb)

        return {"A": nerf_rgb, "B": torch.empty(0), "A_paths": self.nerf_paths[index], "B_paths": ""}

    def __len__(self):
        return len(self.nerf_paths)
