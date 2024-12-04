import numpy as np
import torch
import os
from torchvision import transforms
import torch.utils.data
import PIL
import torchvision.transforms.functional as FT
from PIL import Image
from torch.utils.data import Dataset
import re
from itertools import combinations


if 'DATA_ROOT' in os.environ:
    DATA_ROOT = os.environ['DATA_ROOT']
else:
    DATA_ROOT = './data'

IMAGENET_PATH = './data/imagenet/raw-data'


def pad(img, size, mode):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    return np.pad(img, [(size, size), (size, size), (0, 0)], mode)


mean = {
    'mnist': (0.1307,),
    'cifar10': (0.4914, 0.4822, 0.4465)
}

std = {
    'mnist': (0.3081,),
    'cifar10': (0.2470, 0.2435, 0.2616)
}


class GaussianBlur(object):
    """
        PyTorch version of
        https://github.com/google-research/simclr/blob/244e7128004c5fd3c7805cf3135c79baa6c3bb96/data_util.py#L311
    """
    def gaussian_blur(self, image, sigma):
        image = image.reshape(1, 3, 224, 224)
        radius = np.int(self.kernel_size/2)
        kernel_size = radius * 2 + 1
        x = np.arange(-radius, radius + 1)

        blur_filter = np.exp(
              -np.power(x, 2.0) / (2.0 * np.power(np.float(sigma), 2.0)))
        blur_filter /= np.sum(blur_filter)

        conv1 = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), groups=3, padding=[kernel_size//2, 0], bias=False)
        conv1.weight = torch.nn.Parameter(
            torch.Tensor(np.tile(blur_filter.reshape(kernel_size, 1, 1, 1), 3).transpose([3, 2, 0, 1])))

        conv2 = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size), groups=3, padding=[0, kernel_size//2], bias=False)
        conv2.weight = torch.nn.Parameter(
            torch.Tensor(np.tile(blur_filter.reshape(kernel_size, 1, 1, 1), 3).transpose([3, 2, 1, 0])))

        res = conv2(conv1(image))
        assert res.shape == image.shape
        return res[0]

    def __init__(self, kernel_size, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img):
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)
            if np.random.uniform() < self.p:
                return self.gaussian_blur(img, sigma=np.random.uniform(0.2, 2))
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0}, p={1})'.format(self.kernel_size, self.p)

class CenterCropAndResize(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, proportion, size):
        self.proportion = proportion
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped and image.
        """
        w, h = (np.array(img.size) * self.proportion).astype(int)
        img = FT.resize(
            FT.center_crop(img, (h, w)),
            (self.size, self.size),
            interpolation=PIL.Image.BICUBIC
        )
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(proportion={0}, size={1})'.format(self.proportion, self.size)


class Clip(object):
    def __call__(self, x):
        return torch.clamp(x, 0, 1)


class MultiplyBatchSampler(torch.utils.data.sampler.BatchSampler):
    MULTILPLIER = 2

    def __iter__(self):
        for batch in super().__iter__():
            yield batch * self.MULTILPLIER


class ContinousSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, sampler, n_iterations):
        self.base_sampler = sampler
        self.n_iterations = n_iterations

    def __iter__(self):
        cur_iter = 0
        while cur_iter < self.n_iterations:
            for batch in self.base_sampler:
                yield batch
                cur_iter += 1
                if cur_iter >= self.n_iterations: return

    def __len__(self):
        return self.n_iterations
    
    def set_epoch(self, epoch):
        self.base_sampler.set_epoch(epoch)


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    # given from https://arxiv.org/pdf/2002.05709.pdf
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort


class DummyOutputWrapper(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, dummy):
        self.dummy = dummy
        self.dataset = dataset

    def __getitem__(self, index):
        return (*self.dataset[index], self.dummy)

    def __len__(self):
        return len(self.dataset)
    

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.object_folders = sorted([f for f in os.listdir(root_dir) 
                                    if os.path.isdir(os.path.join(root_dir, f))])
        print(self.object_folders[0])
        
        self.pairs = []
        
        for obj_id, folder in enumerate(self.object_folders):
            obj_path = os.path.join(root_dir, folder)
            frames = sorted([f for f in os.listdir(obj_path) if f.endswith('.jpg')])
            frames.sort(key=lambda x: int(re.search(r'frame_(\d+)', x).group(1)))
            
            # Generate all possible pairs for this object
            frame_paths = [os.path.join(obj_path, frame) for frame in frames]
            obj_pairs = list(combinations(frame_paths, 2))
            self.pairs.extend([(path1, path2, obj_id) for path1, path2 in obj_pairs])
        print("Positive pairs created")
            
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        img1_path, img2_path, obj_id = self.pairs[idx]
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return torch.stack([img1, img2]), obj_id
    
    def save_sample(self, idx, save_dir='pair_samples'):
        """Save a pair of images to visualize the positive pairs"""
        os.makedirs(save_dir, exist_ok=True)
        
        img_pair, obj_id = self[idx]
        # Convert tensor to PIL images
        img1 = transforms.ToPILImage()(img_pair[0])
        img2 = transforms.ToPILImage()(img_pair[1])
        
        # Save images
        img1.save(f'{save_dir}/pair_{idx}_obj{obj_id}_1.jpg')
        img2.save(f'{save_dir}/pair_{idx}_obj{obj_id}_2.jpg')