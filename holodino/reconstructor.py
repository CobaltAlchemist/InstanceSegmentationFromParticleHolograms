import torch
from typing import Sequence, Literal

import numpy as np
import math
from typing import Sequence
from typing_extensions import Literal
import numpy as np
import torch
from functools import lru_cache
from typing import Any, Sequence
from fvcore.transforms.transform import (
    Transform,
)
import random


def create_batches(l: Sequence[Any], n: int, min_size: int = 0):
    assert min_size <= len(l), \
        "Minimum batch size must be less than the maximum batch size"
    for i in range(0, len(l), n):
        yield l[min(i, len(l) - min_size):i+n]

FLOAT_TO_COMPLEX = {
    torch.float: torch.complex64,
    torch.float16: torch.complex64,  # 32 is not supported by pytorch
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}
def adjustPeak(intensityProj, adjust_peak, substraction_factor):
    bin_list, _ = np.histogram(intensityProj, bins=list(range(256)))
    current_peak = list(range(256))[np.argmax(bin_list)]
    intensityProj = intensityProj.astype(np.float32)
    if current_peak > 0:
        bottom_divide = float(current_peak - substraction_factor)

        intensityProj = (intensityProj-substraction_factor) * \
            adjust_peak / (bottom_divide+1)
        intensityProj[intensityProj > 255] = 255
    intensityProj = intensityProj.astype(np.uint8)

    return intensityProj

#@torch.jit.script
def _reconstruct_script(holo: torch.Tensor, z_list: torch.Tensor, wavelength: torch.Tensor, f2: torch.Tensor):
    Fholo = torch.fft.fft2(holo)
    z_coeff = 2 * math.pi * 1j / wavelength
    z_coeff = torch.multiply(z_list, z_coeff)
    Hz = torch.exp(-z_coeff.view(-1, 1, 1) * (f2.unsqueeze(0)))
    phase_temp = torch.exp(z_coeff)
    Fplane = Fholo[None, ...] * Hz
    rec3 = torch.fft.ifft2(Fplane) * phase_temp[:, None, None]
    return rec3


@lru_cache(maxsize=2)
def _image_positions(height: int, width: int, wavelength: float, resolution: float, padding: int, precision=torch.float32, device: str = 'cpu'):
    width = width + 2 * padding
    height = height + 2 * padding
    x = torch.arange(width, dtype=precision, device=device)
    y = torch.arange(height, dtype=precision, device=device)
    x = (x / width - 0.5) ** 2
    y = (y / height - 0.5) ** 2
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    fx = xx.T
    fy = yy.T
    f2 = fx + fy
    f2 = torch.fft.fftshift(f2)
    f2_coeff = (wavelength / resolution) ** 2
    return torch.sqrt(1 - f2 * f2_coeff)

      
class Reconstructor:
    '''
    Reconstructs holograms given a resolution and wavelength
    will automatically switch between CPU and CUDA acceleration
    '''

    def __init__(self, resolution: float, wavelength: float, batch_size=100, precision=torch.float64):
        '''
        Constructs a reconstructor
            Parameters:
                resolution (float): The resolution of the workable images
                wavelength (float): The wavelength of the workable images
                batch_size (int): The maximum amount of reconstructed planes to process at a time
                precision (torch.dtype): The precision used when creating tensors
        '''
        self.device = torch.device('cpu')
        self.precision = precision
        self.batch_size = batch_size
        self.resolution = torch.tensor(
            resolution, dtype=self.precision, device=self.device)
        self.wavelength = torch.tensor(
            wavelength, dtype=self.precision, device=self.device)

    def _reconstruct(self, img, depths: Sequence[float], padding: int = 32):
        assert len(img.shape) == 2, "Only grayscale supported"
        holo = img.clone()
        holo = torch.nn.functional.pad(
            holo, (padding, ) * 4, mode='constant', value=torch.median(holo))
        holo = holo.to(dtype=self.precision, device=self.device)
        f2 = _image_positions(*img.shape, self.wavelength,
                              self.resolution, padding, precision=self.precision, device=self.device)
        t_depths = depths.clone()
        z_batches = list(create_batches(t_depths, self.batch_size))
        processed = torch.zeros((len(depths), *img.shape),
                                dtype=FLOAT_TO_COMPLEX[self.precision], device=self.device)
        i = 0
        for z_batch in z_batches:
            result = _reconstruct_script(
                holo, z_batch, self.wavelength, f2)
            result = result[:, padding:result.shape[1]-padding,
                            padding:result.shape[2]-padding]
            processed[i:i + len(z_batch), ...] = result.cpu()
            i += len(z_batch)
        return processed

    def _check_img(self, img):
        if isinstance(img, np.ndarray):
            img = torch.tensor(img, dtype=self.precision, device=self.device)
        return img

    def _check_depths(self, depths):
        depths = torch.tensor(depths, dtype=self.precision, device=self.device)
        if depths.ndim == 0:
            depths = depths.unsqueeze(0)
        return depths

    def intensity(self, img, depths: Sequence[float], padding: int = 32, reduction: Literal['xy', 'xz', 'yz', 'none'] = 'none'):
        '''
        Reconstructs a hologram image on the provided plane depths
            Parameters:
                img: Any object which may be converted to a 2-d tensor (H,W)
                planes (Sequence[float]): A sequence of depths to reconstruct on (B,)
                padding (int): Padding to apply to each side of the given image, default 32
                reduction (str): Which reduction to apply to the output, default none
            Returns:
                processed (tensor): Output intensity images (B,H,W)
        '''
        img = self._check_img(img)
        depths = self._check_depths(depths)
        processed = self._reconstruct(img, depths, padding=padding)
        processed *= torch.conj(processed)
        processed = torch.real(processed)
        p_min = processed.min()
        p_max = processed.max()
        processed -= p_min
        processed /= (p_max - p_min)
        if reduction == 'none':
            processed = processed
        elif reduction == 'xy':
            processed = processed.min(dim=0)
        elif reduction == 'xz':
            processed = processed.max(dim=1)
        elif reduction == 'yz':
            processed = processed.min(dim=2)
        return processed.cpu()

    def phase(self, img, depths: Sequence[float], padding: int = 32, reduction: Literal['xy', 'xz', 'yz', 'none'] = 'none'):
        '''
        Reconstructs a hologram image on the provided plane depths
            Parameters:
                img: Any object which may be converted to a 2-d tensor (H,W)
                planes (Sequence[float]): A sequence of depths to reconstruct on (B,)
                padding (int): Padding to apply to each side of the given image, default 32
                reduction (str): Which reduction to apply to the output, default none
            Returns:
                processed (tensor): Output phase images (B,H,W)
        '''
        img = self._check_img(img)
        depths = self._check_depths(depths)
        processed = self._reconstruct(img, depths, padding=padding)
        processed[...] = torch.angle(processed[...])
        processed = torch.real(processed)
        if reduction == 'none':
            processed = processed
        elif reduction == 'xy':
            processed = processed.min(dim=0)
        elif reduction == 'xz':
            processed = processed.max(dim=1)
        elif reduction == 'yz':
            processed = processed.min(dim=2)
        return processed.cpu()
    
class ReconstructionTransformer(Transform):
    def __init__(self, wavelength, resolution, depthmin, depthmax, probability=0.5):
        self.reconstructor = Reconstructor(resolution, wavelength)
        self.depthmin = depthmin
        if depthmax < depthmin:
            depthmax, depthmin = depthmin, depthmax
        depthmin = max(depthmin, 1)
        self.depthmax = depthmax
        self.depthmin = depthmin
        self.probability = probability
        
    def detectron2recon(self, img):
        img = img / 255
        img = img.mean(axis=2)
        return img
    
    def recon2detectron(self, img):
        img = (img * 255).astype('uint8')
        img = np.stack([img, img, img], axis=2)
        return img
        
    def intensity(self, img, plane):
        ret = np.real(self.reconstructor.intensity(img, [plane])[0].numpy())
        return ret
    
    def phase(self, img, plane):
        processed = self.reconstructor.phase(img, [plane])[0].numpy()
        ret = ((math.pi + processed)/(math.pi*2))
        return ret

    def pick_depth(self, min_value, max_value, b):
        rand_value = np.random.rand()
        return min_value + (max_value - min_value) * (np.power(rand_value, b))

    def apply_image(self, img: np.ndarray, interp=None) -> np.ndarray:
        t = 0#random.randint(0, 1)
        if random.random() > self.probability:
            ret = img
        else:
            depth = self.pick_depth(self.depthmin, self.depthmax, 2)
            r_img = self.detectron2recon(img)
            if t == 0:
                ret = self.intensity(r_img, depth)
            else:
                ret = self.phase(r_img, depth)
            ret = self.recon2detectron(ret)
            ret = adjustPeak(ret, 105, 0)
        return ret
        
    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords
        
    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def __call__(self, im):
        return im