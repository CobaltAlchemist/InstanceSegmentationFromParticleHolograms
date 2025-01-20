from functools import lru_cache
import cv2
from more_itertools import chunked
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

import torch
from torchvision.transforms import GaussianBlur
from scipy import ndimage


@dataclass
class Medium:
    ior: complex = 1. + 0j  # Index of refraction of medium
    intensity: float = 0.7  # Background light intensity


@dataclass
class Subject:
    ior: complex = 1.5 + 0j  # Index of refraction for object
    depth: float = 0.  # Z location of mask

    def draw(self, medium: Medium, oversample: int, height: int, width: int):
        pass


@dataclass
class HologramMask(Subject):
    mask: np.ndarray = None  # Mask representing object

    def draw(self, medium: Medium, oversample: int, img_size: Tuple[int, int]):
        assert not self.mask is None, "Mask must be defined"

        mask = self.mask
        if any(self.mask.shape[i] != img_size[i] for i in [0, 1]):
            mask = cv2.resize(self.mask, img_size[::-1],
                              interpolation=cv2.INTER_NEAREST)
        return mask, self.depth, {}


@dataclass
class Laser:
    wavelength: float = 0.632  # Wavelength in micrometers


class Synthesizer:
    def __init__(self,
                 resolution: float = 18.6,  # Micrometers per pixel
                 # Output image size of medium + holograms
                 output_size: Tuple[int, int] = (256, 256),
                 oversample: int = 4,
                 gaussian: bool = False,
                 gaussian_sigma: int = 7,
                 padding: bool = False):
        self.resolution = resolution
        output_size = tuple(max(s, 4) for s in output_size)
        self.output_size = np.array(output_size)
        self.oversample = oversample
        self.fft_size = self.output_size*self.oversample
        self.enable_gaussian = gaussian
        self.gaussian_sigma = gaussian_sigma
        self.enable_padding = padding

    @staticmethod
    @lru_cache(maxsize=10)
    def _create_hstack(fft_size: Tuple[int, int], resolution: float, wavelengths: Tuple[float], enable_gaussian, gaussian_sigma) -> np.ndarray:
        nx, ny = fft_size
        x = np.expand_dims(np.arange(nx), 0)
        y = np.expand_dims(np.arange(ny), 0)
        fx = ((x + nx / 2) % nx - nx / 2) / nx
        fy = ((y + ny / 2) % ny - ny / 2) / ny
        f2 = fx.T ** 2 + fy ** 2
        h_vals = np.zeros((len(wavelengths), nx, ny), dtype=np.complex128)
        for i, wavelength in enumerate(wavelengths):
            sqrt_inpt = 1 - (wavelength/resolution)**2 * f2
            # Set negative values to zero
            sqrt_inpt = np.where(sqrt_inpt < 0, 0, sqrt_inpt)
            H = -2*np.pi*1j*np.sqrt(sqrt_inpt.astype(np.complex128))/wavelength
            if enable_gaussian and np.mean(sqrt_inpt == 0) > 0.1:
                H = ndimage.gaussian_filter(H, sigma=gaussian_sigma)
            h_vals[i] = H
        return h_vals

    def _process_hologram(self, medium: Medium, fourier_hologram: np.ndarray, img_size: Tuple[int, int], fft_pad: np.ndarray, raw: bool) -> np.ndarray:
        height, width = img_size
        rec = np.fft.ifft2(fourier_hologram)
        # Find coordinates of the center of the hologram minus the padding
        ul = (fft_pad[0], fft_pad[1])
        lr = (height + fft_pad[0], width + fft_pad[1])
        rec = rec[ul[0]:lr[0], ul[1]:lr[1]]
        holo = rec+medium.intensity
        holo = holo*np.conj(holo)
        holo = np.abs(holo)
        if not raw:
            min = np.min(holo)
            max = np.max(holo)
            holo = (holo-min)/(max-min)
        holo = (holo+medium.intensity)/2
        return cv2.resize(holo, self.output_size[::-1], interpolation=cv2.INTER_LINEAR)

    def _mask_weights(self, medium: Medium, subjects: List[Subject]) -> np.ndarray:
        n1 = medium.ior
        n2 = np.array([subject.ior for subject in subjects])
        reflectance = (np.abs(n1 - n2) / (n1 + n2 + 1e-10))**2
        absorption = np.imag(n2)
        transmittance = 1 - reflectance - absorption
        mask_weight = np.clip(1 - transmittance, 0, 1)
        return mask_weight

    def generate(self, medium: Medium, lasers: List[Laser], subjects: List[Subject], raw: bool = False):
        wavelengths = tuple(
            laser.wavelength/medium.ior.real for laser in lasers)
        real_resolution = self.resolution/self.oversample
        img_size = self.oversample*self.output_size
        fft_pad = img_size // 2 if self.enable_padding else img_size * 0
        fft_size = self.fft_size + fft_pad*2
        h_vals = self._create_hstack(
            tuple(fft_size), real_resolution, wavelengths, self.enable_gaussian, self.gaussian_sigma)
        wavelengths = np.array(wavelengths)
        fourier_holograms = np.zeros(
            (len(wavelengths), *fft_size), dtype=np.complex128)

        for chunk in chunked(subjects, 50):
            # Shape: (num_subjects,)
            weights = self._mask_weights(medium, chunk)
            # Shape: (num_subjects, 1, 1)
            weights = np.expand_dims(weights, (1, 2))

            obj, depth, _ = zip(
                *[subject.draw(medium, self.oversample, img_size) for subject in chunk])
            # Shape: (num_subjects, height, width)
            obj = np.stack(obj).astype(np.float32) / 255
            # Shape: (num_subjects,)
            depth = np.stack(depth)
            # Shape: (num_subjects, num_wavelengths)
            phase = np.exp(np.expand_dims(1j * 2 * np.pi * depth,
                                          1) / np.expand_dims(wavelengths, 0))
            # Shape: (num_subjects, 1, 1)
            depth = np.expand_dims(depth, (1, 2))
            # Shape: (num_subjects, num_wavelengths, 1, 1)
            phase = np.expand_dims(phase, (2, 3))

            for wave_idx in range(len(wavelengths)):
                # Shape: (num_subjects, height, width)
                composite = weights*obj*phase[:, wave_idx]

                # Pad composite to prevent wrap-around
                if self.enable_padding:
                    composite = np.pad(
                        composite, ((0, 0), (fft_pad[0], fft_pad[0]), (fft_pad[1], fft_pad[1])), mode='constant')

                fourier_plane = np.fft.fft2(composite)
                hz = np.exp(depth*h_vals[wave_idx])
                fourier_holograms[wave_idx] -= (fourier_plane*hz).sum(axis=0)

        final_synthesis = np.stack(list(self._process_hologram(
            medium, holo, img_size, fft_pad, raw) for holo in fourier_holograms)).mean(axis=0)
        return final_synthesis

    def to_dict(self):
        return {
            'resolution': self.resolution,
            'output_size': tuple(map(int, self.output_size)),
            'oversample': self.oversample,
        }


class TorchSynthesizer(Synthesizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    @lru_cache(maxsize=8)
    def _get_range(size: int, device: torch.device) -> torch.Tensor:
        return torch.arange(size, dtype=torch.float, device=device).unsqueeze(0)

    @staticmethod
    def _create_hstack_script(fft_size: Tuple[int, int], resolution: float, wavelengths: torch.Tensor, enable_gaussian: bool, gaussian_sigma: int, device: torch.device) -> torch.Tensor:
        nx, ny = fft_size
        x = TorchSynthesizer._get_range(nx, device)
        y = TorchSynthesizer._get_range(ny, device)
        fx = ((x + nx / 2) % nx - nx / 2) / nx
        fy = ((y + ny / 2) % ny - ny / 2) / ny
        f2 = fx.T ** 2 + fy ** 2
        h_vals = torch.zeros((len(wavelengths), nx, ny),
                             dtype=torch.complex64, device=device)
        for i, wavelength in enumerate(wavelengths):
            sqrt_inpt = 1 - (wavelength/resolution)**2*f2
            # Set negative values to zero
            torch.nn.functional.relu(sqrt_inpt, inplace=True)
            H = -2*torch.pi*1j*torch.sqrt(sqrt_inpt)/wavelength
            if enable_gaussian and torch.mean((sqrt_inpt == 0).to(torch.float32)) > 0.1:
                H = GaussianBlur(gaussian_sigma)(H)
            h_vals[i] = H
        return h_vals

    @staticmethod
    @lru_cache(maxsize=10)
    def _create_hstack(fft_size: Tuple[int, int], resolution: float, wavelengths: Tuple[float], enable_gaussian: bool, gaussian_sigma: int, device: str) -> torch.tensor:
        return TorchSynthesizer._create_hstack_script(fft_size, resolution, torch.tensor(wavelengths, dtype=torch.float32, device=device), enable_gaussian, gaussian_sigma, device)

    @staticmethod
    def _process_hologram_script(fourier_hologram, intensity: float, img_size: Tuple[int, int], fft_pad: torch.Tensor):
        height, width = img_size
        rec = torch.fft.ifft2(fourier_hologram)
        ul = (fft_pad[0], fft_pad[1])
        lr = (height + fft_pad[0], width + fft_pad[1])
        rec = rec[ul[0]:lr[0], ul[1]:lr[1]]
        holo = rec+intensity
        holo = holo*torch.conj(holo)
        return torch.abs(holo)

    def _process_hologram(self, medium: Medium, fourier_hologram: torch.Tensor, img_size: Tuple[int, int], fft_pad: torch.Tensor, raw: bool) -> np.ndarray:
        holo = self._process_hologram_script(
            fourier_hologram, medium.intensity, img_size, fft_pad)
        if not raw:
            hmin = torch.min(holo)
            hmax = torch.max(holo)
            holo = (holo-hmin)/(hmax-hmin)
        holo = (holo+medium.intensity)/2
        holo = holo.cpu().numpy()
        return cv2.resize(holo, self.output_size[::-1], interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _mask_weights_script_old(mior: float, sior):
        reflectivity = (torch.abs(mior-sior) /
                        (mior+sior))**2
        transparency = 1-2*reflectivity
        torch.nn.functional.relu(transparency, inplace=True)
        mask_weight = 1-transparency
        return mask_weight

    @staticmethod
    def _mask_weights_script(mior: complex, sior):
        n1 = mior
        n2 = sior
        reflectance = torch.abs((n1 - n2) / (n1 + n2 + 1e-10)) ** 2
        absorption = torch.imag(n2)
        transmittance = 1 - reflectance - absorption
        mask_weight = torch.clip(1 - transmittance, 0, 1)
        return mask_weight

    def _mask_weights(self, medium: Medium, subjects: List[Subject]) -> torch.Tensor:
        sior = torch.tensor(
            [subject.ior for subject in subjects], dtype=torch.complex64, device=self.device)
        return self._mask_weights_script(medium.ior, sior)

    @staticmethod
    def _process_subject(h_vals, wavelengths, obj, depth, weights, enable_padding, fft_pad):
        phase = torch.exp((1j * 2 * torch.pi * depth[:, 0]) /
                          wavelengths[None])[..., None, None]
        fourier_holograms = torch.zeros_like(h_vals)
        for wave_idx, h_val in enumerate(h_vals):
            composite = weights * obj * phase[:, wave_idx]
            if enable_padding:
                composite = torch.nn.functional.pad(
                    composite, (fft_pad[1], fft_pad[1], fft_pad[0], fft_pad[0]), mode='constant')
            fourier_plane = torch.fft.fft2(composite)
            hz = torch.exp(depth * h_val)
            fourier_holograms[wave_idx] -= (fourier_plane*hz).sum(dim=0)
        return fourier_holograms

    def generate(self, medium: Medium, lasers: List[Laser], subjects: List[Subject], raw: bool = False) -> np.ndarray:
        wavelengths = tuple(laser.wavelength /
                            medium.ior.real for laser in lasers)
        real_resolution = self.resolution / self.oversample
        img_size = self.oversample * self.output_size
        fft_pad = img_size // 2 if self.enable_padding else img_size * 0
        fft_size = self.fft_size + fft_pad*2
        h_vals = self._create_hstack(
            tuple(fft_size), real_resolution, wavelengths, self.enable_gaussian, self.gaussian_sigma, self.device)
        wavelengths = torch.tensor(wavelengths, device=str(self.device))
        fourier_holograms = torch.zeros(
            (len(wavelengths), *fft_size), dtype=torch.complex64, device=self.device)
        for chunk in chunked(subjects, 10):
            weights = self._mask_weights(
                medium, chunk).reshape(-1, 1, 1)
            obj, depth, _ = zip(
                *[subject.draw(medium, self.oversample, img_size) for subject in chunk])
            obj = torch.tensor(
                np.array(obj), device=self.device, dtype=torch.float16) / 255
            depth = torch.tensor(depth, device=self.device,
                                 dtype=torch.float32).reshape(-1, 1, 1)
            fourier_holograms += self._process_subject(
                h_vals, wavelengths, obj, depth, weights, self.enable_padding, fft_pad)

        final_synthesis = np.mean([self._process_hologram(
            medium, holo, img_size, fft_pad, raw) for holo in fourier_holograms], axis=0)
        return final_synthesis
