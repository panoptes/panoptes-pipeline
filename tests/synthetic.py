"""Synthetic PANOPTES-like observations for testing the algorithm.

Generates postage stamp cubes that reproduce the systematics the algorithm
exists to defeat: sub-pixel stellar positions interacting with a Bayer color
filter array, per-star color, periodic mount tracking error, and photon plus
read noise. No real data or network access required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panoptes.pipeline.lightcurve import masks


@dataclass
class SyntheticObservation:
    """A generated observation. ``pscs`` has shape ``(num_stars, num_frames, num_pixels)``."""

    pscs: np.ndarray
    times: np.ndarray
    stamp_shape: tuple[int, int]
    fluxes: np.ndarray
    colors: np.ndarray

    @property
    def num_pixels(self) -> int:
        return self.pscs.shape[2]


def gaussian_psf(shape, center, sigma):
    """Unit-sum 2D Gaussian sampled on a pixel grid."""
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    cy, cx = center
    psf = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)))
    total = psf.sum()
    return psf / total if total else psf


def make_observation(
    num_stars: int = 150,
    num_frames: int = 40,
    stamp_shape: tuple[int, int] = (10, 10),
    seed: int = 42,
    sigma: float = 1.4,
    base_flux: float = 2.0e5,
    background: float = 400.0,
    read_noise: float = 10.0,
    drift_amplitude: float = 1.0,
    noise: bool = True,
) -> SyntheticObservation:
    """Generate an observation of ``num_stars`` stars over ``num_frames`` frames."""
    rng = np.random.default_rng(seed)
    height, width = stamp_shape
    rgb = masks.rgb_masks(stamp_shape)

    # Periodic tracking error, shared by every star in the frame.
    times = np.arange(num_frames, dtype=float) * 35.0
    phase = 2.0 * np.pi * times / (8.0 * 60.0)
    drift_y = drift_amplitude * np.sin(phase)
    drift_x = 0.3 * drift_amplitude * np.cos(phase)

    # Per-star properties.
    sub_y = rng.uniform(-0.5, 0.5, num_stars)
    sub_x = rng.uniform(-0.5, 0.5, num_stars)
    fluxes = base_flux * 10 ** (-0.4 * rng.uniform(-1.0, 2.0, num_stars))
    colors = rng.uniform(0.6, 1.4, num_stars)  # B/R response ratio proxy

    pscs = np.empty((num_stars, num_frames, height * width))
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0

    for star in range(num_stars):
        response = np.ones(stamp_shape)
        response[rgb["r"]] = 1.0 / colors[star]
        response[rgb["b"]] = colors[star]
        response[rgb["g"]] = 1.0

        for frame in range(num_frames):
            psf = gaussian_psf(
                stamp_shape,
                (center_y + sub_y[star] + drift_y[frame], center_x + sub_x[star] + drift_x[frame]),
                sigma,
            )
            counts = fluxes[star] * psf * response + background
            if noise:
                counts = rng.poisson(np.clip(counts, 0, None)).astype(float)
                counts += rng.normal(0.0, read_noise, stamp_shape)
            pscs[star, frame] = (counts - background).ravel()

    return SyntheticObservation(
        pscs=pscs,
        times=times,
        stamp_shape=stamp_shape,
        fluxes=fluxes,
        colors=colors,
    )
