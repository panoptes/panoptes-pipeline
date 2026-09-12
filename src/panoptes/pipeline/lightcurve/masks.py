"""Bayer CFA masks and apertures for postage stamp cubes (PSCs).

Conventions used throughout this package
----------------------------------------
A *selection* mask is a boolean array where ``True`` means "this pixel belongs
to the thing being selected" (a color channel, an aperture). This is the
opposite of the ``numpy.ma`` convention, where ``True`` means "masked out".

The legacy notebook code mixed the two conventions silently. Every function
here takes and returns selection masks; convert at the ``numpy.ma`` boundary
with ``~mask`` and nowhere else.

Bayer phase
-----------
The color of a pixel depends on its *absolute* position in the sensor, not its
position within a stamp. A stamp only has a well-defined color layout if its
origin is superpixel-aligned (both coordinates even). ``bayer_color_index``
takes an explicit ``origin`` so this is never assumed.
"""

from __future__ import annotations

import numpy as np

# Index values used in the color-index array returned by `bayer_color_index`.
RED, GREEN, BLUE = 0, 1, 2
COLOR_NAMES = ("r", "g", "b")

# Which color sits at each (row % 2, col % 2) position, for each supported
# pattern. The name gives the 2x2 block read left-to-right, top-to-bottom,
# with the origin at array index [0, 0].
PATTERNS = {
    "RGGB": ((RED, GREEN), (GREEN, BLUE)),
    "BGGR": ((BLUE, GREEN), (GREEN, RED)),
    "GRBG": ((GREEN, RED), (BLUE, GREEN)),
    "GBRG": ((GREEN, BLUE), (RED, GREEN)),
}


def bayer_color_index(
    shape: tuple[int, int],
    origin: tuple[int, int] = (0, 0),
    pattern: str = "RGGB",
) -> np.ndarray:
    """Return an int array giving the Bayer color (RED/GREEN/BLUE) of each pixel.

    Args:
        shape: ``(height, width)`` of the stamp.
        origin: ``(y0, x0)``, the absolute sensor position of stamp pixel
            ``[0, 0]``. Only the parity matters.
        pattern: Bayer pattern of the sensor at absolute ``(0, 0)``.

    Returns:
        Array of shape ``shape`` containing RED, GREEN or BLUE.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown Bayer {pattern=}; expected one of {sorted(PATTERNS)}")

    block = np.asarray(PATTERNS[pattern])
    height, width = shape
    y0, x0 = origin

    rows = (np.arange(height) + y0) % 2
    cols = (np.arange(width) + x0) % 2
    return block[rows[:, None], cols[None, :]]


def rgb_masks(
    shape: tuple[int, int],
    origin: tuple[int, int] = (0, 0),
    pattern: str = "RGGB",
) -> dict[str, np.ndarray]:
    """Return per-color boolean *selection* masks for a stamp.

    ``True`` means the pixel belongs to that color channel. Green selects two
    pixels per superpixel; red and blue select one each.
    """
    colors = bayer_color_index(shape, origin=origin, pattern=pattern)
    return {name: colors == value for name, value in zip(COLOR_NAMES, (RED, GREEN, BLUE))}


def is_superpixel_aligned(origin: tuple[int, int], shape: tuple[int, int]) -> bool:
    """True if a stamp has an even origin and an even size in both axes.

    A stamp that fails this test cannot be compared pixel-for-pixel against
    another stamp, because the two carry different Bayer phases. See
    conformance audit 4.3.
    """
    y0, x0 = origin
    height, width = shape
    return (y0 % 2 == 0) and (x0 % 2 == 0) and (height % 2 == 0) and (width % 2 == 0)


def circular_aperture(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
) -> np.ndarray:
    """Boolean selection mask for pixels whose centres lie within ``radius``.

    Args:
        shape: ``(height, width)`` of the stamp.
        center: ``(y, x)`` in stamp-local pixel coordinates.
        radius: Radius in pixels.
    """
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    cy, cx = center
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2


def grow_to_superpixels(mask: np.ndarray) -> np.ndarray:
    """Expand a selection mask so every touched 2x2 superpixel is fully included.

    Apertures that clip a superpixel change the R:G:B pixel ratio inside the
    aperture, which biases the per-color fluxes. Growing to whole superpixels
    keeps the ratio fixed at 1:2:1 regardless of where the aperture lands.
    """
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    if height % 2 or width % 2:
        raise ValueError(f"Stamp {mask.shape=} must have even dimensions to grow to superpixels")

    blocks = mask.reshape(height // 2, 2, width // 2, 2)
    touched = blocks.any(axis=(1, 3))
    return np.repeat(np.repeat(touched, 2, axis=0), 2, axis=1)


def weighted_aperture(profile: np.ndarray, noise_variance: np.ndarray | float) -> np.ndarray:
    """Optimal (inverse-variance) photometric weights for a known PSF profile.

    Returns weights normalised so that ``sum(w * profile) == 1``, i.e. applying
    them to a frame recovers total flux rather than a weighted average. This is
    the standard optimal-extraction weighting and is the drop-in replacement for
    a hard-edged aperture in improvement plan 3.4.
    """
    profile = np.asarray(profile, dtype=float)
    weights = profile / np.asarray(noise_variance, dtype=float)
    norm = float((weights * profile).sum())
    if norm == 0:
        raise ValueError("Degenerate profile: weights sum to zero")
    return weights / norm
