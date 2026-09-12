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


def infer_green_offsets(
    sky_stamps: np.ndarray,
    stamp_shape: tuple[int, int],
) -> set[tuple[int, int]]:
    """Find which two of the four quad positions are green, from the data.

    The two green pixels of a Bayer quad sit behind the same filter, so their
    sky levels match each other and differ from red and blue. That identifies
    the pattern family without trusting any convention -- which matters, because
    getting it wrong silently mixes red and blue into one "green" channel and
    every per-channel result is then mislabeled.

    Args:
        sky_stamps: ``(k, n)`` or ``(k, m, n)`` of star-free stamps. The faintest
            sources in a sequence work well; their stamps are effectively sky.
        stamp_shape: ``(height, width)``.

    Returns:
        The two ``(row % 2, col % 2)`` offsets that are green.
    """
    height, width = stamp_shape
    flat = np.asarray(sky_stamps, dtype=float).reshape(-1, height * width)
    grid = flat.reshape(-1, height, width)

    levels = {
        (dy, dx): float(np.median(grid[:, dy::2, dx::2]))
        for dy in (0, 1)
        for dx in (0, 1)
    }
    diagonal = abs(levels[(0, 0)] - levels[(1, 1)])
    anti = abs(levels[(0, 1)] - levels[(1, 0)])
    return {(0, 0), (1, 1)} if diagonal < anti else {(0, 1), (1, 0)}


def infer_pattern(
    sky_stamps: np.ndarray,
    stamp_shape: tuple[int, int],
    red_offset: tuple[int, int] | None = None,
) -> str:
    """Infer the Bayer pattern of stored data.

    Green positions are recoverable from sky levels alone (see
    :func:`infer_green_offsets`). Telling red from blue is not, reliably, so
    pass ``red_offset`` -- the FITS header's ``MEASRGGB`` white-balance values,
    which ``extract_metadata`` already parses, settle it.

    Raises:
        ValueError: if red and blue cannot be distinguished and no
            ``red_offset`` was given. Guessing here silently swaps two channels.
    """
    greens = infer_green_offsets(sky_stamps, stamp_shape)
    others = sorted({(0, 0), (0, 1), (1, 0), (1, 1)} - greens)

    if red_offset is None:
        raise ValueError(
            f"Greens are at {sorted(greens)}, so red and blue are at {others}, but which is "
            "which cannot be read from sky levels. Pass red_offset (from the MEASRGGB header) "
            "rather than guessing -- a wrong choice swaps the red and blue lightcurves."
        )
    if tuple(red_offset) not in others:
        raise ValueError(
            f"red_offset {tuple(red_offset)} is a green position; expected one of {others}"
        )

    blue_offset = [o for o in others if o != tuple(red_offset)][0]
    layout = {tuple(red_offset): RED, blue_offset: BLUE}
    for green in greens:
        layout[green] = GREEN

    block = ((layout[(0, 0)], layout[(0, 1)]), (layout[(1, 0)], layout[(1, 1)]))
    for name, candidate in PATTERNS.items():
        if tuple(tuple(row) for row in candidate) == block:
            return name
    raise ValueError(f"Inferred layout {block} matches no known pattern")


def circular_aperture(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
) -> np.ndarray:
    """Boolean selection mask for pixels whose centers lie within ``radius``.

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

    Returns weights normalized so that ``sum(w * profile) == 1``, i.e. applying
    them to a frame recovers total flux rather than a weighted average. This is
    the standard optimal-extraction weighting and is the drop-in replacement for
    a hard-edged aperture in improvement plan 3.6.
    """
    profile = np.asarray(profile, dtype=float)
    weights = profile / np.asarray(noise_variance, dtype=float)
    norm = float((weights * profile).sum())
    if norm == 0:
        raise ValueError("Degenerate profile: weights sum to zero")
    return weights / norm
