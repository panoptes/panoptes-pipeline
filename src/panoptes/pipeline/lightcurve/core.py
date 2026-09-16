"""Reference implementation of the PANOPTES differential photometry algorithm.

This module is a direct, I/O-free transcription of Gee et al., *On-sky
Demonstration of Precision Photometry with Bayer Color Filter Arrays*,
section 3.2. Equation numbers in the docstrings refer to that paper.

Everything here operates on plain numpy arrays. There is no pandas, no HDF5,
no cloud client and no plotting, so every step is unit-testable in isolation
and can be swapped independently. See improvement plan 2.1 for why.

Array shapes
------------
A postage stamp cube (PSC) for a single source is ``(m, n)``: ``m`` frames by
``n`` pixels, with the two spatial axes flattened into ``j`` exactly as in the
paper. A collection of ``r`` reference PSCs is ``(r, m, n)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

SolveMethod = Literal["ols", "ridge", "lasso", "nnls", "lasso_positive"]


# --------------------------------------------------------------------------
# Preprocessing (precedes paper 3.2.1)
# --------------------------------------------------------------------------


def subtract_stamp_sky(
    psc: np.ndarray,
    sky_mask: np.ndarray,
    channel_masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Remove a per-frame, per-color sky pedestal estimated from the stamp itself.

    **This is a stopgap, not the right fix.** Paper section 4.2 subtracts a
    global three-channel background from the full frame before stamps are cut,
    and that is where it belongs. The current pipeline computes that background
    and then discards it (conformance audit 5.0), so stamps stored in
    ``observation.h5`` still carry bias plus sky -- typically 780 ADU per pixel,
    which for a median source is over 95% of the stamp sum.

    Any precision or transit depth measured on un-subtracted stamps is diluted
    by that pedestal and is not a photometric result. Use this to get usable
    numbers out of already-processed observations; fix ``ProcessFITS.ipynb`` to
    stop needing it.

    Args:
        psc: ``(m, n)`` or ``(r, m, n)`` stamps.
        sky_mask: ``(n,)`` selection mask of pixels taken to be sky -- typically
            everything outside the central few pixels.
        channel_masks: Optional ``{name: (n,) mask}``. Each color gets its own
            sky level, which matters because the Bayer channels have different
            responses and the sky is not gray.

    Returns:
        Array of the same shape, pedestal removed.
    """
    psc = np.asarray(psc, dtype=float)
    sky_mask = np.asarray(sky_mask, dtype=bool)

    if not sky_mask.any():
        raise ValueError("sky_mask selects no pixels")

    if channel_masks is None:
        channel_masks = {"all": np.ones(psc.shape[-1], dtype=bool)}

    out = psc.copy()
    for mask in channel_masks.values():
        mask = np.asarray(mask, dtype=bool)
        sky_pixels = mask & sky_mask
        if not sky_pixels.any():
            continue
        level = np.median(psc[..., sky_pixels], axis=-1, keepdims=True)
        out[..., mask] = psc[..., mask] - level

    return out


def central_sky_mask(stamp_shape: tuple[int, int], core_radius: float = 3.0) -> np.ndarray:
    """Flat selection mask of pixels outside ``core_radius`` of the stamp center."""
    height, width = stamp_shape
    yy, xx = np.mgrid[:height, :width]
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    outside = (np.abs(yy - center_y) >= core_radius) | (np.abs(xx - center_x) >= core_radius)
    return outside.ravel()


# --------------------------------------------------------------------------
# 3.2.1 Prepare PSCs
# --------------------------------------------------------------------------


def normalize_psc(psc: np.ndarray, pixel_mask: np.ndarray | None = None) -> np.ndarray:
    """Flux-marginalize a PSC, per frame (paper Eq. 1).

    Each frame is divided by its own summed flux, leaving only the stellar
    *morphology* as projected onto the Bayer pattern.

    Args:
        psc: ``(m, n)`` or ``(r, m, n)`` array of raw (background-subtracted)
            pixel values.
        pixel_mask: Optional boolean selection mask of length ``n``. When given,
            the normalization sum is taken over the selected pixels only and
            unselected pixels are set to zero. Use this to normalize within a
            single color channel.

    Returns:
        Array with the same shape as ``psc``. Frames whose sum is zero or
        non-finite are returned as all-NaN so they can be dropped downstream
        rather than silently contributing zeros.
    """
    psc = np.asarray(psc, dtype=float)

    if pixel_mask is not None:
        pixel_mask = np.asarray(pixel_mask, dtype=bool)
        psc = np.where(pixel_mask, psc, 0.0)

    totals = psc.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = psc / totals
    bad = ~np.isfinite(totals) | (totals == 0)
    return np.where(bad, np.nan, out)


# --------------------------------------------------------------------------
# 3.2.2 Find reference stars
# --------------------------------------------------------------------------


def similarity_scores(
    target_norm: np.ndarray,
    refs_norm: np.ndarray,
    frame_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Summed-squared morphological difference to the target (paper Eq. 2).

    Lower is more similar. A star compared against itself scores exactly zero.

    Args:
        target_norm: ``(m, n)`` normalized target PSC.
        refs_norm: ``(r, m, n)`` normalized reference PSCs.
        frame_weights: Optional ``(m,)`` non-negative weights. Frames with a
            weight of zero are excluded, which is how out-of-transit-only
            reference selection is done.

    Returns:
        ``(r,)`` array of scores. References with no usable frames score
        ``inf`` so they sort to the back.
    """
    target_norm = np.asarray(target_norm, dtype=float)
    refs_norm = np.asarray(refs_norm, dtype=float)

    diff_sq = (refs_norm - target_norm[None, ...]) ** 2
    valid = np.isfinite(diff_sq)
    diff_sq = np.where(valid, diff_sq, 0.0)

    if frame_weights is None:
        scores = diff_sq.sum(axis=(1, 2))
        used = valid.any(axis=2).sum(axis=1)
    else:
        frame_weights = np.asarray(frame_weights, dtype=float)
        scores = (diff_sq.sum(axis=2) * frame_weights[None, :]).sum(axis=1)
        used = ((valid.any(axis=2)) * (frame_weights > 0)[None, :]).sum(axis=1)

    return np.where(used > 0, scores, np.inf)


def select_references(scores: np.ndarray, num_refs: int = 100) -> np.ndarray:
    """Indices of the ``num_refs`` most similar references, best first.

    The caller is responsible for having excluded the target from ``scores``.
    Passing a score array that still contains the target is the defect
    described in conformance audit 3.4, so this function refuses a perfect
    zero score.
    """
    scores = np.asarray(scores, dtype=float)
    if np.any(scores == 0):
        raise ValueError(
            "A reference scored exactly 0.0, which means the target is present in its own "
            "reference pool. Remove it before calling select_references()."
        )
    order = np.argsort(scores, kind="stable")
    order = order[np.isfinite(scores[order])]
    return order[:num_refs]


# --------------------------------------------------------------------------
# 3.2.3 Determine coefficients for the comparison star
# --------------------------------------------------------------------------


def solve_coefficients(
    target_norm: np.ndarray,
    refs_norm: np.ndarray,
    method: SolveMethod = "lasso",
    alpha: float = 1e-5,
    pixel_mask: np.ndarray | None = None,
    frame_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Least-squares coefficients for the synthetic comparison star (paper Eq. 4).

    One coefficient per reference, fit jointly across every frame and pixel.

    The paper notes that "in practice a regularization term can also [be]
    applied" and its Figure 7 shows 46 of 100 coefficients driven to exactly
    zero -- a signature of an L1 penalty, not of plain least squares. Plain
    ``ols`` is therefore *not* the published configuration; it is provided as
    the unregularized baseline to measure against.

    Args:
        target_norm: ``(m, n)`` normalized target PSC.
        refs_norm: ``(r, m, n)`` normalized reference PSCs.
        method: One of ``ols``, ``ridge``, ``lasso``, ``lasso_positive``, ``nnls``.
        alpha: Regularization strength (ignored for ``ols`` and ``nnls``).
        pixel_mask: Optional ``(n,)`` selection mask restricting the fit to
            certain pixels, e.g. a single color channel.
        frame_weights: Optional ``(m,)`` non-negative weights.

    Returns:
        ``(r,)`` coefficient vector.
    """
    target_norm = np.asarray(target_norm, dtype=float)
    refs_norm = np.asarray(refs_norm, dtype=float)
    num_refs = refs_norm.shape[0]

    design, rhs = _build_design_matrix(target_norm, refs_norm, pixel_mask, frame_weights)

    if design.shape[0] == 0:
        raise ValueError("No finite (frame, pixel) samples available to fit coefficients")

    if method == "ols":
        from scipy import linalg

        coeffs, *_ = linalg.lstsq(design, rhs)
    elif method == "nnls":
        from scipy.optimize import nnls

        coeffs, _ = nnls(design, rhs)
    elif method in ("ridge", "lasso", "lasso_positive"):
        from sklearn import linear_model

        if method == "ridge":
            model = linear_model.Ridge(alpha=alpha, fit_intercept=False)
        else:
            model = linear_model.Lasso(
                alpha=alpha,
                fit_intercept=False,
                positive=(method == "lasso_positive"),
                max_iter=20_000,
            )
        model.fit(design, rhs)
        coeffs = np.asarray(model.coef_, dtype=float)
    else:
        raise ValueError(f"Unknown solve {method=}")

    return coeffs.reshape(num_refs)


def _build_design_matrix(
    target_norm: np.ndarray,
    refs_norm: np.ndarray,
    pixel_mask: np.ndarray | None,
    frame_weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (frame, pixel) into rows of a design matrix, dropping bad samples."""
    num_refs, num_frames, num_pixels = refs_norm.shape

    sample_weight = np.ones((num_frames, num_pixels), dtype=float)
    if frame_weights is not None:
        sample_weight *= np.asarray(frame_weights, dtype=float)[:, None]
    if pixel_mask is not None:
        sample_weight *= np.asarray(pixel_mask, dtype=bool)[None, :]

    finite = np.isfinite(target_norm) & np.isfinite(refs_norm).all(axis=0)
    keep = finite & (sample_weight > 0)

    # sqrt-weighting turns a weighted least-squares problem into an ordinary one.
    root_w = np.sqrt(sample_weight[keep])
    design = (refs_norm[:, keep] * root_w[None, :]).T
    rhs = target_norm[keep] * root_w
    return design, rhs


# --------------------------------------------------------------------------
# 3.2.4 Build the comparison star
# --------------------------------------------------------------------------


def build_comparison(refs_raw: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Apply normalized-space coefficients to the raw references (paper Eq. 5).

    The coefficients come from the flux-marginalized fit; they are applied to
    the flux-carrying stamps. That asymmetry is the heart of the method: the
    comparison star is built without ever seeing the target's flux, so a real
    brightness change in the target cannot leak into its own comparison.

    Args:
        refs_raw: ``(r, m, n)`` raw reference PSCs.
        coefficients: ``(r,)`` coefficient vector.

    Returns:
        ``(m, n)`` synthetic comparison PSC.
    """
    refs_raw = np.asarray(refs_raw, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape[0] != refs_raw.shape[0]:
        raise ValueError(
            f"Have {coefficients.shape[0]} coefficients for {refs_raw.shape[0]} references"
        )
    return np.tensordot(coefficients, refs_raw, axes=(0, 0))


# --------------------------------------------------------------------------
# 3.2.5 Perform differential photometry
# --------------------------------------------------------------------------


def differential_lightcurve(
    target_raw: np.ndarray,
    comparison: np.ndarray,
    pixel_mask: np.ndarray | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Ratio of summed target flux to summed comparison flux (paper Eq. 6).

    The ratio is already a relative flux: the comparison ensemble sets the
    scale. It does not need, and must not be given, a further normalization to
    unit median.

    ``normalize`` defaults to **False**, which differs from paper Eq. 6 and is
    deliberate. PANOPTES targets long-period planets, whose transits can last
    longer than a single night's observation, and the survey's goal is to stitch
    partial lightcurves from units at different longitudes into one event.
    Dividing by the median of a window that is wholly or partly in transit
    subtracts the signal from itself: a fully in-transit segment normalizes to a
    flat line at 1.0 and the depth is gone. See algorithm design 6.

    Set ``normalize=True`` only for display, or when the window is known to be
    dominated by out-of-transit baseline and nothing downstream will stitch it.

    Args:
        target_raw: ``(m, n)`` raw target PSC.
        comparison: ``(m, n)`` synthetic comparison PSC.
        pixel_mask: Optional ``(n,)`` aperture / color selection mask. When
            omitted the whole stamp is summed, which is what the legacy
            notebook did (conformance audit 3.5).
        normalize: Divide by the median. Destroys absolute depth; see above.

    Returns:
        ``(m,)`` flux relative to the comparison ensemble.
    """
    target_raw = np.asarray(target_raw, dtype=float)
    comparison = np.asarray(comparison, dtype=float)

    if pixel_mask is not None:
        pixel_mask = np.asarray(pixel_mask, dtype=bool)
        target_raw = np.where(pixel_mask, target_raw, 0.0)
        comparison = np.where(pixel_mask, comparison, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        flux = target_raw.sum(axis=1) / comparison.sum(axis=1)

    if normalize:
        median = np.nanmedian(flux)
        if np.isfinite(median) and median != 0:
            flux = flux / median

    return flux


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class LightcurveResult:
    """Everything needed to reproduce, plot and audit one differential lightcurve."""

    flux: np.ndarray
    """``(m,)`` relative flux, median-normalized to unity."""

    comparison: np.ndarray
    """``(m, n)`` synthetic comparison PSC."""

    coefficients: np.ndarray
    """``(r,)`` fitted coefficients, aligned with ``reference_indices``."""

    reference_indices: np.ndarray
    """``(r,)`` indices into the reference pool, ordered best-match first."""

    scores: np.ndarray
    """``(r,)`` similarity scores of the selected references."""

    channel: str = "all"
    """Which pixels the photometry summed: ``all``, ``r``, ``g`` or ``b``."""

    meta: dict = field(default_factory=dict)
    """Free-form record of the settings used, for provenance."""

    @property
    def num_active_references(self) -> int:
        """How many coefficients survived regularization (paper Fig. 7: 46 of 100)."""
        return int(np.count_nonzero(self.coefficients))


def make_lightcurve(
    target_psc: np.ndarray,
    reference_pool: np.ndarray,
    num_refs: int = 100,
    method: SolveMethod = "lasso",
    alpha: float = 1e-5,
    aperture_mask: np.ndarray | None = None,
    channel_mask: np.ndarray | None = None,
    channel: str = "all",
    select_on_channel: bool = False,
    frame_weights: np.ndarray | None = None,
) -> LightcurveResult:
    """Run paper sections 3.2.1 through 3.2.5 end to end.

    Args:
        target_psc: ``(m, n)`` raw target PSC.
        reference_pool: ``(p, m, n)`` raw candidate reference PSCs. Must not
            contain the target.
        num_refs: How many references to carry into the coefficient fit.
        method: Coefficient solver, see :func:`solve_coefficients`.
        alpha: Regularization strength.
        aperture_mask: ``(n,)`` selection mask for the final photometry.
        channel_mask: ``(n,)`` color-channel selection mask for the final
            photometry, combined with ``aperture_mask`` by logical AND.
        channel: Label recorded on the result.
        select_on_channel: If True, run selection and the coefficient fit using
            only the channel's pixels rather than the whole stamp. The paper
            does selection on all pixels and splits color only at the last
            step; this flag exists to test the alternative (improvement plan 3.5).
        frame_weights: Optional ``(m,)`` weights, e.g. zero for in-transit
            frames so the comparison is built from out-of-transit data only.

    Returns:
        A :class:`LightcurveResult`.
    """
    target_psc = np.asarray(target_psc, dtype=float)
    reference_pool = np.asarray(reference_pool, dtype=float)

    fit_mask = channel_mask if (select_on_channel and channel_mask is not None) else None

    target_norm = normalize_psc(target_psc, pixel_mask=fit_mask)
    pool_norm = normalize_psc(reference_pool, pixel_mask=fit_mask)

    all_scores = similarity_scores(target_norm, pool_norm, frame_weights=frame_weights)
    chosen = select_references(all_scores, num_refs=num_refs)

    coefficients = solve_coefficients(
        target_norm,
        pool_norm[chosen],
        method=method,
        alpha=alpha,
        pixel_mask=fit_mask,
        frame_weights=frame_weights,
    )

    comparison = build_comparison(reference_pool[chosen], coefficients)

    photometry_mask = _combine_masks(aperture_mask, channel_mask)
    flux = differential_lightcurve(target_psc, comparison, pixel_mask=photometry_mask)

    return LightcurveResult(
        flux=flux,
        comparison=comparison,
        coefficients=coefficients,
        reference_indices=chosen,
        scores=all_scores[chosen],
        channel=channel,
        meta=dict(
            num_refs=num_refs,
            method=method,
            alpha=alpha,
            select_on_channel=select_on_channel,
            pool_size=int(reference_pool.shape[0]),
            num_frames=int(target_psc.shape[0]),
            num_pixels=int(target_psc.shape[1]),
        ),
    )


def _combine_masks(*masks: np.ndarray | None) -> np.ndarray | None:
    """Logical-AND any number of optional selection masks."""
    present = [np.asarray(m, dtype=bool) for m in masks if m is not None]
    if not present:
        return None
    combined = present[0]
    for mask in present[1:]:
        combined = combined & mask
    return combined
