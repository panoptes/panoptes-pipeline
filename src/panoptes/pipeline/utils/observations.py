import numpy as np
import pandas
from loguru import logger
from numpy import typing as npt


def make_stamps(
    stamp_positions: pandas.DataFrame,
    data: npt.ArrayLike,
) -> pandas.DataFrame:
    """Cut one stamp per row of `stamp_positions` out of `data`.

    Returns a frame indexed on `picid` with one `pixel_NNN` column per pixel,
    row-major over ``data[stamp_y_min:stamp_y_max, stamp_x_min:stamp_x_max]``,
    so the flat axis unpacks as ``(height, width)``.

    Every stamp in a cube has to be the same size, so the size is taken from
    the median extent of the positions and any row whose slice would come out
    a different size -- which only happens at the frame edge -- is dropped with
    a warning rather than padded. Padding would put pixels in the cube that
    were never observed.

    This is one gather over the whole frame rather than a slice per source.
    The per-source loop it replaces built and concatenated a one-row DataFrame
    per star, which at a few thousand stars per frame was most of the cost of
    extracting an observation.
    """
    if len(stamp_positions) == 0:
        return pandas.DataFrame()

    data = np.asarray(data)
    height, width = data.shape

    stamp_width = int(stamp_positions.stamp_x_max.median() - stamp_positions.stamp_x_min.median())
    stamp_height = int(stamp_positions.stamp_y_max.median() - stamp_positions.stamp_y_min.median())
    total_stamp_size = stamp_width * stamp_height
    logger.debug(
        f"Making stamps of {total_stamp_size=} for {len(stamp_positions)} sources from data {data.shape}"
    )

    y_min = stamp_positions.stamp_y_min.to_numpy(dtype=int)
    x_min = stamp_positions.stamp_x_min.to_numpy(dtype=int)
    y_max = stamp_positions.stamp_y_max.to_numpy(dtype=int)
    x_max = stamp_positions.stamp_x_max.to_numpy(dtype=int)

    # A slice is the right size only if it is the nominal size *and* lies
    # inside the frame; numpy would silently clip the latter.
    usable = (
        (y_max - y_min == stamp_height)
        & (x_max - x_min == stamp_width)
        & (y_min >= 0)
        & (x_min >= 0)
        & (y_max <= height)
        & (x_max <= width)
    )
    if not usable.all():
        bad = stamp_positions.index[~usable].tolist()
        logger.warning(f"Dropping {len(bad)} stamp(s) that do not fit the frame: {bad[:10]}")

    y_min, x_min = y_min[usable], x_min[usable]
    rows = y_min[:, None, None] + np.arange(stamp_height)[None, :, None]
    cols = x_min[:, None, None] + np.arange(stamp_width)[None, None, :]
    stamps = data[rows, cols].reshape(len(y_min), total_stamp_size)

    psc_data = pandas.DataFrame(
        stamps,
        index=stamp_positions.index[usable].rename("picid"),
        columns=[f"pixel_{i:03d}" for i in range(total_stamp_size)],
    )
    return psc_data.sort_index()
