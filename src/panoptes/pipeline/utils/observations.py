import pandas
import pandas as pd
from loguru import logger
from numpy import typing as npt


def make_stamps(
    stamp_positions: pandas.DataFrame,
    data: npt.DTypeLike,
) -> pandas.DataFrame:
    """Make stamps from the data."""
    if len(stamp_positions) == 0:
        return pd.DataFrame()

    stamp_width = int(stamp_positions.stamp_x_max.median() - stamp_positions.stamp_x_min.median())
    stamp_height = int(stamp_positions.stamp_y_max.median() - stamp_positions.stamp_y_min.median())

    total_stamp_size = int(stamp_width * stamp_height)
    logger.debug(
        f"Making stamps of {total_stamp_size=} for {len(stamp_positions)} sources from data {data.shape}"
    )

    stamps = []
    for picid, row in stamp_positions.iterrows():
        # Get the stamp data.
        row_slice = slice(int(row.stamp_y_min), int(row.stamp_y_max))
        col_slice = slice(int(row.stamp_x_min), int(row.stamp_x_max))
        psc0 = data[row_slice, col_slice].reshape(-1)

        # Make sure stamp is correct size (errors at edges).
        if psc0.shape == (total_stamp_size,):
            stamp = pd.DataFrame(psc0).T
            stamp.columns = [f"pixel_{i:03d}" for i in range(total_stamp_size)]
            stamp["picid"] = picid
            stamp.set_index(["picid"], inplace=True)
            stamps.append(stamp)
        else:
            print(f"Bad stamp size for {picid=} {psc0.shape=} {total_stamp_size=}")

    # Make one dataframe.
    if len(stamps) > 0:
        psc_data = pd.concat(stamps).sort_index()
    else:
        psc_data = pd.DataFrame()

    return psc_data
