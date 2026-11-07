#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "astropy",
# ]
# ///
"""Write camera registry profiles from a table of `measure_gain.py` results.

The registry is data, not code: it lives under a root outside this repository,
at ``units/{unit_id}/cameras/{uid}.json``, and the pipeline reads it from
``CameraSettings.registry_dir`` (data contract 6.2). This turns measurements
into profiles in that tree.

Input is one row per camera, with the columns of ``plans/figures/gain_units.csv``
-- ``unit``, ``camera``, ``sensor``, ``black_assumed``, ``black_intercept``,
``clip``, ``whtlvln``, ``num_sequences``, the four ``gain_<channel>`` fits,
``read_noise_e``, ``clean_gain``, ``clean_range`` and ``sequences``. The rules:

- **Black** is the ``--black`` given to `measure_gain.py`, recorded as the
  body's value, with the free intercept beside it as the check.
- **Clip** is the histogram clip. 13584 is written as 13583: some frames round
  the raw conversion up, which is not headroom.
- **Scaled raw**: a 2048-black body that clips at 16383 has had its values
  scaled before the 14-bit container clipped them (PAN019, data contract 6.2).
  Its clip, gain and read noise depend on the scaling, so the profile gets
  black and a warning and nothing else.
- **Gain** is the sensor family's value from `FAMILY_GAIN`, not the camera's own
  fit, which goes in the note: the per-camera spread is within the method's
  ~10% floor. A sensor with no family value gets no gain rather than a guess.
- **Read noise** is the camera's own intercept. A non-positive one means the
  intercept sat above black, which measures nothing; the profile records no
  read noise and a note saying why, never a zero.

Existing profiles are never overwritten unless ``--force`` is given: the tree is
state that people edit by hand, and a note added there is worth more than a
regenerated one. Without ``--force`` a differing profile is reported and left.

Usage::

    uv run scripts/write_registry.py plans/figures/gain_units.csv --registry ~/panoptes/registry \\
        --images ~/panoptes/images
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

#: Sky photon-transfer gain by sensor family, e-/ADU, with one-sigma: data
#: contract 6.2. Lower bounds, to be replaced by flat-field photon transfer.
FAMILY_GAIN: dict[str, tuple[float, float, str]] = {
    "5208x3476": (2.3, 0.2, "EOS 100D class"),
    "6020x4016": (2.1, 0.2, "24 MP class at a 13583 clip"),
    "6024x4020": (2.3, 0.1, "PAN007 class"),
}

GAIN_CAVEAT = (
    "Sky photon transfer is a lower bound and its method floor is ~10%; "
    "replace with a flat-field photon transfer (panoptes/panoptes-pipeline#99)."
)

SCALED_WARNING = (
    "Raw values are digitally scaled above black: this 2048-black body clips at 16383, not 13583 "
    "(PAN019 measured a 1.61 scale, with a comb of empty codes in the histogram and ISO 100 in the "
    "header). Clip, gain and read noise are left out until the scaling is understood. See data "
    "contract 6.2."
)

CHANNELS = ("gain_G2", "gain_B", "gain_R", "gain_G1")


def is_scaled(black: int, clip: int) -> bool:
    """A 2048-black Canon body clips at 13583 at base ISO; 16383 means scaled."""
    return black == 2048 and clip >= 16383


def profile(row: dict[str, str], serial: str | None = None) -> dict:
    """The registry profile for one row of measurements."""
    unit, uid, sensor = row["unit"], row["camera"], row["sensor"]
    black = int(row["black_assumed"])
    clip = int(row["clip"])
    num_sequences = int(row["num_sequences"])

    document: dict = dict(
        unit_id=unit,
        camera_uid=uid,
        serial_number=serial,
        sensor=sensor,
        iso=int(row.get("iso") or 100),
        black_level=dict(
            value=black,
            method="body",
            note=(
                f"Known black of this body type; the free photon-transfer intercept reads "
                f"{row['black_intercept']}, black minus the read-noise term."
            ),
        ),
    )

    if is_scaled(black, clip):
        document["warning"] = SCALED_WARNING
        return document

    if clip == 13584:
        clip = 13583
    document["clip"] = dict(
        value=clip,
        method="histogram",
        note=f"Raw value the histogram piles up at; WHTLVLN reads {row['whtlvln']}.",
    )

    gains = [float(row[c]) for c in CHANNELS]
    own = f"own shared fit {min(gains):.2f}-{max(gains):.2f} per channel over {num_sequences} sequence(s)"
    if row.get("clean_gain"):
        own += f", clean per-pair median {float(row['clean_gain']):.2f} ({row['clean_range']})"
    else:
        own += ", no clean sequence in the sample"

    if sensor in FAMILY_GAIN:
        value, uncertainty, family = FAMILY_GAIN[sensor]
        note = f"{family} value. This camera: {own}. {GAIN_CAVEAT}"
        if num_sequences < 2:
            note += " Only one dark sequence per camera in the archive, so the camera's own number is weak."
        document["gain"] = dict(value=value, uncertainty=uncertainty, method="family", note=note)

    read_noise = float(row["read_noise_e"])
    if read_noise > 0:
        document["read_noise"] = dict(
            value=round(read_noise, 1),
            method="sky-ptc",
            note="Photon-transfer intercept with black fixed; indicative to a few e-.",
        )
    else:
        document["note"] = (
            f"No read noise: the photon-transfer intercept ({row['black_intercept']} ADU) lies above "
            f"the {black} black, so the fit implies a negative read-noise term and measures nothing. "
            "Left unknown rather than recorded as zero; a flat-field photon transfer "
            "(panoptes/panoptes-pipeline#99) or a bias frame would measure it."
        )

    return document


def read_serial(images: Path, row: dict[str, str]) -> str | None:
    """``CAMSN`` from the first frame of each measured sequence, if they agree."""
    from astropy.io import fits

    serials = set()
    for sequence in row["sequences"].split():
        frames = sorted(images.glob(f"{row['unit']}/**/{row['camera']}/{sequence}/*.fits*"))
        if frames:
            header = fits.getheader(frames[0], ext=1 if frames[0].suffix == ".fz" else 0)
            if (serial := header.get("CAMSN")) is not None:
                serials.add(str(serial))
    if len(serials) > 1:
        camera = f"{row['unit']}/{row['camera']}"
        raise SystemExit(f"{camera}: CAMSN disagrees across sequences: {sorted(serials)}")
    return serials.pop() if serials else None


def write(document: dict, registry: Path, force: bool = False) -> str:
    """Write `document` under `registry`; return what happened."""
    path = registry / "units" / document["unit_id"] / "cameras" / f"{document['camera_uid']}.json"
    text = json.dumps(document, indent=2) + "\n"
    if path.exists():
        if json.loads(path.read_text()) == document:
            return f"unchanged {path}"
        if not force:
            return f"DIFFERS, left as is (use --force) {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return f"wrote {path}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("table", type=Path, help="CSV of measure_gain.py results, one row per camera")
    parser.add_argument("--registry", type=Path, required=True, help="registry root to write into")
    parser.add_argument("--images", type=Path, help="raw tree to read CAMSN from, as a cross-reference")
    parser.add_argument("--force", action="store_true", help="overwrite profiles that differ")
    args = parser.parse_args(argv)

    with args.table.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        serial = read_serial(args.images, row) if args.images else None
        print(write(profile(row, serial), args.registry, force=args.force))
    return 0


if __name__ == "__main__":
    sys.exit(main())
