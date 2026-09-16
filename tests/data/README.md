# Test frames

Real POCS-written FITS files, copied from [POCS](https://github.com/panoptes/POCS)
`tests/data/` at commit `7370af45`. They are used rather than synthesized
because the thing under test is agreement with a header vocabulary this
repository does not control -- a fixture written from the pipeline's own
assumptions would agree with them by construction and prove nothing.

| File | Cards | Shape | What it is |
|---|---|---|---|
| `tiny.fits` | 45 | 200x200 | A raw frame. `CAMSN`, `WHTLVLN`/`WHTLVLS`, `SEQID`, `IMAGEID`, no WCS, no `IMAGEW`/`IMAGEH`. |
| `solved.fits.fz` | 342 | 700x700 | The same frame family after plate solving: full WCS, and the `IMAGEW`/`IMAGEH` that astrometry.net adds. |
| `noheader.fits` | 8 | 200x200 | `SIMPLE`, `BITPIX`, `NAXIS*` and nothing else. The degenerate case. |

Two quirks to know before writing a test against them:

- **`camera_id` is the placeholder `XXXXXX`**, not a real six-character camera
  id, so they exercise the path layout without naming a real unit's camera.
- **`IMAGEID` equals `SEQID`**, so image time equals sequence time and a frame
  directory nests the same timestamp twice. Real frames differ; anything
  depending on the two being distinct needs its own fixture.

`tiny.fits` is preferred over POCS's `unsolved.fits`, which carries an
identical header in an 11x larger file.

## `widefield.fits.fz`

Not from POCS. A real PAN025 frame -- `20220227T062239`, the FU Orionis field --
rebinned 4x4 and stored as `uint16` through `fpack`.

It exists because **plate solving has to be tested against a frame at PANOPTES
angular scale.** `plate_solve` hardcodes `--scale-low 10 --scale-high 20
--scale-units degw`, and POCS's `unsolved.fits` is a 700x700 crop spanning about
1.7 degrees, so it fails with our options -- a test built on it would exercise
the plumbing while bypassing the real scale hints.

Rebinning preserves angular size where cropping cannot: 1505x1004 covering the
same 14.9 x 9.9 degrees, 0.92 MB instead of 18 MB, solving in about 2 seconds
with `index-4116`. 8x binning does not work, because `plate_solve` also passes
`--downsample 4` and the two compound to 32x.

To regenerate it, mean-bin a raw frame 4x4, clip to `uint16`, keep only the
keywords the pipeline reads, and `fpack` the result.

It carries no `FILENAME` keyword, which is deliberate: `ImagePathInfo`
reads that first and only falls back to `SEQID`/`IMAGEID` on a `ValueError`,
so a frame without it used to raise `KeyError`. The fixture keeps that path
covered.

**Solving a `.fz` consumes it.** `get_solve_field` unpacks a compressed input
and does not restore it when `replace=False`, so a test must copy this file into
`tmp_path` and never solve it in place.


## `widefield_catalog.parquet`

The Gaia DR3 sources landing on `widefield.fits.fz`, so the end-to-end test has
something to match against. 6,929 rows, 361 kB.

Built by solving the fixture, projecting the full cone through the resulting
WCS, and keeping sources inside the frame plus a 50-pixel margin and brighter
than G=11 -- the margin so the catalog is not cut exactly at the frame edge.

G=11 rather than the pipeline's G=13 default because the fixture is
undersampled: 4x4 binning plus `plate_solve`'s `--downsample 4` leaves a FWHM of
about 1.2 pixels, and it matches 1,350 sources at full depth, of which 1,238 are
brighter than G=11. The remaining 8% would triple the file. Matching 1,238 of
6,929 candidates keeps it a real exercise rather than a lookup that cannot fail.

`picid` is the Gaia DR3 `source_id`, so `scripts/fetch_catalog.py` can rebuild a
superset of this from the field centre with no crossmatch step.
