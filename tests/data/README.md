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
