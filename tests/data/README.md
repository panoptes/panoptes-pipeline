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
