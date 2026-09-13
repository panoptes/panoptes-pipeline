# Data contract

What the pipeline reads, what it produces, where those live, and how four
repositories agree on the vocabulary. Cite as "data contract 3.2".

This document exists because the pipeline stopped producing. The papermill
path, `image.py`, `observation.py` and the Firestore writer were deleted in
the cloud removal, and `extract_metadata` survives with nothing calling it.
Everything downstream -- the Firestore documents, the `observations.csv`
summary, and the `panoptes-data` query client -- reads products that are no
longer being made. Restoring that circuit without Google Cloud is what this
plan specifies.

This document is the *reasoning*. What is planned, in progress and done is in
the [issue tracker](https://github.com/panoptes/panoptes-pipeline/issues), and
anything needing a human decision is an issue with the `decision` label.

---

## 1. The shape of the problem

### 1.1 Which direction the data flows

The pipeline is the **producer**, and this is the fact the rest of the
document follows from.

[POCS](https://github.com/panoptes/POCS) controls the units and writes the FITS
headers. The pipeline reads those headers, processes the frames, and emits
metadata. That metadata populated Firestore, which generated the
`observations.csv` summary, which
[`panoptes-data`](https://github.com/panoptes/panoptes-data) queries.

So `panoptes-data` is **downstream** of the pipeline for everything except raw
frame discovery, and the metadata schema is not an external substrate to query
around -- it is a pipeline output we own and may redefine. An earlier reading of
improvement plan 4.6 had this backwards and treated the client's query surface
as the binding constraint. It is not; it follows.

The schema is already specified in code. `extract_metadata` returns
`dict(unit=..., sequence=..., image=...)`, and the deleted orchestrator read
that from `assets/metadata.json` and wrote each key to its own Firestore
document. **JSON was always the wire format and Firestore was a sink.**
Removing Google Cloud means keeping the file and dropping the forward.

### 1.2 Four repositories, one vocabulary, nothing enforcing it

| Repository | Role |
|---|---|
| `POCS` | Writes the FITS headers. Upstream of everything. |
| `panoptes-utils` | Shared base library. Deliberately generic, not survey-specific. |
| `panoptes-pipeline` | Reads headers, processes frames, produces the documents and products. Owns this contract. |
| `panoptes-data` | Discovery and fetch, then queries what the pipeline produced. |

The vocabulary spans all four and nothing checks it, which is how every break
recorded in this document happened. Two examples, one at each end:

`ObservationInfo` fails on every sequence from 2025 onward, because the record
carries `fits_public_url` where it used to carry `public_url` and the reader
was never told. Probing one sequence per year dates the break cleanly to the
start of 2025: 2016--2024 have `public_url`, 2025--2026 do not. The CLI wraps
the call in a bare `except Exception` and prints in red, which is why nobody
noticed. **Dataset B (1.3 in the improvement plan) is entirely 2024+, so the
long-sequence benchmark is unreachable through the client.**

At the other end, `extract_metadata` reads four header keywords that current
POCS no longer writes (2.2). It degrades silently: `header.get("WHTLVLN")`
returns `None`, `MEASRGGB` falls back to `"0 0 0 0"`. Same class of defect,
one repository upstream.

Both are the same failure: a vocabulary shared across repository boundaries
with no declared contract and no loud failure when it drifts.

## 2. The header vocabulary

The input half of the contract. The pipeline does not get to choose these --
POCS writes them -- but it does get to declare which it requires and fail
loudly when one is missing.

### 2.1 What a raw frame carries

From a 2025-11 frame written by `POCSv0.8.0.dev220`, 44 header cards total:

**Present:** `CAMSN`, `INTSN`, `WHTLVLN`, `WHTLVLS`, `ISO`, `EXPTIME`,
`AIRMASS`, `MOONFRAC`, `MOONSEP`, `CAMTEMP`, `MEASRGGB`, `REDBAL`, `BLUEBAL`,
`COLORTMP`, `CIRCCONF`, `RA-MNT`, `DEC-MNT`, `HA-MNT`, `FIELD`, `CREATOR`,
`INSTRUME`, `IMAGEID`, `SEQID`, `LAT-OBS`, `LONG-OBS`, `ELEV-OBS`, `OBSERVER`.

**Absent:** `GAIN` / `EGAIN`, `RDNOISE`, any black-level keyword, `IMAGEW` /
`IMAGEH`, `BAYERPAT`, and any WCS.

Three consequences:

- **Saturation is free on historical data.** `WHTLVLN`/`WHTLVLS` give the
  white level per frame, offline, with no registry and no prior processing.
  Measured values are 11435/11892 on this PAN007 camera and 11765/12277 on the
  paper's PAN012 camera, against `saturation: float = 15872.0` hardcoded for
  the whole fleet. Both cameras sit roughly 25% below the threshold, so the
  fleet-wide default currently fails to mask saturated pixels on both. That is
  the silent-corruption mode improvement plan 1.4 names, confirmed on real
  data.
- **Gain, read noise and black level are not in the header at all.** They are
  the entire load-bearing content of a camera registry (6).
- **`IMAGEW`/`IMAGEH` do not exist**, so `int(header.get("IMAGEW", 0))` yields
  a silent zero. That is why 16% of the observation index has null dimensions,
  and why `image_width: int = 6000` exists to paper over it. `NAXIS1`/`NAXIS2`
  are the real source.

`BAYERPAT` being absent is why `masks.infer_pattern` exists, and the absence of
WCS is expected: plate solving is the pipeline's job (5).

### 2.2 What POCS has already dropped

Checked against POCS `main` at `7370af45`:

| Keyword | Pipeline reads it | Current POCS |
|---|---|---|
| `INSTRUME` / `CAM-ID` | via `path_info.camera_id` | written |
| `MOONFRAC`, `MOONSEP`, `AIRMASS` | yes | written |
| `EGAIN`, `BITDEPTH` | **no** | written |
| `CAMSN` -- body serial | yes | **absent repo-wide** |
| `WHTLVLN` / `WHTLVLS` -- white level | yes | **absent repo-wide** |
| `INTSN` -- lens serial | yes | **absent repo-wide** |
| `MEASRGGB` | yes | **absent repo-wide** |

The archive is unaffected -- frames already written still carry these -- but the
break is forward-looking and it undoes two things this plan would otherwise
rely on. Saturation stops being free on new frames (2.1), and `CAMSN` stops
being a viable registry key (2.3).

POCS keeps its header vocabulary in a single authoritative `fields` map in
`camera.py`. That is the thing to pin against: the pipeline declares the
keywords it requires, and a keyword it requires that POCS no longer writes is
a loud failure in both repositories rather than a `None` that propagates.

### 2.3 Identifiers: three of them, only one is a key

| Identifier | Example | Source | Use as a hardware key |
|---|---|---|---|
| camera uid | `f6eb3d` | first six characters of the camera's gphoto `serialnumber`, written as `INSTRUME` and `CAM-ID` | **yes** |
| body serial | `042070008650` | `CAMSN`, the number printed on the body | cross-reference only |
| slot name | `Cam00`, `CAM1` | assigned per unit, independent of hardware | **never** |

The uid is the middle component of every `sequence_id` and `image_id`
(`PAN007_f6eb3d_20251127T044424`), so it is present in every path, every
filename and every record, and it is derived from hardware. It is not derived
from `CAMSN`; the two are different serial numbers for the same body, which is
why they do not convert into one another.

**The uid is the registry key**, for three reasons: it is hardware-derived, it
is never missing, and current POCS still writes it. `CAMSN` fails the last two
-- it is absent from 13--19% of recent index rows and has been removed from POCS
entirely. Where `CAMSN` is present it is a useful human-readable
cross-reference, and it must be read as a string: it carries leading zeros, and
a float coercion turns `032071000633` into `3.207100e+10`. This is the `picid`
hazard in a different column.

The slot name is a label. It is assigned per unit regardless of which body is
installed, so a profile keyed on it would silently apply one camera's
calibration to another.

**Two apparent counterexamples in the index are data defects, not instability**,
and the shape says so. `camera_id` `14d3bd` on PAN001 appears against two
serials: 2,332 sequences with `012070048413` spanning 2016--2020, against four
sequences with `022071246706` confined to a 22-minute window on 2018-09-27.
The same pattern appears as `358d0f` (175 rows) versus `358df0` (1 row), a
transposition. Index rebuilds should flag a uid whose serial is inconsistent
rather than trusting the minority reading.

### 2.4 Required, optional, and failing loudly

`extract_metadata` currently mixes three behaviors with no principle:
`float(header.get("ELEV-OBS"))` raises `TypeError` if absent,
`int(header.get("IMAGEW", 0))` silently yields zero, and
`header.get("WHTLVLN")` silently yields `None`.

Each keyword the pipeline reads is declared **required** or **optional**. A
missing required keyword raises, naming the keyword and the file. An optional
one is recorded as absent, never as a default that cannot be distinguished
from a measurement. This is improvement plan 1.4's "fail loudly rather than
defaulting", applied at the point where values enter.

## 3. The document

### 3.1 Shape

`extract_metadata` already returns the right thing: a nested dictionary with
three top-level keys, `unit`, `sequence` and `image`. Those correspond to the
three Firestore document levels, and the path hierarchy is the key:

```
units/{unit_id}
units/{unit_id}/observations/{sequence_id}
units/{unit_id}/observations/{sequence_id}/images/{image_id}
```

Keep the shape. One addition is required:

**The camera uid and, where present, the body serial belong in the `image`
document.** Today the camera block sits only in `sequence_info`, which is why
joining a per-frame white level to a per-camera profile is awkward -- the
measurement is per frame and the identifier is per sequence. Both are in every
header, so this costs nothing.

### 3.2 Firestore compatibility

The document must stay loadable into a document store without transformation,
so that re-attaching one later is an upload rather than a migration:

- Times as ISO 8601 strings, parsed on ingest.
- Nested maps, no arrays of arrays.
- Field names free of `.`. The dotted `camera.serial_number` form in
  `observations.csv` is a *view* over the nested map, not storage.
- Bulk tables stay out of the document. `sources.parquet` sits beside it, which
  is what keeps the document far below the 1 MB Firestore limit.

### 3.3 Layout on disk

The directory tree mirrors the document path, which already mirrors the raw
bucket layout that `ImagePathInfo` parses:

```
<root>/PAN007/f6eb3d/20251127T044424/
  observation.json          # the `sequence` document
  20251127T044520/
    metadata.json           # {unit, sequence, image}
    image.fits              # reduced
    extras.fits             # named ImageHDUs
    sources.parquet
  20251127T044551/
    ...
```

`FileSettings` already names those four per-image artifacts. Uploading to a
document store is then a walk, and `rsync` to the processing server works with
no special handling.

### 3.4 Provenance, and the fingerprint that makes it useful

Every `metadata.json` records the parameters it was produced with. That block
exists today, and it is the reason `params_camera_saturation=15872` appears in
archive records -- the pipeline faithfully recording its own settings. Two
changes make it load-bearing rather than decorative:

**Record resolved values with their provenance tier, not a dump of global
settings.** "Saturation was 11435, read from `WHTLVLN`" and "saturation was
15872, because nobody knew" currently serialize identically. Each calibration
value carries where it came from: header, measured from this observation,
registry, or model default.

**Compare the fingerprint, do not merely store it.** The old flow stored
`params` and never checked them, so a settings change left stale products with
no signal. The fingerprint is the cache key (5.2).

Because the pipeline now records provenance, one rule follows for anything
reading the archive: **`params_camera_*` in existing records is never a source
for a camera profile.** Those fields are the fleet-wide defaults improvement
plan 1.4 exists to remove. Reading them back would close a loop in which the
pipeline rediscovers its own wrong constants.

### 3.5 Products

`image.fits`, `extras.fits` and `sources.parquet` per frame, as `FileSettings`
already specifies. At the observation level the stamp collection is **parquet,
not HDF5**: `make_stamps` already builds a DataFrame indexed on `picid` with
`pixel_000`...`pixel_NNN` columns, which is a parquet-shaped object that has
been living in an HDF5 file. This also removes `h5py` and `tables` from the
dependency set.

## 4. Storage

### 4.1 Firestore was solving distribution, not scale

The whole metadata archive, across ten years and the entire fleet:

| | |
|---|---|
| Sequences | 12,439 |
| Frames | 563,566 |
| Per-frame record | ~3.4 kB of JSON, 59 populated fields |
| As JSON files | ~1.9 GB |
| As one parquet file | **~76 MB** |
| In memory as a DataFrame | ~0.27 GB |

**The entire archive fits in memory.** Firestore was arbitrating concurrent
writes from cloud functions firing on arbitrary bucket events -- a distribution
problem. Remove the distribution and the problem it was hired for does not
exist.

This rules out the two obvious replacements:

**MongoDB** is the right shape and the wrong operational cost. It is a server:
a daemon, a port, credentials, backups, a version to pin, something to start
before the pipeline runs. That re-creates the coupling the cloud removal
deleted and breaks improvement plan 4.3's requirement that running the
algorithm needs no credentials. Not worth it for 76 MB.

**HDF5** is not a document store and not NoSQL; it is a container for arrays.
Single-writer, no concurrency story, corrupts on an interrupted write, and
locking is a recurring source of pain. Holding 60-field nested documents means
either a rigid compound dtype -- which fights exactly the schema evolution this
document is about -- or one group per frame, which is a filesystem with worse
tooling and no `grep`. HDF5's role here was the array products, and 3.5 moves
those to parquet.

**SQLite** is the one worth keeping in reserve: a file rather than a server,
in the standard library, ACID, WAL for concurrent readers, and `json_extract`
lets a nested document be stored verbatim and still queried into. That is
Firestore's semantics at no operational cost, if 5.2 ever needs it.

### 4.2 Files are the truth; the index is derived

JSON documents on disk are the source of truth: document-store-compatible by
construction, provenance beside the products it describes, one corrupt file is
not a corrupt archive, and ordinary file tools work.

The query surface is a parquet index built by walking the tree. That is what
replaces `observations.csv`, and it is what `panoptes-data` reads.

**The rule that keeps this honest: the index must be deletable.** If it cannot
be removed, rebuilt from the tree, and give the same answers, something has
entered the index that belongs in the documents. That test is what prevents a
derived index quietly becoming a second source of truth -- which is exactly how
the current `observations.csv` ended up as the only holder of numbers nothing
else records.

### 4.3 When this changes

If a shared multi-user archive returns, or writes arrive from machines we do
not control, a server is needed again and MongoDB is then the right analogue.
The layout here preserves that option for free. Deferring is cheap; adopting
now is not.

## 5. Processing

The cloud flow had two abilities worth keeping exactly: process a single FITS
as it arrives, or process everything in a batch. Both survive, with bucket
events replaced by directory enumeration.

### 5.1 Three walks, and only one is expensive

Distinguishing these matters, because two of the three are seconds and the
third is hours:

**Index walk.** Reads the `metadata.json` files that already exist, writes
parquet. Read-only, no FITS opened. This is the selection surface (4.2).

**Work-list walk.** Enumerates raw FITS, compares each against its output
directory and params fingerprint, and emits the work list with a reason per
frame: missing, params changed, prior error, or forced. Also cheap, also opens
no pixels. This is an inspectable artifact rather than a loop buried inside
the batch command -- over 563,566 frames the scope and the reasons are worth
seeing before committing, and a diffable work list is how a parameter change
proves it invalidated what was expected and nothing more.

**Processing.** The expensive per-FITS work, consuming the work list. Bias,
background, **plate solve**, source detection, catalog match, then the four
artifacts. Plate solving belongs here, at the single-frame stage.

The observation-level command runs the work-list walk, processes what needs
processing, then aggregates and writes `observation.json`. That is the deleted
`observation.py` structure with `list_blobs` replaced by a glob; its
`fits_matcher` regex survives unchanged.

### 5.2 Idempotency

The old status machine lived in Firestore: read `status`, skip if at or past
`PROCESSING` unless forced. Locally:

1. No `metadata.json` -- process.
2. Present, status at or past `PROCESSING`, fingerprint matches -- skip.
3. Present, fingerprint differs, or status is `ERROR` -- reprocess.
4. Forced -- always reprocess.

This needs `ImageStatus` and `ObservationStatus`, which currently exist only in
`panoptes-data` -- enums describing the *pipeline's* stages (`CALIBRATING`,
`SOLVING`, `MATCHING`, `EXTRACTING`), owned by the reader and absent from the
writer. They are versioned with this contract, so they belong here.

Concurrency does not force a database. The per-frame unit of work is
embarrassingly parallel with **no shared mutable state**, because each worker
writes only into its own `unit/camera/sequence/frame/` directory -- the path
hierarchy partitions the write set. The only contention is two workers on the
same sequence document, which is a lock file or one SQLite row, not a reason
for a server.

### 5.3 Output roots

The processed tree is an **independent root, named per run**, not a sibling
nested inside the raw tree. The raw tree may be read-only or mirrored; outputs
should be regenerable without touching it; and -- the deciding reason -- this
project's whole activity is comparing algorithm variants. `processed/baseline/`
beside `processed/candidate/` is only possible if outputs are not nested in the
inputs. The params fingerprint in each document says which root is which.

## 6. Camera calibration

improvement plan 1.4 requires per-camera values with no fleet-wide defaults.
This is where they come from.

### 6.1 Resolution order

Keyed on the camera uid (2.3), for each of saturation, gain, read noise, black
level and sensor dimensions:

1. **Header**, where POCS writes it.
2. **Measured** from this observation.
3. **Registry**, the stored profile for that uid.
4. **Fail loudly.** No fleet-wide default at any level.

The tier reached is recorded with the value (3.4).

### 6.2 The registry is small

Given 2.1, the registry only ever has to hold what no header carries: gain,
read noise and black level. Three numbers per camera uid, across 37 cameras.

It lives in the same tree as everything else, at
`units/{unit_id}/cameras/{uid}.json` -- the same shape a document store would
use, readable offline, no new mechanism. It is not a schema addition to
`panoptes-data`; putting the pipeline's calibration constants in the read
client is the same inversion as the status enums in 5.2.

### 6.3 Gain is a POCS problem

`effective_gain: float = 1.5` in `settings.py` is not this repository's
invention. POCS's Canon camera class returns a hardcoded
`1.5 * (u.electron / u.adu)` as a class property, with `bit_depth` likewise
fixed at 12 bits, and stamps that into `EGAIN`. The pipeline copied a POCS
constant.

So the correct fix is upstream: **make `egain` per-body**, measured by photon
transfer, and POCS writes the true value into every header from then on. The
pipeline then reads `EGAIN` as it reads any other keyword, and the registry
only has to cover frames written before that lands. Black level and read noise
follow the same argument -- the unit knows its own camera, and a value the unit
records is available to every consumer forever, while a value only this
pipeline measures is available to one.

This is the clearest case for POCS being in scope rather than treated as
fixed.

## 7. Old data and new data

Ten years of archive exist and must stay usable; the survey should also get
better going forward. Those pull in different directions exactly once -- on the
header vocabulary -- and the resolution is to read the old and fix the new.

**Reading old data.** The pipeline supports keywords POCS has retired
(`CAMSN`, `WHTLVLN`, `WHTLVLS`, `INTSN`, `MEASRGGB`). They are marked optional,
present-where-present, and never required. A frame from 2018 and a frame from
2026 both produce a valid document; they differ in which provenance tiers the
values came from, and the document says so.

**Improving new data.** The keywords worth having back in POCS are the ones
that make a measurement available to every consumer instead of just this
pipeline: white level, and per-body `EGAIN`. A value in the header is a value
no registry has to carry and no reprocessing has to recover.

**The registry bridges the gap**, and its size is the measure of how well this
is going: large for old cameras whose frames carry little, and ideally empty
for frames written after the POCS work lands.

## 8. What each repository does

**`POCS`** -- declare the header vocabulary as a contract rather than an
implementation detail; make `egain` per-body; restore white level; keep the
uid as the hardware identifier.

**`panoptes-utils`** -- unchanged. Nothing from this contract moves there:
`extract_metadata` reads POCS-specific keywords, which is survey convention,
not generic astronomy. `ImagePathInfo` already lives there and stays, though
it parses a PANOPTES-specific path convention and sits on the wrong side of
that line.

**`panoptes-pipeline`** -- owns this contract: the header vocabulary it
requires, the document, the products, the three walks, the registry, and the
index builder. The status enums return here.

**`panoptes-data`** -- discovery and fetch of raw frames, then a query client
over the index. Its `public_url` breakage dissolves rather than needing a
patch: the field is named by the contract instead of guessed at. Its query
surface gains what benchmark selection needs (9).

## 9. What this unblocks in benchmark selection

improvement plan 1.3 assumes benchmark sequences are selectable through the
existing client. They are not, and the specifics say what the query surface
has to become.

**Duration is not expressible.** Of 12,439 sequences, 82 have 300 or more
frames -- all of them 2024 or later, and `total_exptime` is null for **every
one**. Dataset B is defined by "300+ frames over 3+ hours" and the second half
cannot be evaluated at the index level at all.

**Frame count means something different from what it appears to.**
`num_images` counts frames *uploaded*, not frames usable. One 372-frame
sequence has metadata for 310; the remaining 62 are errors. The 300-frame
criterion needs the usable count, which only per-frame records carry.

**Selection on pipeline-derived quantities is circular.** FWHM, source counts
and plate-solved positions in the archive are outputs of the implementation
being replaced. Ranking benchmark candidates by them, in order to build the
substrate for judging new code, is a trap that would be hard to detect
afterwards. Selection therefore runs in two passes: **cut on header facts**
(frame count, duration, unit, camera uid, ISO, exposure, moon, airmass,
field), download, then **measure drift and seeing locally** and cut again.

That second pass is not a workaround. improvement plan 3.2 makes drift the
strongest predictor of achievable precision, so measuring it with current code
on raw frames is what selecting on it honestly requires.

---

## Status of the claims in this document

Measured against the live archive and both repositories, and reproducible:

- Archive shape and sizes (4.1), the 82-sequence dataset B pool and the null
  `total_exptime` (9), and the 372-versus-310 frame count -- from
  `observations.csv` as of 2026-09-12 and the per-observation metadata
  endpoint.
- Header contents (2.1) -- one frame,
  `PAN007/f6eb3d/20251127T044424/20251127T044520`, written by
  `POCSv0.8.0.dev220`. **One frame from one camera**: the keyword list should
  be confirmed across POCS versions before the required/optional split (2.4)
  is fixed.
- White levels (2.1) -- two cameras, 2018 and 2025. Enough to show the
  fleet-wide threshold is wrong; not a fleet survey.
- POCS keyword absences and the hardcoded `egain` (2.2, 6.3) -- POCS `main` at
  `7370af45`.
- The uid derivation and the two index defects (2.3) -- POCS source and the
  full index.
- The 2025 `ObservationInfo` failure (1.2) -- reproduced directly, and dated
  by probing one sequence per year.
