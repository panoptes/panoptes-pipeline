"""Support code the pipeline uses but that is not the algorithm.

Calibration, source detection, plate solving and catalog matching
(`images`), catalog lookups against a local parquet (`sources`), sequence
helpers (`observations`) and plotting (`plot`).

This package had no `__init__.py` and worked anyway, as an implicit namespace
package nested inside a regular one. Nothing in Python objects to that, but a
static reader -- `griffe`, which builds the API reference from these modules --
cannot follow it, so the documentation quietly lost a whole subpackage.
"""
