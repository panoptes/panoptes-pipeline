"""Test-suite support package.

Makes ``tests`` importable as a package so ``from tests.synthetic import ...``
resolves under plain ``pytest`` (which, unlike ``python -m pytest``, does not put
the working directory on ``sys.path``).
"""
