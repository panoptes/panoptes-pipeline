import pytest


def pytest_addoption(parser):
    parser.addoption("--network-tests", action="store_true", default=False, help="Perform network tests")
