#!/usr/bin/env python
# Licensed under an MIT style license - see LICENSE.txt

from setuptools import setup, find_namespace_packages

import itertools

from configparser import ConfigParser
from distutils.command.build_py import build_py

import builtins
builtins._PANOPTES_SETUP_ = True

from panoptes.piaa.version import __version__

# Get some values from the setup.cfg
conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))

AUTHOR = metadata.get('author', '')
AUTHOR_EMAIL = metadata.get('author_email', '')
DESCRIPTION = metadata.get('description', '')
KEYWORDS = metadata.get('keywords', 'Project PANOPTES')
LICENSE = metadata.get('license', 'unknown')
LONG_DESCRIPTION = metadata.get('long_description', '')
NAME = metadata.get('name', 'panoptes-piaa')
PACKAGENAME = metadata.get('package_name', 'packagename')
URL = metadata.get('url', 'https://projectpanoptes.org')

requirements = list()
requirements_fn = 'requirements.txt'
with open(requirements_fn) as f:
    requirements = f.read().splitlines()

modules = {
    'required': requirements,
    'testing': [
        'codecov',
        'coverage',
        'coveralls',
        'mocket',
        'pycodestyle==2.3.1',
        'pytest>=3.6',
        'pytest-cov',
        'pytest-remotedata>=0.3.1'
    ],
}


setup(name=NAME,
      version=__version__,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      keywords=KEYWORDS,
      python_requires='>=3.6',
      setup_requires=['pytest-runner'],
      tests_require=modules['testing'],
      # List additional groups of dependencies here (e.g. development
      # dependencies). You can install these using the following syntax,
      # for example:
      # $ pip install -e .[dev,test]
      install_requires=modules['required'],
      extras_require={
          'testing': modules['testing'],
          'all': list(set(itertools.chain.from_iterable(modules.values())))
      },
      packages=find_namespace_packages(exclude=['tests', 'test_*']),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: POSIX',
          'Programming Language :: C',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
      ],
      cmdclass={'build_py': build_py}
      )
