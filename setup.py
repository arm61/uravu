#! /usr/bin/env python
"""
setup.py for uravu

@author: Andrew R. McCluskey (andrew.mccluskey@diamond.ac.uk)
"""

# System imports
import io
from os import path
from setuptools import setup, find_packages

PACKAGES = find_packages()

# versioning
MAJOR = 1
MINOR = 2
MICRO = 1
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


THIS_DIRECTORY = path.abspath(path.dirname(__file__))
with io.open(path.join(THIS_DIRECTORY, 'README.md')) as f:
    LONG_DESCRIPTION = f.read()

INFO = {
        'name': 'uravu',
        'description': 'Bayesian methods for analytical relationships',
        'author': 'Andrew R. McCluskey',
        'author_email': 'andrew.mccluskey@diamond.ac.uk',
        'packages': PACKAGES,
        'include_package_data': True,
        'setup_requires': ['numpy', 'scipy>=1.5.4', 'emcee', 'tqdm',
                           'uncertainties', 'dynesty>=1.0.1'],
        'install_requires': ['numpy', 'scipy>=1.5.4', 'emcee', 'tqdm',
                             'uncertainties', 'dynesty>=1.0.1'],
        'version': VERSION,
        'license': 'MIT',
        'long_description': LONG_DESCRIPTION,
        'long_description_content_type': 'text/markdown',
        'classifiers': ['Development Status :: 4 - Beta',
                        'Intended Audience :: Science/Research',
                        'License :: OSI Approved :: MIT License',
                        'Natural Language :: English',
                        'Operating System :: OS Independent',
                        'Programming Language :: Python :: 3.6',
                        'Programming Language :: Python :: 3.7',
                        'Programming Language :: Python :: 3.8',
                        'Topic :: Scientific/Engineering',
                        'Topic :: Scientific/Engineering :: Chemistry',
                        'Topic :: Scientific/Engineering :: Physics']
        }

####################################################################
# this is where setup starts
####################################################################


def setup_package():
    """
    Runs package setup
    """
    setup(**INFO)


if __name__ == '__main__':
    setup_package()
