#!/usr/bin/env python

from distutils.core import setup
from pysdp import __version__ as version
from setuptools import setup

setup(name='pySDP',
      version=version,
      description='Statistically downscaled projections',
      license='GPLv3',
      author='Georg Seyerl',
      author_email='georg.seyerl@zamg.ac.at',
      packages=['pysdp', 'pysdp.io', 'pysdp.sdp'],
      classifiers=[
          'Development Status :: 1 - Planning',
          'Topic :: Scientific/Engineering :: Physics',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
      ],
      keywords = ['climate', ],

#      tests_require = ['nose'],
     )
