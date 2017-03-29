# (C) Wegener Center for Climate and Global Change, University of Graz, 2015
#
# This file is part of pySDP.
#
# pySDP is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 3 as published by the
# Free Software Foundation.
#
# pySDP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pySDP. If not, see <http://www.gnu.org/licenses/>.
"""
pySDP
=====

A package for analyzing climate data
"""

import logging

# pySDP version
__version__ = '0.1'

# restrict imports when using "from pysdp import *"
__all__ = ['config', 'tmp_path', 'data_path']

# configuration
import os
_writeable_dir = os.path.join(os.path.expanduser('~'), '.local', 'share')
_data_dir = os.path.join(os.environ.get("XDG_DATA_HOME", _writeable_dir),
                         'pySDP')
_tmp_dir = os.path.join(os.environ.get("TMPDIR", '/tmp'), 'pySDP')
config = {
    'data_dir': _data_dir,
    'tmp_dir': _tmp_dir,
}
"""
The config dictionary stores global configuration values for pysdp.

In the first instance, the config is defined in ``pysdp/__init__.py``. It
is possible to provide site wide customisations by including a
``siteconfig.py`` file along with the pySDP source code. ``siteconfig.py``
should contain a function called ``update_config`` which takes the config
dictionary instance as its first and only argument (from where it is
possible to update the dictionary howsoever desired).

Keys in the config dictionary:

 * ``data_dir`` - the absolute path to a directory where data will be found

 * ``tmp_dir`` - the absolute path to a directory where temporary data
                 will be stored
"""
del _data_dir
del _writeable_dir
del _tmp_dir

# try to import the siteconfig file
try:
    from pysdp.siteconfig import update_config as _update_config
    _update_config(config)
except ImportError:
    pass


def tmp_path(*path_to_join):
    return os.path.join(config['tmp_dir'], *path_to_join)


def data_path(*path_to_join):
    return os.path.join(config['data_dir'], *path_to_join)
