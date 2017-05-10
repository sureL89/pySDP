# -*- coding: utf-8 -*-

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
Utility functions needed for generating temporal constraints on
:class:`iris.cube.Cubes <iris.cube.Cube>`
"""

from iris.time import PartialDateTime
from iris import Constraint

def generate_year_constraint_with_window(year, window):
    """
    generate a :class:`iris.Constraint` on the time axis for specified year ± window

    Args:

    * year (int):
        centered year for the constraint

    * window (int):
        number of years around the given year

    Returns:

        an :class:`iris.Constraint` on the time-axis
    """
    first_year = PartialDateTime(year=year-window)
    last_year = PartialDateTime(year=year+window)
    return Constraint(time=lambda cell: first_year <= cell.point <= last_year)
