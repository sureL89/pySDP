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
from eofs.iris import Eof
import numpy as np


def eof_pc_modes(cube, fraction_explained, neofs_pred=None, show_info=False):
    n_eofs = 0
    n_fraction = 0
    solver = Eof(cube, weights='coslat')
    if neofs_pred is None:
        # Number of EOFs needed to explain the fraction (fraction_explained) of the total variance
        while n_fraction < fraction_explained:
            n_eofs = n_eofs+1
            cube.eof_var = solver.varianceFraction(neigs=n_eofs)
            n_fraction = np.sum(cube.eof_var.data)
        cube.eof = solver.eofs(neofs=n_eofs)
        cube.pcs = solver.pcs(npcs=n_eofs)
        cube.solver = solver
        cube.neofs = n_eofs
    else:
        cube.eof = solver.eofs(neofs=neofs_pred)
        cube.pcs = solver.pcs(npcs=neofs_pred)
        cube.solver = solver
        cube.neofs = neofs_pred

    # Function return
    if show_info:
        for i in range(0,n_eofs):
            print('EOF '+str(i+1)+' fraction: '+str("%.2f" % (cube.eof_var.data[i]*100))+'%')
        print(str("%.2f" % (n_fraction*100))+'% of the total variance explained by '+str(n_eofs)+' EOF modes.')
        return cube
    elif show_info == False:
        return cube
    else:
        print 'Missing show_info=True or show_info=False'

def calculate_anomaly_monthlymean(var_inp, var_monthMean=None):
    """ Calculates the anomalies and monthly mean for a numpy array

    Args:

    * var_inp (numpy.array):
        Input Array. First axis must be time and divisible by 12
    """
    orig_dim = var_inp.shape
    reshape_dim = [-1,12]
    reshape_dim.extend(var_inp.shape[1:])

    var_reshape = np.reshape(var_inp, reshape_dim, order='C')

    if var_monthMean is None:
        var_monthMean = np.mean(var_reshape, axis=0)

    var_anom = var_reshape - var_monthMean
    var_anom = np.reshape(var_anom, orig_dim, order='C')

    return var_anom, var_monthMean

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

def seasonal_mean(c, *args, **kwargs):
    # Calculate the seasonal mean for all cubes in the cube list
    import iris.coord_categorisation

    iris.coord_categorisation.add_season(c, 'time', name='clim_season')
    iris.coord_categorisation.add_season_year(c, 'time', name='season_year')
    c = c.aggregated_by(
        ['clim_season', 'season_year'],
        iris.analysis.MEAN)

    return c

def seasonal_sum(c, *args, **kwargs):
    # Calculate the seasonal mean for all cubes in the cube list
    import iris.coord_categorisation

    iris.coord_categorisation.add_season(c, 'time', name='clim_season')
    iris.coord_categorisation.add_season_year(c, 'time', name='season_year')
    c = c.aggregated_by(
        ['clim_season', 'season_year'],
        iris.analysis.SUM)

    return c
