import logging
import numpy as np


def _calculate_coefficients(obs_cube, mod_cube, *args, **kwargs):
    """
    Calculate Cofficients

    Args:

    * obs_cube (:class:`iris.cube.Cube`):
        the observational data

    * mod_cube (:class:`iris.cube.Cube`):
        the model data at the reference period
    """

red_area = iris.Constraint(
    latitude=lambda cell: 35 <= cell <= 65,
)

# Extract the reduced area
obs_cube = obs_cube.extract(red_area)
obs_cube = obs_cube.intersection(longitude=(-50, 30))


obs_data = obs_cube.data

# Calculate Anomalies and monthly Means
obs_anom, obs_monthMean = calc_anom(obs_data)

# Detrend the Anomalies
obs_detrended = detrend(obs_anom, type='constant', axis=0)

obs_cube.data = obs_detrended
#obs_cube.data = obs_anom

# Seasonal Mean
iris.coord_categorisation.add_season(obs_cube, 'time', name='clim_season')
iris.coord_categorisation.add_season_year(obs_cube, 'time', name='season_year')

obs_cube_seas = obs_cube.aggregated_by(
    ['clim_season', 'season_year'],
    iris.analysis.MEAN)

# Filter anomalies with less than 3 Months
spans_three_months = lambda t: (t.bound[1] - t.bound[0]) > 3*28*24.0
three_months_bound = iris.Constraint(time=spans_three_months)
obs_cube_seas = obs_cube_seas.extract(three_months_bound)

# Extract one Season
obs_cube_seas_ext = obs_cube_seas.extract(iris.Constraint(clim_season='djf'))

# Calculate seasonal EOFs
eof_solver = Eof(obs_cube_seas_ext, weights='coslat')
obs_cube_seas_ext_eof = eof_solver.eofs(neofs=5)
