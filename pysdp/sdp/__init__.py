import numpy as np
import iris
import iris.coord_categorisation
from scipy.signal import detrend

iris.FUTURE.netcdf_promote = True


class Downscaler(object):
    """
    Base class for all Downscaling classes
    """

    # def __init__(self, call_func, observation, model, reference_spatial,
    #              reference_period, time_unit='month', correction_period=None):

    def __init__(self, observation, reanalysis, reference_spatial,
                 reference_period, validation_period=None, time_unit='month'):
        """
        Args:

        * call_func (callable):
            | *call signature*: (obs_cube, ref_cube, sce_cubes,
                                 \*args, \**kwargs)

        * observation (:class:`pycat.io.Dataset`):
            the observation dataset

        * model (:class:`pycat.io.dataset.Dataset`):
            the model dataset

        * reference_period (tuple of :class:`datetime.datetime`):
            the reference period that observations and model share

        * reference_spatial (tuple of 4 floats (north, east, south, west)): 
          geographical coordinates that describe the bounding box

        Kwargs:

        * time_unit (str):
            correction will be performed on daily (day) or
            monthly (month) basis

        * correction_period (tuple of :class:`datetime.datetime`):
            the period for which the correction shall be done
        """

        # self.call_func = call_func
        self.obs = observation
        self.rea = reanalysis
        #self.mod = model
        self.reference_period = reference_period
        self.reference_spatial = reference_spatial
        self.time_unit = time_unit


    def prepare(self, *args, **kwargs):
        """
        prepare data that is given to the model coefficient calculation method

        kwargs are passed to :meth:`call_func`

        Args:

        * unit_list (None, int, iterable):
            depending on self.time_unit this is interpreted as
            all days/months of year (None), single day/month (int) or
            list of days/months (iterable)
        """
        from .utils import generate_year_constraint_with_window

        # Reanalysis cube list
        for rea_cube in self.rea:
            # Extract the reduced area
            rea_cube = rea_cube.intersection(latitude=self.reference_spatial[0:2],
                                             longitude=self.reference_spatial[2:])

            rea_data = rea_cube.data
            # Calculate Anomalies and monthly Means
            rea_anom, rea_monthMean = self._calc_anom(rea_data)

            # Detrend the Anomalies
            rea_detrended = detrend(rea_anom, type='constant', axis=0)

            rea_cube.data = rea_detrended

        # Observation cube list
        for obs_cube in self.obs:
            obs_data = obs_cube.data
            # Calculate Anomalies and monthly Means
            obs_anom, obs_monthMean = self._calc_anom(obs_data)

            # Detrend the Anomalies
            obs_detrended = detrend(obs_anom, type='constant', axis=0)

            obs_cube.data = obs_detrended


        # call the correction function
        # self.call_func(obs_cube, *args, **kwargs)


        # # Extract one Season
        # obs_seas_ext = obs_seas.extract(iris.Constraint(clim_season='djf'))

        # # Calculate seasonal EOFs
        # eof_solver = Eof(obs_seas_ext, weights='coslat')
        # obs_seas_ext_eof = eof_solver.eofs(neofs=5)


    def seasonal_mean(self, *args, **kwargs):
        self.rea = self._calc_seasonal_mean(self.rea)
        self.obs = self._calc_seasonal_mean(self.obs)


    def calculate_eof(self):
        return True
        # # Calculate seasonal EOFs
        # eof_solver = Eof(obs_seas_ext, weights='coslat')
        # obs_seas_ext_eof = eof_solver.eofs(neofs=5)


    def _calc_seasonal_mean(self, cube_list, *args, **kwargs):
        # Calculate the seasonal mean for all cubes in the cube list
        for c in cube_list:
            iris.coord_categorisation.add_season(c, 'time', name='clim_season')
            iris.coord_categorisation.add_season_year(c, 'time', name='season_year')
            c = c.aggregated_by(
                ['clim_season', 'season_year'],
                iris.analysis.MEAN)
            print(c)

            # Filter anomalies with less than 3 Months
            spans_three_months = lambda t: (t.bound[1] - t.bound[0]) > 3*28*24.0
            three_months_bound = iris.Constraint(time=spans_three_months)
            c = c.extract(three_months_bound)

        return cube_list


    def _calc_anom(self,var_inp):
        """ Calculates the anomalies and monthly mean for a numpy array"""
        orig_dim = var_inp.shape
        reshape_dim = [-1,12]
        reshape_dim.extend(var_inp.shape[1:])

        var_reshape = np.reshape(var_inp, reshape_dim, order='C')

        var_monthMean = np.mean(var_reshape, axis=0)

        var_anom = var_reshape - var_monthMean
        var_anom = np.reshape(var_anom, orig_dim, order='C')

        return var_anom, var_monthMean


class GARDownscaler(Downscaler):
    """
    convenience class for Matulla Downscaler
    """

    def __init__(self, observation, reanalysis, reference_spatial,
                 reference_period, validation_period, *args, **kwargs):
        super(GARDownscaler, self).__init__(
            observation, reanalysis, reference_spatial,
            reference_period, validation_period, time_unit='month', *args, **kwargs)
        self.validation_period = validation_period


