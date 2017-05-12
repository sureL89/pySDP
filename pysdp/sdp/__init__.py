import iris

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
        from .utils import calculate_anomaly_monthlymean
        from scipy.signal import detrend

        # Iterate over list of cube lists
        for cube_list in [self.rea, self.obs]:
            for i,c in enumerate(cube_list):
                # Extract the reduced area if cube is not a timeseries
                if not c.attributes.get('featureType', '') == 'timeSeries':
                    c = c.intersection(latitude=self.reference_spatial[0:2],
                                       longitude=self.reference_spatial[2:])
                # Extract data
                c_data = c.data
                # Calculate Anomalies and monthly Means
                c_anom, c_monthMean = calculate_anomaly_monthlymean(c_data)

                # Detrend the Anomalies
                c_detrended = detrend(c_anom, type='constant', axis=0)

                c.data = c_detrended
                cube_list[i] = c


    def seasonal_mean(self, *args, **kwargs):
        # Calculate the seasonal mean for all cubes in the cube list
        import iris.coord_categorisation

        for cube_list in [self.rea, self.obs]:
            for i,c in enumerate(cube_list):
                iris.coord_categorisation.add_season(c, 'time', name='clim_season')
                iris.coord_categorisation.add_season_year(c, 'time', name='season_year')
                cube_list[i] = c.aggregated_by(
                    ['clim_season', 'season_year'],
                    iris.analysis.MEAN)

            # Filter anomalies with less than 3 Months
            # spans_three_months = lambda t: (t.bound[1] - t.bound[0]) > 3*28*24.0
            # three_months_bound = iris.Constraint(time=spans_three_months)
            # self.rea[i] = c.extract(three_months_bound)

        self.time_unit = "seasonal"


    def eof_analysis(self):
        # Calculate EOFs
        if self.time_unit == "seasonal":
            pass

        # eof_solver = Eof(obs_seas_ext, weights='coslat')
        # obs_seas_ext_eof = eof_solver.eofs(neofs=5)



class GARDownscaler(Downscaler):
    """
    Convenience class for Greater Alpine Reagion downscaler
    """

    def __init__(self, observation, reanalysis, reference_spatial,
                 reference_period, validation_period, *args, **kwargs):
        super(GARDownscaler, self).__init__(
            observation, reanalysis, reference_spatial,
            reference_period, validation_period, time_unit='month', *args, **kwargs)
        self.validation_period = validation_period


