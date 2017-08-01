import iris

class Downscaler(object):
    """
    Base class for all Downscaling classes
    """

    # def __init__(self, call_func, observation, model, reference_spatial,
    #              reference_period, time_unit='month', correction_period=None):

    def __init__(self, observation, reanalysis, reference_spatial,
                 reference_period, time_unit='month', explained_variance=0.9):
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
        self.explained_variance = explained_variance


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
        import numpy

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
                if type(c_anom) != numpy.ndarray:
                    c_detrended = c_anom
                    c_detrended[~c_anom.mask] = detrend(c_anom[~c_anom.mask], type='constant', axis=0)
                else:
                    c_detrended = detrend(c_anom, type='constant', axis=0)

                c.data = c_detrended
                cube_list[i] = c

    def validate(self, reference_period, validation_period, *args, **kwargs):
        """
        validate data

        kwargs are passed to :meth:`call_func`

        Args:

        * reference_period
        * validation_period
        """
        from .utils import calculate_anomaly_monthlymean, eof_pc_modes, seasonal_mean
        from scipy.signal import detrend
        import numpy as np
        import iris.iterate

        with iris.FUTURE.context(cell_datetime_objects=True):
            rea_ref = self.rea.extract(reference_period)
            rea_val = self.rea.extract(validation_period)
            obs_ref = self.obs.extract(reference_period)
            obs_val = self.obs.extract(validation_period)

        # Iterate over list of cube lists
        print("---------------------------------------")
        print("Reduce area, detrend, seasonal mean, calculate seasonal EOFs")
        for cube_list in [obs_ref, obs_val, rea_ref, rea_val]:
            print("\tReduce area, detrend, seasonal mean")
            for i,c in enumerate(cube_list):
                print("\t"+c.name())
                c = self.area_detrended_anomalies(c)
                c = seasonal_mean(c)
                self.time_unit = "seasonal"

                c.seas = iris.cube.CubeList()
                for j,seas in enumerate(set(c.coord('clim_season').points)):
                    print("\t\tEOF calucation for cube " + c.name() + " season " + seas)
                    c.seas.append(c.extract(iris.Constraint(clim_season=seas)))
                    c.seas[j] = eof_pc_modes(c.seas[j], self.explained_variance)

                cube_list[i] = c

        # Calculate model Coefficients
        print("---------------------------------------")
        print("Calculate model coefficients")

        #c_modelcoeff = iris.cube.CubeList()
        c_modelcoeff = iris.cube.CubeList()
        c_projection = iris.cube.CubeList()

        for c_obs_ref,c_obs_val in zip(obs_ref,obs_val):
            print("\tObservation: "+ c_obs_ref.name())
            for i,c_obs_seas in enumerate(c_obs_ref.seas):
                print("\t\tSeason: "+ str(i))
                pc_all_rea_fields = np.concatenate([c_rea.seas[i].pcs.data for c_rea in rea_ref],axis=1)
                # modelcoeff.append(np.linalg.lstsq(pc_all_rea_fields, c_obs_seas.data))
                c_modelcoeff.append(iris.cube.Cube(np.linalg.lstsq(pc_all_rea_fields, c_obs_seas.data)[0], long_name='model_coefficients', var_name='coeff'))
                c_modelcoeff[i].add_dim_coord(
                    iris.coords.DimCoord(
                        range(pc_all_rea_fields.shape[-1]),
                        long_name='pc_number',
                        var_name='pc'),
                    0)
                c_modelcoeff[i].add_dim_coord(c_obs_ref.coord('station_wmo_id'), 1)

            for i,c_obs_seas in enumerate(c_obs_val.seas):
                pc_all_rea_fields = np.concatenate([c_rea_ref.seas[i].solver.projectField(c_rea_val.seas[i], neofs=c_rea_ref.seas[i].neofs).data for c_rea_ref, c_rea_val in zip(rea_ref, rea_val)],axis=1)
                c_projection.append(iris.cube.Cube(np.dot(pc_all_rea_fields,c_modelcoeff[i].data), long_name='projection', var_name='proj'))
                c_projection[i].add_dim_coord(
                    iris.coords.DimCoord(
                        range(pc_all_rea_fields.shape[0]),
                        long_name='time',
                        var_name='time'),
                    0)
                c_projection[i].add_dim_coord(c_obs_val.coord('station_wmo_id'), 1)



        self.modelcoeff = c_modelcoeff
        self.projection = c_projection
        self.obs_ref = obs_ref
        self.rea_ref = rea_ref
        self.obs_val = obs_val
        self.rea_val = rea_val


        # for c in rea_ref:
        #     c.rea_ref_seas = {}
        #     for seas in set(c.coord('clim_season').points):
        #         print("---------------------------------------")
        #         print("EOF calucation for cube " + c.name() + " season " + seas)
        #         c.rea_ref_seas[seas]  = c.extract(iris.Constraint(clim_season=seas))
        #         c.rea_ref_seas[seas].eof, c.rea_ref_seas[seas].pcs, c.rea_ref_seas[seas].solver  = eof_pc_modes(c.rea_ref_seas[seas], self.explained_variance, show_info=True)


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

        self.time_unit = "seasonal"


    def eof_analysis(self):
        from .utils import eof_pc_modes

        # Calculate EOFs
        for cube_list in [self.rea, self.obs]:
            for i,c in enumerate(cube_list):
                c.eof, c.pcs = eof_pc_modes(c, self.explained_variance, show_info=True)

    def area_detrended_anomalies(self, c):
        from .utils import calculate_anomaly_monthlymean, eof_pc_modes
        from scipy.signal import detrend
        import numpy as np

        # Extract the reduced area if cube is not a timeseries
        if not c.attributes.get('featureType', '') == 'timeSeries':
            c = c.intersection(latitude=self.reference_spatial[0:2],
                                longitude=self.reference_spatial[2:])
        # Extract data
        c_data = c.data
        # Calculate Anomalies and monthly Means
        c_anom, c_monthMean = calculate_anomaly_monthlymean(c_data)

        # Detrend the Anomalies
        if type(c_anom) != np.ndarray:
            c_detrended = c_anom
            c_detrended[~c_anom.mask] = detrend(c_anom[~c_anom.mask], type='constant', axis=0)
        else:
            c_detrended = detrend(c_anom, type='constant', axis=0)

        c.data = c_detrended

        return c



class GARDownscaler(Downscaler):
    """
    Convenience class for Greater Alpine Reagion downscaler
    """

    def __init__(self, observation, reanalysis, reference_spatial,
                 reference_period, *args, **kwargs):
        super(GARDownscaler, self).__init__(
            observation, reanalysis, reference_spatial,
            reference_period, time_unit='month', *args, **kwargs)

