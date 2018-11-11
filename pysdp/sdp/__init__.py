import iris
from tempfile import gettempdir


class Downscaler(object):
    """
    Base class for all Downscaling classes
    """

    def __init__(self, observation, reanalysis, gcm_picontrol, gcm_scenarios,
                 reference_spatial, reference_period, time_unit='month',
                 explained_variance=0.9, work_dir=gettempdir()):
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

        self.obs = observation
        self.rea = reanalysis
        self.picontrol = gcm_picontrol
        self.scenarios = gcm_scenarios
        self.reference_period = reference_period
        self.reference_spatial = reference_spatial
        self.explained_variance = explained_variance
        self.time_unit = time_unit


    def prepare(self, neofs_pred=None, *args, **kwargs):
        """
        prepare data that is given to the model coefficient calculation method

        kwargs are passed to :meth:`call_func`

        Args:

        * unit_list (None, int, iterable):
            depending on self.time_unit this is interpreted as
            all days/months of year (None), single day/month (int) or
            list of days/months (iterable)
        """
        import iris.coord_categorisation
        from .utils import calculate_anomaly_monthlymean, eof_pc_modes
        from scipy.linalg import lstsq
        import numpy as np

        with iris.FUTURE.context(cell_datetime_objects=True):
            rea_ref = self.rea.extract(self.reference_period)
            obs_ref = self.obs.extract(self.reference_period)

        cl_modelcoeff = iris.cube.CubeList()
        cl_obs = iris.cube.CubeList()
        cl_rea = iris.cube.CubeList()

        # Calculate anomalies and mean, detrend, extract time
        for c in obs_ref:
            c = self.area_detrended_anomalies(c)
            # TODO einzelne Monate detrenden
            iris.coord_categorisation.add_month(c, 'time', name='month')

            # Extract by time
            c.cl_time = iris.cube.CubeList()
            for j,j_month in enumerate(set(c.coord('month').points)):
                c.cl_time.append(c.extract(iris.Constraint(month=j_month)))

            cl_obs.append(c)


        # Reduce area, calculate anomalies, detrend, extract time, calculate EOFs
        for c in rea_ref:
            # TODO einzelne Monate detrenden
            c = self.area_detrended_anomalies(c)
            iris.coord_categorisation.add_month(c, 'time', name='month')

            # Extract by time
            c.cl_time = iris.cube.CubeList()
            for j,j_month in enumerate(set(c.coord('month').points)):
                c.cl_time.append(c.extract(iris.Constraint(month=j_month)))

                c.cl_time[j].data = c.cl_time[j].data/np.nanstd(c.cl_time[j].data)
                if neofs_pred is None:
                    c.cl_time[j] = eof_pc_modes(c.cl_time[j], self.explained_variance)
                else:
                    c.cl_time[j] = eof_pc_modes(c.cl_time[j], self.explained_variance, neofs_pred[i])

            cl_rea.append(c)


        # Calculate model Coefficients
        for c_obs in cl_obs:
            for i,cl_time in enumerate(c_obs.cl_time):
                pc_all_rea_fields = np.concatenate(
                    [c_rea.cl_time[i].pcs.data for c_rea in cl_rea],
                    axis=1)

                if isinstance(cl_time.data, np.ma.MaskedArray):
                    mask = ~cl_time.data.mask
                elif isinstance(cl_time.data, np.ndarray):
                    cl_time.data = np.ma.MaskedArray(
                        cl_time.data,
                        ~np.ma.make_mask(cl_time.data))
                    mask = (cl_time.data!=np.nan)
                else:
                    print("Wrong data Object, we need numpy.ndarray or numpy.ma.MaskedArray")
                    raise

                # lstsq for every station in loop due to missing values
                cl_modelcoeff.append(iris.cube.Cube(
                    np.vstack(
                        [ lstsq(pc_all_rea_fields[~cl_time.data[:,j].mask,:],
                                cl_time.data[~cl_time.data[:,j].mask,j].data)[0]
                          for j in range(cl_time.shape[-1]) ]).swapaxes(0,-1),
                    long_name='ESD Modelcoefficients of ' + self.obs[0].name(),
                    var_name='_'.join(['coeff', self.obs[0].var_name])))

                cl_modelcoeff[i].add_dim_coord(iris.coords.DimCoord(
                    range(pc_all_rea_fields.shape[-1]),
                    long_name = 'coeff',
                    var_name = 'coeff'),
                0)

                cl_modelcoeff[i].add_dim_coord(c_obs.coord('station_wmo_id'), 1)
                [cl_modelcoeff[i].add_aux_coord(aux_c,1) for aux_c in self.obs[0].aux_coords]

        [c.remove_coord('month') for c in cl_rea]
        [c.remove_coord('month') for c in cl_obs]

        self.rea = cl_rea
        self.obs = cl_obs
        self.modelcoeff = cl_modelcoeff


    def project(self, picontrol, scenarios, *args, **kwargs):
        """
        project the scenario data

        kwargs are passed to :meth:`call_func`

        Args:

        * unit_list (None, int, iterable):
            depending on self.time_unit this is interpreted as
            all days/months of year (None), single day/month (int) or
            list of days/months (iterable)
        """
        import iris.coord_categorisation
        from scipy.linalg import lstsq
        import numpy as np
        import itertools

        cl_projection = iris.cube.CubeList()
        cl_scenarios = iris.cube.CubeList()
        cl_picontrol = iris.cube.CubeList()

        # import ipdb; ipdb.set_trace()

        for c_pi, c_sce in itertools.izip(picontrol, scenarios):
            c_sce, c_pi.monthly_mean = self.gcm_redarea_anomalies(self.reference_spatial, c_pi, c_sce)
            iris.coord_categorisation.add_month(c_sce, 'time', name='month')

            # Extract by time
            c_sce.cl_time = iris.cube.CubeList()
            for j,j_month in enumerate(set(c_sce.coord('month').points)):
                c_sce.cl_time.append(c_sce.extract(iris.Constraint(month=j_month)))
                c_sce.cl_time[j].data = c_sce.cl_time[j].data/np.nanstd(c_sce.cl_time[j].data)

            cl_scenarios.append(c_sce)
            cl_picontrol.append(c_pi)

        # import ipdb; ipdb.set_trace()

        # Project validation onto reference EOFs and calculate projections
        for c_obs in self.obs:
            for i,cl_time in enumerate(c_obs.cl_time):
                pc_all_rea_fields = np.concatenate(
                    [c_rea.cl_time[i].solver.projectField(
                        c_scenarios.cl_time[i],
                        neofs=c_rea.cl_time[i].neofs).data
                    for c_rea, c_scenarios in zip(self.rea, cl_scenarios)],
                    axis=1)

                cl_projection.append(iris.cube.Cube(
                    np.dot(pc_all_rea_fields,self.modelcoeff[i].data),
                    long_name='ESD Projection of ' + self.obs[0].name(),
                    var_name='_'.join(['esd','proj', self.obs[0].var_name]),
                    units=cl_time.units))

                cl_projection[i].add_dim_coord(cl_scenarios[0].cl_time[i].coord('time'), 0)
                cl_projection[i].add_dim_coord(c_obs.coord('station_wmo_id'), 1)
                [cl_projection[i].add_aux_coord(aux_c,1) for aux_c in self.obs[0].aux_coords]

            cl_projection_final = iris.cube.CubeList()
            for i in range(cl_scenarios[0].shape[0]/len(cl_projection)):
                [cl_projection_final.append(c[i:i+1,...]) for c in cl_projection]

            cl_projection_final = cl_projection_final.concatenate()

        return cl_projection_final



    def validate(self, reference_period, validation_period, neofs_pred=None, *args, **kwargs):
        """
        validate data

        Args:
        * reference_period
        * validation_period
        """
        from .utils import calculate_anomaly_monthlymean, eof_pc_modes
        from scipy.signal import detrend
        from scipy.linalg import lstsq
        import numpy as np
        import iris.coord_categorisation

        with iris.FUTURE.context(cell_datetime_objects=True):
            rea_ref = self.rea.extract(reference_period)
            rea_val = self.rea.extract(validation_period)
            obs_ref = self.obs.extract(reference_period)
            obs_val = self.obs.extract(validation_period)

        # Iterate over list of cube lists
        # Reduce area, detrend, calculate EOFs
        for cube_list in [obs_ref, obs_val]:
            for i,c in enumerate(cube_list):
                c = self.area_detrended_anomalies(c)
                iris.coord_categorisation.add_month(c, 'time', name='month')

                # Extract by time
                c.cl_time = iris.cube.CubeList()
                for j,j_month in enumerate(set(c.coord('month').points)):
                    c.cl_time.append(c.extract(iris.Constraint(month=j_month)))

                cube_list[i] = c

        for cube_list in [rea_ref, rea_val]:
            for i,c in enumerate(cube_list):
                c = self.area_detrended_anomalies(c)
                iris.coord_categorisation.add_month(c, 'time', name='month')

                # Extract by time
                c.cl_time = iris.cube.CubeList()
                for j,j_month in enumerate(set(c.coord('month').points)):
                    c.cl_time.append(c.extract(iris.Constraint(month=j_month)))

                    c.cl_time[j].data = c.cl_time[j].data/np.nanstd(c.cl_time[j].data)
                    if neofs_pred is None:
                        c.cl_time[j] = eof_pc_modes(c.cl_time[j], self.explained_variance)
                    else:
                        c.cl_time[j] = eof_pc_modes(c.cl_time[j], self.explained_variance, neofs_pred[i])

                cube_list[i] = c


        cl_modelcoeff = iris.cube.CubeList()
        cl_projection = iris.cube.CubeList()

        for c_obs_ref,c_obs_val in zip(obs_ref,obs_val):
            # Calculate model Coefficients
            for i,c_obs_cl_time in enumerate(c_obs_ref.cl_time):
                pc_all_rea_fields = np.concatenate(
                    [c_rea.cl_time[i].pcs.data for c_rea in rea_ref],
                    axis=1)

                if isinstance(c_obs_cl_time.data, np.ma.MaskedArray):
                    mask = ~c_obs_cl_time.data.mask
                elif isinstance(c_obs_cl_time.data, np.ndarray):
                    c_obs_cl_time.data = np.ma.MaskedArray(
                        c_obs_cl_time.data,
                        ~np.ma.make_mask(c_obs_cl_time.data))
                    mask = (c_obs_cl_time.data!=np.nan)
                else:
                    print("Wrong data Object, we need numpy.ndarray or numpy.ma.MaskedArray")
                    raise

                # lstsq for every station in loop due to missing values
                cl_modelcoeff.append(iris.cube.Cube(
                    np.vstack(
                        [ lstsq(pc_all_rea_fields[~c_obs_cl_time.data[:,j].mask,:],
                               c_obs_cl_time.data[~c_obs_cl_time.data[:,j].mask,j].data)[0]
                         for j in range(c_obs_cl_time.shape[-1]) ]).swapaxes(0,-1),
                        # [lstsq(pc_all_rea_fields[mask[:,j],:],
                        #        c_obs_cl_time.data[mask[:,j],j].data)[0]
                        #  for j in range(c_obs_cl_time.shape[-1]) ]).swapaxes(0,-1),
                    long_name='ESD Modelcoefficients of ' + self.obs[0].name(),
                    var_name='_'.join(['coeff', self.obs[0].var_name])))

                cl_modelcoeff[i].add_dim_coord(iris.coords.DimCoord(
                    range(pc_all_rea_fields.shape[-1]),
                    long_name = 'coeff',
                    var_name = 'coeff'),
                0)

                cl_modelcoeff[i].add_dim_coord(c_obs_val.coord('station_wmo_id'), 1)
                [cl_modelcoeff[i].add_aux_coord(aux_c,1) for aux_c in self.obs[0].aux_coords]

            # Project validation onto reference EOFs and calculate projections
            for i,c_obs_cl_time in enumerate(c_obs_val.cl_time):
                pc_all_rea_fields = np.concatenate(
                    [c_rea_ref.cl_time[i].solver.projectField(
                        c_rea_val.cl_time[i],
                        neofs=c_rea_ref.cl_time[i].neofs).data
                     for c_rea_ref, c_rea_val in zip(rea_ref, rea_val)],
                    axis=1)

                cl_projection.append(iris.cube.Cube(
                    np.dot(pc_all_rea_fields,cl_modelcoeff[i].data),
                    long_name='ESD Projection of ' + self.obs[0].name(),
                    var_name='_'.join(['esd','proj', self.obs[0].var_name]),
                    units=c_obs_cl_time.units))

                # cl_projection[i].add_dim_coord(rea_val[0].cl_time[i].coord('time'), 0)
                cl_projection[i].add_dim_coord(c_obs_cl_time.coord('time'), 0)
                cl_projection[i].add_dim_coord(c_obs_val.coord('station_wmo_id'), 1)
                [cl_projection[i].add_aux_coord(aux_c,1) for aux_c in self.obs[0].aux_coords]

            cl_projection_final = iris.cube.CubeList()
            for i in range(c_obs_val.shape[0]/len(cl_projection)):
                [cl_projection_final.append(c[i:i+1,...]) for c in cl_projection]

            cl_projection_final = cl_projection_final.concatenate()

        [c_obs_val.remove_coord('month') for c_obs_val in obs_val]
        return cl_modelcoeff, cl_projection_final, obs_val



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
        c.monthMean = c_monthMean

        return c

    def gcm_redarea_anomalies(self, reference_spatial, c_pi, c_sce):
        from .utils import calculate_anomaly_monthlymean, eof_pc_modes
        from scipy.signal import detrend
        import numpy as np

        # import ipdb; ipdb.set_trace()
        # Extract the reduced area if cube is not a timeseries
        # for c_pi, c_sce in zip(self.picontrol, self.scenarios):
        c_pi = c_pi.intersection(latitude=reference_spatial[0:2],
                                longitude=reference_spatial[2:])
        c_sce = c_sce.intersection(latitude=reference_spatial[0:2],
                                longitude=reference_spatial[2:])
        # Extract data
        c_pi_data = c_pi.data
        c_sce_data = c_sce.data
        # Calculate anomaly and means
        _, c_pi_monthMean = calculate_anomaly_monthlymean(c_pi_data)
        # Calculate monthly Anomalies
        c_sce.data, c_sce_monthMean = calculate_anomaly_monthlymean(c_sce_data, c_pi_monthMean)
        #c_sce.data, c_sce_monthMean = calculate_anomaly_monthlymean(c_sce_data, None)

        return c_sce, c_pi_monthMean



class ValidateDownscaler(Downscaler):
    """
    Convenience class for Greater Alpine Reagion downscaler
    """

    def __init__(self, observation, reanalysis, reference_spatial,
                 reference_period, *args, **kwargs):
        super(ValidateDownscaler, self).__init__(
            observation, reanalysis, None, None, reference_spatial,
            reference_period, time_unit='month', *args, **kwargs)


class ProjectDownscaler(Downscaler):
    """
    Convenience class for Greater Alpine Reagion downscaler
    """

    def __init__(self, observation, reanalysis, gcm_picontrol, gcm_scenarios,
                 reference_spatial, reference_period, *args, **kwargs):
        super(ProjectDownscaler, self).__init__(
            observation, reanalysis, gcm_picontrol, gcm_scenarios,
            reference_spatial, reference_period, time_unit='month', *args, **kwargs)
