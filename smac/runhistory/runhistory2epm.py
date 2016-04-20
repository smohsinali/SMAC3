import abc
import copy
from collections import OrderedDict
import logging

import numpy as np

from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.configspace import impute_inactive_values
import smac.epm.base_imputor

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "AGPLv3"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RunType(object):

    '''
       class to define numbers for status types.
       Makes life easier in select_runs
    '''
    SUCCESS = 1
    TIMEOUT = 2
    CENSORED = 3


class AbstractRunHistory2EPM(object):
    __metaclass__ = abc.ABCMeta

    '''
        takes a runhistory object and preprocess data in order to train EPM
    '''

    def __init__(self, scenario, num_params,
                 success_states=None,
                 impute_censored_data=False,
                 impute_state=None,
                 imputor=None,
                 rs=None):
        '''
        Constructor
        Parameters
        ----------
        scenario: Scenario Object
            Algorithm Configuration Scenario
        num_params : int
            number of parameters in config space
        instances: list
            list of instance names
        success_states: list, optional
            list of states considered as successful (such as StatusType.SUCCESS)
        impute_censored_data: bool, optional
            should we impute data?
        imputor: epm.base_imputor Instance
            Object to impute censored data
        impute_state: list, optional
            list of states that mark censored data (such as StatusType.TIMEOUT)
            in combination with runtime < cutoff_time
        rs : numpy.random.RandomState
            only used for reshuffling data after imputation
        '''
        self.logger = logging.getLogger("runhistory2epm")

        # General arguments
        self.scenario = scenario
        self.rs = rs
        self.num_params = num_params
        self.instances = scenario.train_insts

        # Configuration
        self.success_states = success_states
        self.impute_censored_data = impute_censored_data
        self.impute_state = impute_state
        self.cutoff_time = self.scenario.cutoff
        self.imputor = imputor

        # Fill with some default values
        if rs is None:
            self.rs = np.random.RandomState()

        if self.impute_state is None:
            self.impute_state = [StatusType.TIMEOUT, ]

        if self.success_states is None:
            self.success_states = [StatusType.SUCCESS, ]

        self.config = OrderedDict({
            'success_states': success_states,
            'impute_censored_data': impute_censored_data,
            'cutoff_time': scenario.cutoff,
            'impute_state': impute_state,
        })

        self.logger = logging.getLogger("runhistory2epm")
        self.num_params = num_params

        # Sanity checks
        # TODO: Decide whether we need this
        if impute_censored_data and scenario.run_obj != "runtime":
            # So far we don't know how to handle censored quality data
            self.logger.critical("Cannot impute censored data when not "
                                 "optimizing runtime")
            raise NotImplementedError("Cannot impute censored data when not "
                                      "optimizing runtime")

        # Check imputor stuff
        if impute_censored_data and self.imputor is None:
            self.logger.critical("You want me to impute cencored data, but "
                                 "I don't know how. Imputor is None")
            raise ValueError("impute_censored data, but no imputor given")
        elif impute_censored_data and not \
                isinstance(self.imputor, smac.epm.base_imputor.BaseImputor):
            raise ValueError("Given imputor is not an instance of "
                             "smac.epm.base_imputor.BaseImputor, but %s" %
                             type(self.imputor))

    @abc.abstractmethod
    def _build_matrix(self, run_list, runhistory, instances=None):
        # TODO: why is "instances" an argument? it is never used in the methods
        # down below
        raise NotImplementedError()

    def transform(self, runhistory):
        '''
        returns vector representation of runhistory

        Parameters
        ----------
        runhistory : list of dicts
                parameter configurations

        Returns
        -------
        configs: np.ndarray of shape = [n_configs, n_params]
        f_map: np.ndarray of shape = [n_samples, 2]
        y: np.ndarray of shape = [n_samples,]
        '''
        assert isinstance(runhistory, RunHistory)

        # consider only successfully finished runs
        s_run_list = self.__select_runs(rh_data=copy.deepcopy(runhistory.data),
                                        select=RunType.SUCCESS)
        # Store a list of instance IDs
        s_instance_id_list = [k.instance_id for k in s_run_list.keys()]
        configs, f_map, Y = self._build_matrix(run_list=s_run_list, runhistory=runhistory,
                                               instances=s_instance_id_list)

        # Also get TIMEOUT runs
        t_run_list = self.__select_runs(rh_data=copy.deepcopy(runhistory.data),
                                        select=RunType.TIMEOUT)
        t_instance_id_list = [k.instance_id for k in s_run_list.keys()]

        _, tf_map, tY = self._build_matrix(run_list=t_run_list, runhistory=runhistory,
                                           instances=t_instance_id_list)

        # if we don't have successful runs,
        # we have to return all timeout runs
        if not s_run_list:
            return configs, tf_map, tY

        if self.impute_censored_data:
            # Get all censored runs
            c_run_list = self.__select_runs(rh_data=copy.deepcopy(runhistory.data),
                                            select=RunType.CENSORED)
            if len(c_run_list) == 0:
                self.logger.debug("No censored data found, skip imputation")
            else:
                # Store a list of instance IDs
                c_instance_id_list = [k.instance_id for k in c_run_list.keys()]

                _, cen_f_map, cen_Y = self._build_matrix(run_list=c_run_list,
                                                         runhistory=runhistory,
                                                         instances=c_instance_id_list)

                # Also impute TIMEOUTS
                cen_f_map = np.vstack((cen_f_map, tf_map))
                cen_Y = np.concatenate((cen_Y, tY))
                self.logger.debug("%d TIMOUTS, %d censored, %d regular" %
                                  (tf_map.shape[0], cen_f_map.shape[0], f_map.shape[0]))

                imp_Y = self.imputor.impute(configs=configs,
                                            censored_f_map=cen_f_map,
                                            censored_y=cen_Y,
                                            uncensored_f_map=f_map,
                                            uncensored_y=Y)

                # Shuffle data to mix censored and imputed data
                f_map = np.vstack((f_map, cen_f_map))
                Y = np.concatenate((Y, imp_Y))
        else:
            # If we do not impute,we also return TIMEOUT data
            f_map = np.vstack((f_map, tf_map))
            Y = np.concatenate((Y, tY))

        return configs, f_map, Y

    def __select_runs(self, rh_data, select):
        '''
        select runs of a runhistory

        Parameters
        ----------
        rh_data : runhistory
            dict of ConfigSpace.config

        select : RunType.SUCCESS
            one of "success", "timeout", "censored"
            return only runs for this type
        Returns
        -------
        list of ConfigSpace.config
        '''
        new_dict = dict()

        if select == RunType.SUCCESS:
            for run in rh_data.keys():
                if rh_data[run].status in self.success_states:
                    new_dict[run] = rh_data[run]
        elif select == RunType.TIMEOUT:
            for run in rh_data.keys():
                if (rh_data[run].status == StatusType.TIMEOUT and
                        rh_data[run].time >= self.cutoff_time):
                    new_dict[run] = rh_data[run]
        elif select == RunType.CENSORED:
            for run in rh_data.keys():
                if rh_data[run].status in self.impute_state \
                        and rh_data[run].time < self.cutoff_time:
                    new_dict[run] = rh_data[run]
        else:
            err_msg = "select must be in (%s), but is %s" % \
                      (",".join(["%d" % t for t in
                                 [RunType.SUCCESS, RunType.TIMEOUT,
                                  RunType.CENSORED]]), select)
            self.logger.critical(err_msg)
            raise ValueError(err_msg)

        return new_dict


class RunHistory2EPM4LogCost(AbstractRunHistory2EPM):

    def _build_matrix(self, run_list, runhistory, instances=None):
        '''
            builds a matrices for X, y (and f_map) given a list of runs 
            and the runhistory

            Arguments
            ---------
            run_list: list
                list of (key,run)
            runhistory: Runhistory()
                runhistory from which the matrices are build
            instances: ??
                ???

            Returns
            -------
            configs: np.ndarray of shape = [n_configs, n_params]
            f_map: np.ndarray of shape = [n_samples, 2]
            y: np.ndarray of shape = [n_samples,]

        '''
        n_rows = len(run_list)
        configs = np.array([impute_inactive_values(runhistory.ids_config[id_+1]).get_array()
                            for id_ in range(runhistory._n_id)])
        y = np.ones([n_rows, 1])
        f_map = np.ones([n_rows, 2], dtype=np.uint)

        # Then populate matrix
        for row, (key, run) in enumerate(run_list.items()):
            # Scaling is automatically done in configSpace
            f_map[row] = [
                key.config_id-1, self.instances.index(key.instance_id)]
            # run_array[row, -1] = instances[row]
            y[row, 0] = run.cost

        y = np.log10(y)

        return configs, f_map, y


class RunHistory2EPM4Cost(AbstractRunHistory2EPM):

    def _build_matrix(self, run_list, runhistory, instances=None):
        '''
            builds a matrices for X, y (and f_map) given a list of runs 
            and the runhistory

            Arguments
            ---------
            run_list: list
                list of (key,run)
            runhistory: Runhistory()
                runhistory from which the matrices are build
            instances: ??
                ???

            Returns
            -------
            configs: np.ndarray of shape = [n_configs, n_params]
            f_map: np.ndarray of shape = [n_samples, 2]
            y: np.ndarray of shape = [n_samples,]

        '''
        n_rows = len(run_list)
        configs = np.array([impute_inactive_values(runhistory.ids_config[id_+1]).get_array()
                            for id_ in range(runhistory._n_id)])
        y = np.ones([n_rows, 1])
        f_map = np.ones([n_rows, 2], dtype=np.uint)

        # Then populate matrix
        for row, (key, run) in enumerate(run_list.items()):
            # Scaling is automatically done in configSpace
            f_map[row] = [
                key.config_id-1, self.instances.index(key.instance_id)]
            # run_array[row, -1] = instances[row]
            y[row, 0] = run.cost

        return configs, f_map, y


class RunHistory2EPM4EIPS(AbstractRunHistory2EPM):

    def _build_matrix(self, run_list, runhistory, instances=None):
        '''
            builds a matrices for X, y (and f_map) given a list of runs 
            and the runhistory

            Arguments
            ---------
            run_list: list
                list of (key,run)
            runhistory: Runhistory()
                runhistory from which the matrices are build
            instances: ??
                ???

            Returns
            -------
            X: np.ndarray of shape = [n_samples, n_params]
            y: np.ndarray of shape = [n_samples,]
            f_map: np.ndarray of shape = [n_samples, 2]

        '''
        # First build nan-matrix of size #configs x #params+1
        n_rows = len(run_list)
        configs = np.array([impute_inactive_values(runhistory.ids_config[id_+1]).get_array()
                            for id_ in range(runhistory._n_id)])
        Y = np.ones([n_rows, 1])
        f_map = np.ones([n_rows, 2], dtype=np.uint)

        # Then populate matrix
        for row, (key, run) in enumerate(run_list.items()):
            # Scaling is automatically done in configSpace
            f_map[row] = [
                key.config_id-1, self.instances.index(key.instance_id)]
            # run_array[row, -1] = instances[row]
            Y[row, 0] = run.cost
            Y[row, 1] = np.log(1 + run.time)

        return configs, f_map, Y
