#!/bin/python

import os
import sys
import inspect
import logging

import numpy as np
from sklearn.cross_validation import KFold

from smac.scenario.scenario import Scenario
from smac.configspace import Configuration
from smac.tae.execute_ta_run import StatusType
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, RunHistory2EPM4EIPS, RunHistory2EPM4LogCost
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from smac.epm.rfr_imputator import RFRImputator
import json
import copy
from collections import OrderedDict

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.append(cmd_folder)

__author__ = "Marius Lindauer, Andre Biedenkapp"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "GPLv3"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class EPMImportance(object):
    def __init__(self, scenario_fn, runhistory_fn, traj_fn, cost_='cost'):
        """
        Constructor
        """
        self.complete_source_dict = {}
        self.complete_target_dict = {}
        self.traj_fn = traj_fn
        scen = Scenario(scenario_fn)
        hist = RunHistory()
        hist.load_json(fn=runhistory_fn, cs=scen.cs)

        types = np.zeros(len(scen.cs.get_hyperparameters()),
                         dtype=np.uint)

        for i, param in enumerate(scen.cs.get_hyperparameters()):
            if isinstance(param, CategoricalHyperparameter):
                n_cats = len(param.choices)
                types[i] = n_cats

        if scen.feature_array is not None:
            types = np.hstack(
                (types, np.zeros((scen.feature_array.shape[1]))))

        types = np.array(types, dtype=np.uint)

        self.model = RandomForestWithInstances(types,
                                               scen.feature_array)

        self.num_params = len(scen.cs.get_hyperparameters())
        self._cost = cost_

        RunHistory2EPM = RunHistory2EPM4Cost
        if cost_ == 'log' or scen.run_obj == "runtime":
            RunHistory2EPM = RunHistory2EPM4LogCost
            self._cost = 'log'
        elif cost_ == 'eips':
            RunHistory2EPM = RunHistory2EPM4EIPS

        if scen.run_obj == "runtime":
            if scen.run_obj == "runtime":
                # if we log the performance data,
                # the RFRImputator will already get
                # log transform data from the runhistory
                cutoff = np.log10(scen.cutoff)
                threshold = np.log10(scen.cutoff *
                                     scen.par_factor)
            else:
                cutoff = scen.cutoff
                threshold = scen.cutoff * scen.par_factor

            imputor = RFRImputator(cs=scen.cs,
                                   rs=np.random.RandomState(42),
                                   cutoff=cutoff,
                                   threshold=threshold,
                                   model=self.model,
                                   change_threshold=0.01,
                                   max_iter=10)
            rh2EPM = RunHistory2EPM(scenario=scen,
                                    num_params=self.num_params,
                                    success_states=[StatusType.SUCCESS, ],
                                    impute_censored_data=True,
                                    impute_state=[StatusType.TIMEOUT, ],
                                    imputor=imputor)  # ,
            # log_y=scen.run_obj == "runtime")
        else:
            rh2EPM = RunHistory2EPM(scenario=scen,
                                    num_params=self.num_params,
                                    success_states=None,
                                    impute_censored_data=False,
                                    impute_state=None)  # ,
            # log_y=scen.run_obj == "runtime")

        self.X, self.Y = rh2EPM.transform(hist)

        self.types = types
        self.scen = scen
        self._MAX_P = min(10, self.num_params)

    def run(self, type_):
        """
            main method
        """
        if type_ in ['fs', 'forwardselection']:
            return self.forwardselection()
        else:
            return self.ablation(self.traj_fn)

    @staticmethod
    def read_file(file_):
        """
        Helper method to read the contents of a file, if it exists
        :param file_: name/path to a file
        :return: contents of the file
        """
        file_ = os.path.abspath(file_)
        assert os.path.exists(file_), 'Specified file %s does not exist' % file_

        with open(file_, 'r') as iF:
            data = iF.read()

        return data

    def config_str_to_config(self, cstr):
        """
        Parses a list of strings of the type x='2' into a dictionary param_name -> value
        :param cstr: list of strings
        :return: Configuration
        """
        cdict = {}
        source = {}
        all_params = self.scen.cs.get_hyperparameters()
        for param in all_params:
            source[param.name] = param.default
        self.complete_source_dict = source
        for idx, elem in enumerate(cstr):
            cstr[idx] = elem.split('=')
            cstr[idx][1] = cstr[idx][1].replace("'", '')
        for pname, value in cstr:
            if isinstance(source[pname], int):
                self.complete_target_dict[pname] = int(value)
            elif isinstance(source[pname], float):
                self.complete_target_dict[pname] = float(value)
            else:
                self.complete_target_dict[pname] = value

        for pname, value in cstr:
            conditions = self.scen.cs.get_parent_conditions_of(pname)
            if conditions:
                parents = self.scen.cs.get_parents_of(pname)
                conditions = conditions[0]
                parent = parents[0]
                if self.complete_target_dict[parent.name] in conditions.values:
                    cdict[pname] = self.complete_target_dict[pname]
            else:
                cdict[pname] = self.complete_target_dict[pname]

        return Configuration(self.scen.cs, cdict)

    @staticmethod
    def diff_in_configs(source, target):
        """
        Helper method to determine the differences from the source configuration to the target
        :param source: Configuration
        :param target: Configuration
        :return: list of parameters that differ from source to target / source to target
        """
        differences = list()
        for param in source:
            logging.debug(param + ':')
            if source[param] != target[param]:
                differences.append([param])
                logging.debug('Modified!')
            else:
                logging.debug('Unmodified')

        return differences

    def _pred_on_available_instance_set(self, config):
        """
        Wrapper to get the predictions of the model over the whole instance set
        :param config: dictionary param -> value
        :return: list of [predictions_over_all_test_instances, variances_for_these_predictions]
        """
        mat = []
        round_config = Configuration(self.scen.cs, config).get_array()
        for instance in self.scen.feature_dict:
            mat.append(np.hstack((round_config, self.scen.feature_dict[instance])))

        return self.model.predict(np.array(mat))

    def _get_source_and_target(self, traj_file):
        """
        Method to get the source Configuration and the target. Determines the differences between
        the two Configurations.
        :param traj_file: trajectory_file path/name
        :return: source (Configuration), target (Configuration), differences (list of parameters that are different)
        """
        logging.info('#' * 120)
        source_config_ = self.scen.cs.get_default_configuration()
        logging.info('Source %s' % str(source_config_)[:-1])

        traj_data = self.read_file(traj_file).split('\n')
        if traj_data[-1] == '':
            traj_data = traj_data[:-1]

        target = self.config_str_to_config(json.loads(traj_data[-1])['incumbent'])
        logging.info('Target %s' % str(target)[:-1])
        logging.info('#' * 120)

        differences = self.diff_in_configs(source_config_, target)
        logging.info('Source and target differ in %d parameters' % len(differences))
        return source_config_, target, differences

    def _get_target_improvement(self, source_config_, target):
        """
        Predicts the source and target performances and calculates the difference between the two.
        :param source_config_: Configuration
        :param target: Configuration
        :return: source_performance (float32), target_performance (float32), delta (float32)
        """
        source_performance = self.predict_mean_var_on_available_instance_set(source_config_)[0]
        logging.info('#' * 120)
        logging.info('%10s performance %8s' % ('Source'.rjust(10), ('%02.4f' % source_performance).rjust(8)))
        target_performance = self.predict_mean_var_on_available_instance_set(target)[0]
        logging.info('%10s performance %8s' % ('Target'.rjust(10),
                                               ('%02.4f' % target_performance).rjust(8)))
        delta = source_performance - target_performance
        logging.info('#' * 120)
        return source_performance, target_performance, delta

    def predict_mean_var_on_available_instance_set(self, config):
        pred, var = self._pred_on_available_instance_set(config)
        if self.scen.overall_obj == 'runtime' or self._cost in ['log']:
            pred = np.power(10, pred)
        return np.mean(pred), np.mean(var)

    @staticmethod
    def _log_round_info(candidate, this_rounds_predictions, current_improvement, overall_improvement):
        """
        Helper method to prettily log the information about a current candidate parameter
        :param candidate: Parameter that has been temporarily flipped (String)
        :param this_rounds_predictions: list of all performances for the given rounds flips so far (list of floats)
        :param current_improvement: Contribution to the improvement after flipping candidate (float32)
        :param overall_improvement: Overall Contribution to the improvement after flipping candidate (float32)
        """
        if len(candidate) == 1:
            candidate = candidate[0]
        else:
            candidate = ', '.join(candidate)
        if len(candidate) >= 14:
            candidate = candidate[:5] + '...' + candidate[-5:]
        # TODO instead of hardcoding these string formatting constants, declare them in the "preamble"
        logging.info(('%s predicted performance %8s | Flip improved performance by %9s%% | ' +
                      'Overall improvement %9s%%') % ((candidate + ':').rjust(14),
                                                      ('%02.4f' % this_rounds_predictions[-1]).rjust(8),
                                                      ('%04.4f' % current_improvement).rjust(8),
                                                      ('%02.4f' % (overall_improvement + current_improvement)).rjust(
                                                          8)))

    def determine_parent_child_pairs(self, source_config_, target, differences):
        """
        Method to determine the parent-child relationships (paired flips or not)
        Removes non flippable parameters from differences
        :param source_config_: Configuration
        :param target: Configuration
        :param differences: list of flippable parameters
        :return: dict (parent_name -> child_name), dict (child_name -> parent-name), list (updated differences)
        """
        is_child_of_dict = {}
        is_parent_of_dict = {}
        for param in source_config_:
            parents = self.scen.cs.get_parents_of(param)
            children = self.scen.cs.get_children_of(param)
            logging.info('Param %s has parents %s / children %s' % (param, str(parents), str(children)))
            if parents:
                is_child_of_dict[param] = parents
            if children:
                is_parent_of_dict[param] = children

        for child in is_child_of_dict:
            if [child] in differences:
                for parent, condition in zip(is_child_of_dict[child], self.scen.cs.get_parent_conditions_of(child)):
                    condition = condition.values
                    if child in differences and source_config_[parent.name] not in condition:
                        differences.remove([child])  # TODO fix bug where children are deleted even if it is active in
                                                     # target and source
                    if target[parent.name] in condition and not source_config_[parent.name] in condition:
                        logging.info('%s has to be flipped together with %s' % (child, parent.name))
                        if [parent.name, child] not in differences:
                            differences.append([parent.name, child])
                            differences.remove([parent.name])
                            differences.remove([child])

        return is_parent_of_dict, is_child_of_dict, differences

    def determine_forbidden(self):
        """
        Method to determine forbidden clauses.
        :return: list of lists [[pname, pvalue, pname2, pvalue2]]
        """
        forbidden_clauses = self.scen.cs.forbidden_clauses
        forbidden_descendants = map(lambda x: x.get_descendant_literal_clauses(), forbidden_clauses)
        forbidden_names_value_paris = list()
        for forbidden_literals in forbidden_descendants:
            elem = []
            for literal in forbidden_literals:
                elem.append(literal.hyperparameter.name)
                elem.append(literal.value)
            forbidden_names_value_paris.append(elem)
        return forbidden_names_value_paris

    @staticmethod
    def check_not_forbidden(forbidden_name_value_pairs, modifiable_config):
        """
        Helper method to determine if a current configuration dictionary is forbidden or not
        :param forbidden_name_value_pairs: list of lists of forbidden parameter settings
        :param modifiable_config: dict param_name -> param_value
        :return: boolean. not_forbidden (True)
        """
        not_forbidden = True
        for forbidden_clause in forbidden_name_value_pairs:
            sum_forbidden = 0
            for key in modifiable_config:
                if key in forbidden_clause:
                    at_ = forbidden_clause.index(key) + 1
                    if modifiable_config[key] == forbidden_clause[at_]:
                        sum_forbidden += 1
            not_forbidden = not_forbidden and not (sum_forbidden == len(forbidden_clause))
        return not_forbidden

    @staticmethod
    def _summarize_result(modified_so_far, improved_by, best_runtime):
        """
        Method to log the results as neatly arranged list and return this table/list as ordered dict
        :param modified_so_far: list of parameter names in the order they were flipped
        :param improved_by: list of improvements in percent
        :return: OrderedDict param_name -> improvement
        """
        # This part is just to summarize the result and create a dictionary with param -> improvement
        prev_perf = None
        result = OrderedDict()
        logging.info('Parameters listed by their improvement:')
        at = 0
        for param, percentage in zip(modified_so_far, improved_by):
            param = ', '.join(param)
            if prev_perf is None:
                prev_perf = percentage
            else:
                tmp = percentage
                percentage -= prev_perf
                prev_perf = tmp
            tparam = str(param)
            if len(tparam) >= 14:
                tparam = tparam[:5] + '...' + tparam[-5:]
            logging.info("%s's improvement: %9s%%" % (tparam.rjust(14), ('%02.4f' % percentage).rjust(8)))
            result[param] = (percentage / 100., best_runtime[at])
            at += 1
        logging.info('#' * 120)
        return result

    def ablation(self, traj_file):
        """
        Main Ablation Method
        :param traj_file: Trajectory file path / name
        :return: Ordered dictionary of type parameter_name -> improvement (in percent)
        """
        self.model.train(self.X, self.Y)  # Train the model using the runhistory

        source_config_, target, differences = self._get_source_and_target(traj_file)  # determine source and target

        # Determine Child - Parent relationship
        is_parent_of_dict, is_child_of_dict, differences = self.determine_parent_child_pairs(source_config_, target,
                                                                                             differences)

        modifiable_config = copy.deepcopy(source_config_.get_dictionary())
        modified_so_far = []

        source_performance, target_performance, delta = self._get_target_improvement(source_config_, target)

        prev_perf = source_performance
        overall_improvement = 0
        improved_by = list()
        best_runtime = list()

        forbidden_name_value_pairs = self.determine_forbidden()

        while len(differences) > 0:
            for param_tuple in modified_so_far:  # Set already flipped parameters to the values in the target
                for param in param_tuple:
                    modifiable_config[param] = target[param]

            this_rounds_predictions = []
            for candidate_tuple in differences:  # For each parameter ...

                # TODO there might be a "prettier" or cleaner way of doing this
                popped = list()
                for candidate in candidate_tuple:  # Check if any children are affected by their parents
                    if target[candidate] is None:
                        modifiable_config[candidate] = self.complete_target_dict[candidate]
                    else:
                        modifiable_config[candidate] = target[candidate]
                    if candidate in is_parent_of_dict:
                        children = is_parent_of_dict[candidate]
                        for child in children:
                            conditions = np.array(map(lambda x: self.scen.cs.get_parent_conditions_of(child)[x].values,
                                             range(len(children)))).flatten()
                            if child.name in modifiable_config and modifiable_config[candidate] not in conditions:
                                popped.append(child.name)
                                modifiable_config.pop(child.name)

                not_forbidden = self.check_not_forbidden(forbidden_name_value_pairs, modifiable_config)

                if not not_forbidden:
                    for candidate in candidate_tuple:
                        modifiable_config[candidate] = source_config_[candidate]
                    continue

                pred, var = self.predict_mean_var_on_available_instance_set(modifiable_config)  # predict it's
                                                                                                # performance ...
                for candidate in candidate_tuple:
                    modifiable_config[candidate] = source_config_[candidate]
                for elem in popped:
                    modifiable_config[elem] = source_config_[elem]

                this_rounds_predictions.append(pred)
                current_improvement = (prev_perf - pred) * 100. / delta  # ... and the percentage in improvement
                self._log_round_info(candidate_tuple, this_rounds_predictions, current_improvement, overall_improvement)

            # Determine the best flip
            best = np.argmin(this_rounds_predictions)  # Then determine the best parameter greedily
            overall_improvement += (prev_perf - this_rounds_predictions[best]) * 100. / delta
            prev_perf = this_rounds_predictions[best]
            best_param = differences[best]
            if len(best_param) == 1:  # When fixing the values for the permanently flipped parameter we again have to
                differences.remove(best_param)  # check for any affected children
                if best_param[0] in is_parent_of_dict:
                    children = is_parent_of_dict[best_param[0]]
                    for child in children:
                        conditions = np.array(map(lambda x: self.scen.cs.get_parent_conditions_of(child)[x].values,
                                         range(len(children)))).flatten()
                        if target[best_param[0]] not in conditions:
                            if [child.name] in differences:
                                differences.remove([child.name])
                            if [child.name] in modified_so_far:
                                modified_so_far.remove([child.name])
                            if child.name in modifiable_config:
                                modifiable_config.pop(child.name)
            else:
                for param in best_param:
                    if [param] in differences:
                        differences.remove([param])
                differences.remove(best_param)
            logging.info('#' * 120)
            logging.info('Flipped %s from %s to %s' % (best_param, list(map(lambda x: source_config_[x], best_param)),
                                                       list(map(lambda x: target[x], best_param))))
            logging.info('#' * 120)

            modified_so_far.append(best_param)
            improved_by.append(overall_improvement)
            best_runtime.append(prev_perf)
        improved_by.append(overall_improvement)
        best_runtime.append(prev_perf)
        # Ablation is done here

        return self._summarize_result(modified_so_far, improved_by, best_runtime)

    def forwardselection(self):
        X = self.X
        y = self.Y

        kf = KFold(X.shape[0], n_folds=10)

        param_ids = range(self.num_params)
        used = []
        used.extend(range(self.num_params, len(self.types)))  # always use all features

        result = []

        for _ in range(self._MAX_P):
            scores = []
            for p in param_ids:

                used.append(p)
                X_l = X[:, used]

                model = RandomForestWithInstances(self.types[used],
                                                  self.scen.feature_array)

                rmses = []
                for train, test in kf:
                    X_train = X_l[train]
                    y_train = y[train]
                    X_test = X_l[test]
                    y_test = y[test]

                    model.train(X_train, y_train)
                    y_pred = np.mean(model.predict(X_test)[0])  # TODO AB ask ML if change is correct

                    rmse = np.sqrt(np.mean((y_pred - y_test[:, 0]) ** 2))
                    rmses.append(rmse)
                scores.append(np.mean(rmses))
                used.pop()
            best_indx = np.argmin(scores)
            best_score = scores[best_indx]
            p = param_ids.pop(best_indx)
            used.append(p)

            logging.info("%s : %.4f (RMSE)" % (p, best_score))
            result.append([p, best_score])

        return result