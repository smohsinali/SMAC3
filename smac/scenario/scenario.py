from collections import defaultdict
import os
import sys
import logging
import numpy
import shlex
import time
import datetime
import copy

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from smac.utils.io.input_reader import InputReader
from smac.utils.scenario_options import scenario_options
from smac.configspace import pcs

__author__ = "Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.2"


def _is_truthy(arg):
    return arg in ["1", "true", True]


class Scenario(object):

    '''
    main class of SMAC
    '''

    def __init__(self, scenario, cmd_args=None):
        """Construct scenario object from file or dictionary.

        Parameters
        ----------
        scenario : str or dict
            if str, it will be interpreted as to a path a scenario file
            if dict, it will be directly to get all scenario related information
        cmd_args : dict
            command line arguments that were not processed by argparse

        """
        self.logger = logging.getLogger("scenario")
        self.PCA_DIM = 7

        self.in_reader = InputReader()

        if type(scenario) is str:
            scenario_fn = scenario
            self.logger.info("Reading scenario file: %s" % (scenario_fn))
            scenario = self.in_reader.read_scenario_file(scenario_fn)
        elif type(scenario) is dict:
            scenario = copy.copy(scenario)
        else:
            raise TypeError(
                "Wrong type of scenario (str or dict are supported)")

        if cmd_args:
            scenario.update(cmd_args)

        self._arguments = {}
        self._groups = defaultdict(set)
        self._add_arguments()

        # Mapping external to internal aliases
        self.options_ext2int = {}
        for o in scenario_options:
            dest = self._arguments[o]['dest']
            if dest: self.options_ext2int[o] = dest
            else:    self.options_ext2int[o] = o

        # Parse arguments
        parsed_arguments = {}
        for key, value in self._arguments.items():
            arg_name, arg_value = self._parse_argument(key, scenario, **value)
            parsed_arguments[arg_name] = arg_value

 
        if len(scenario) != 0:
            raise ValueError('Could not parse the following arguments: %s' %
                             str(list(scenario.keys())))

        for group, potential_members in self._groups.items():
            n_members_in_scenario = 0
            for pm in potential_members:
                if pm in parsed_arguments:
                    n_members_in_scenario += 1

            if n_members_in_scenario != 1:
                raise ValueError('Exactly one of the following arguments must '
                                 'be specified in the scenario file: %s' %
                                 str(potential_members))

        for arg_name, arg_value in parsed_arguments.items():
            setattr(self, arg_name, arg_value)

        self._transform_arguments()

        if self.output_dir:
            self._write()

    def add_argument(self, name, help, callback=None, default=None,
                     dest=None, required=False, mutually_exclusive_group=None,
                     choice=None):
        """Add argument to the scenario object.

        Parameters
        ----------
        name : str
            Argument name
        help : str
            Help text which can be displayed in the documentation.
        callback : callable, optional
            If given, the callback will be called when the argument is
            parsed. Useful for custom casting/typechecking.
        default : object, optional
            Default value if the argument is not given. Default to ``None``.
        dest : str
            Assign the argument to scenario object by this name.
        required : bool
            If True, the scenario will raise an error if the argument is not
            given.
        mutually_exclusive_group : str
            Group arguments with similar behaviour by assigning the same string
            value. The scenario will ensure that exactly one of the arguments is
            given. Is used for example to ensure that either a configuration
            space object or a parameter file is passed to the scenario. Can not
            be used together with ``required``.
        """
        if not isinstance(required, bool):
            raise TypeError("Argument 'required' must be of type 'bool'.")
        if required is not False and mutually_exclusive_group is not None:
            raise ValueError("Cannot make argument '%s' required and add it to"
                             " a group of mutually exclusive arguments." % name)
        if choice is not None and not isinstance(choice, (list, set, tuple)):
            raise TypeError('Choice must be of type list/set/tuple.')

        self._arguments[name] = {'default': default,
                                 'required': required,
                                 'help': help,
                                 'dest': dest,
                                 'callback': callback,
                                 'choice': choice}

        if mutually_exclusive_group:
            self._groups[mutually_exclusive_group].add(name)

    def _parse_argument(self, name, scenario, help, callback=None, default=None,
                        dest=None, required=False, choice=None):
        """Search the scenario dict for a single allowed argument and parse it.

        Side effect: the argument is removed from the scenario dict if found.

        name : str
            Argument name, as specified in the Scenario class.
        scenario : dict
            Scenario dict as provided by the user or as parsed by the cli
            interface.
        help : str
            Help string of the argument
        callback : callable, optional (default=None)
            If given, will be called to transform the given argument.
        default : object, optional (default=None)
            Will be used as default value if the argument is not given by the
            user.
        dest : str, optional (default=None)
            Will be used as member name of the scenario.
        required : bool (default=False)
            If ``True``, the scenario will raise an Exception if the argument is
            not given.
        choice : list, optional (default=None)
            If given, the scenario checks whether the argument is in the
            list. If not, it raises an Exception.

        Returns
        -------
        str
            Member name of the attribute.
        object
            Value of the attribute.
        """
        normalized_name = name.lower().replace('-', '').replace('_', '')
        value = None

        # Allows us to pop elements in order to remove all parsed elements
        # from the dictionary
        for key in list(scenario.keys()):
            # Check all possible ways to spell an argument
            normalized_key = key.lower().replace('-', '').replace('_', '')
            if normalized_key == normalized_name:
                value = scenario.pop(key)

        if dest is None:
            dest = name.lower().replace('-', '_')

        if required is True:
            if value is None:
                raise ValueError('Required scenario argument %s not given.' %
                                 name)

        if value is None:
            value = default

        if value is not None and callable(callback):
            value = callback(value)

        if value is not None and choice:
            value = value.strip()
            if value not in choice:
                raise ValueError('Argument %s can only take a value in %s, '
                                 'but is %s' % (name, choice, value))

        return dest, value

    def _add_arguments(self):
        # Add allowed arguments
        # If you add an argument that should be set through the
        # scenario.txt-file, please make sure to add it to
        # smac/utils/scenario_options
        self.add_argument(name='algo', help=None, dest='ta',
                          callback=shlex.split)
        self.add_argument(name='execdir', default='.', help=None)
        self.add_argument(name='deterministic', default="0", help=None,
                          callback=_is_truthy)
        self.add_argument(name='paramfile', help=None, dest='pcs_fn',
                          mutually_exclusive_group='cs')
        self.add_argument(name='run_obj', help=None, default='runtime')
        self.add_argument(name='overall_obj', help=None, default='par10')
        self.add_argument(name='cutoff_time', help=None, default=None,
                          dest='cutoff', callback=float)
        self.add_argument(name='memory_limit', help=None)
        self.add_argument(name='tuner-timeout', help=None, default=numpy.inf,
                          dest='algo_runs_timelimit',
                          callback=float)
        self.add_argument(name='wallclock_limit', help=None, default=numpy.inf,
                          callback=float)
        self.add_argument(name='runcount_limit', help=None, default=numpy.inf,
                          callback=float, dest="ta_run_limit")
        self.add_argument(name='minR', help=None, default=1, callback=int,
                          dest='minR')
        self.add_argument(name='maxR', help=None, default=2000, callback=int,
                          dest='maxR')
        self.add_argument(name='instance_file', help=None, dest='train_inst_fn')
        self.add_argument(name='test_instance_file', help=None,
                          dest='test_inst_fn')
        self.add_argument(name='feature_file', help=None, dest='feature_fn')
        self.add_argument(name='output_dir', help=None,
                          default="smac3-output_%s" % (
                              datetime.datetime.fromtimestamp(
                                  time.time()).strftime(
                                  '%Y-%m-%d_%H:%M:%S')))
        self.add_argument(name='shared_model', help=None, default='0',
                          callback=_is_truthy)
        self.add_argument(name='instances', default=[[None]], help=None,
                          dest='train_insts')
        self.add_argument(name='test_instances', default=[[None]], help=None,
                          dest='test_insts')
        self.add_argument(name='initial_incumbent', default="DEFAULT",
                          help=None, dest='initial_incumbent',
                          choice=['DEFAULT', 'RANDOM'])
        # instance name -> feature vector
        self.add_argument(name='features', default={}, help=None,
                          dest='feature_dict')
        # ConfigSpace object
        self.add_argument(name='cs', help=None, mutually_exclusive_group='cs')

    def _transform_arguments(self):
        self.n_features = len(self.feature_dict)
        self.feature_array = None

        if self.overall_obj[:3] in ["PAR", "par"]:
            self.par_factor = int(self.overall_obj[3:])
        elif self.overall_obj[:4] in ["mean", "MEAN"]:
            self.par_factor = int(self.overall_obj[4:])
        else:
            self.par_factor = 1

        # read instance files
        if self.train_inst_fn:
            if os.path.isfile(self.train_inst_fn):
                self.train_insts = self.in_reader.read_instance_file(
                    self.train_inst_fn)
            else:
                self.logger.error(
                    "Have not found instance file: %s" % (self.train_inst_fn))
                sys.exit(1)
        if self.test_inst_fn:
            if os.path.isfile(self.test_inst_fn):
                self.test_insts = self.in_reader.read_instance_file(
                    self.test_inst_fn)
            else:
                self.logger.error(
                    "Have not found test instance file: %s" % (
                        self.test_inst_fn))
                sys.exit(1)

        self.instance_specific = {}

        def extract_instance_specific(instance_list):
            insts = []
            for inst in instance_list:
                if len(inst) > 1:
                    self.instance_specific[inst[0]] = " ".join(inst[1:])
                insts.append(inst[0])
            return insts

        self.train_insts = extract_instance_specific(self.train_insts)
        if self.test_insts:
            self.test_insts = extract_instance_specific(self.test_insts)

        # read feature file
        if self.feature_fn:
            if os.path.isfile(self.feature_fn):
                self.feature_dict = self.in_reader.read_instance_features_file(
                    self.feature_fn)[1]

        if self.feature_dict:
            self.feature_array = []
            for inst_ in self.train_insts:
                self.feature_array.append(self.feature_dict[inst_])
            self.feature_array = numpy.array(self.feature_array)
            self.n_features = self.feature_array.shape[1]
            
            # reduce dimensionality of features of larger than PCA_DIM
            if self.feature_array.shape[1] > self.PCA_DIM:
                X = self.feature_array
                # scale features
                X = MinMaxScaler().fit_transform(X)
                X = numpy.nan_to_num(X) # if features with max == min
                #PCA
                pca = PCA(n_components=self.PCA_DIM)
                self.feature_array = pca.fit_transform(X)
                self.n_features = self.feature_array.shape[1]
                # update feature dictionary
                for feat, inst_ in zip(self.feature_array, self.train_insts):
                    self.feature_dict[inst_] = feat

        # read pcs file
        if self.pcs_fn and os.path.isfile(self.pcs_fn):
            with open(self.pcs_fn) as fp:
                pcs_str = fp.readlines()
                self.cs = pcs.read(pcs_str)
                self.cs.seed(42)
        elif self.pcs_fn:
            self.logger.error("Have not found pcs file: %s" %
                              (self.pcs_fn))
            sys.exit(1)

        # you cannot set output dir to None directly
        # because None is replaced by default always
        if self.output_dir == "":
            self.output_dir = None
            self.logger.debug("Deactivate output directory.")
        else:
            self.logger.info("Output to %s" % (self.output_dir))

    def _write(self):
        """Write scenario to file in a format that can be easily retrieved using
        the input_reader. Will overwrite if file already exists.

        Sideeffect: creates output-directory if it doesn't exist.
        """
        self.logger.debug("Writing scenario-file to {}.".format(self.output_dir))
        # Reverse mapping of option-names to internal -> external
        options_int2ext = {v: k for k, v in self.options_ext2int.items()}

        # We need to modify certain values when writing them
        def modify(key, value):
            ''' returns the correct value to write into a file '''
            if key == 'ta' : return " ".join(value)
            else: return value

        # Create output-dir if necessary
        if not os.path.isdir(self.output_dir):
            self.logger.debug("Output directory does not exist! Will be created.")
            try:
                os.makedirs(self.output_dir)
            except OSError:
                self.logger.error(
                    "Could not make output directory: {}".format(self.output_dir))
                sys.exit(3)

        # Write into output_dir/scenario.txt
        path = os.path.join(self.output_dir, "scenario.txt")
        d = self.__getstate__()
        with open(path, 'w') as f:
            for key in d:
                if key in options_int2ext and d[key] != None:
                    f.write("{} = {}\n".format(options_int2ext[key], modify(key, d[key])))

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.logger = logging.getLogger("scenario")
