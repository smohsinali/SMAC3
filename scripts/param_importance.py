#!/bin/python

import os
import sys
import inspect
import logging

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from smac.parameter_importance.epmimportance import EPMImportance

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


if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    req_opts = parser.add_argument_group("Required Options")
    req_opts.add_argument("--scenario_file", required=True,
                          help="scenario file in AClib format")
    req_opts.add_argument("--runhistory", required=True,
                          help="runhistory file")

    req_opts.add_argument("--verbose_level", default=logging.INFO,
                          choices=["INFO", "DEBUG"],
                          help="random seed")
    req_opts.add_argument("--type", default='fs',
                          choices=['fs', 'ab', 'forwardselection', 'ablation'],
                          help='Importance evaluation approach')
    req_opts.add_argument('--cost', default='cost',
                          choices=['cost', 'log', 'eips'],
                          help='Cost function.')
    req_opts.add_argument('--trajectory_file', default=None,
                          help='Used to determine the incumbent.')

    args_ = parser.parse_args()

    logging.basicConfig(level=args_.verbose_level)

    if args_.type in ['ab', 'ablation']:
        assert args_.trajectory_file is not None, 'The traj file is needed to determine the incumbent for ablation!'

    epm_imp = EPMImportance(scenario_fn=args_.scenario_file,
                            runhistory_fn=args_.runhistory, traj_fn=args_.trajectory_file, cost_=args_.cost)

    importances = epm_imp.run(args_.type)
    oobe = epm_imp.model.rf.out_of_bag_error()
    logging.info('Importances: %s' % str(importances))  # ordered dict: key -> tuple (percentage, runtime)
    logging.info('Out-of-bag-error: %s' % str(oobe))  # float
