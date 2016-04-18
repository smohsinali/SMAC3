import logging
import time
import sys
import pickle
from smac.tae.execute_ta_run import ExecuteTARun, StatusType

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "GPLv3"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class ExecuteTAMongoDB(ExecuteTARun):

    """
        executes a target algorithm run with a given configuration
        on a given instance and some resource limitations

        Attributes
        ----------
        ta : string
            the command line call to the target algorithm (wrapper)
    """

    def __init__(self, tae_runner,
                 ip_mongodb='127.0.0.1',
                 port=27017,
                 db="smac3",
                 collection="tae_run"):
        """
        Constructor

        Parameters
        ----------
            ta_runner: ExecuteTARun()
                instance of ExecuteTARun that will be used in the client to run the TA,
                needs to pickleable
            ip_mongodb: str
                IP address (or hostname) of mongodb 
            port: int
                port number of mongodb
            db: str
                name of database
            collection: str
                name of collection in db
        """
        self.logger = logging.getLogger("ExecuteTARun")

        # imported in the constructor to have the package optional
        from pymongo import MongoClient
        self.client = MongoClient("mongodb://%s:%d" % (ip_mongodb, port))
        self.collection = self.client[db][collection]

        # TODO: polling should be avoided
        self.SLEEP_TIME = 1  # sec

        self.WARN_FACTOR = 10
        self.ABORT_FACTOR = 100

        # this requires that SMAC and the worker is working in the same
        # directory
        with open("tae_runner.pkl", "wb") as fp:
            pickle.dump(tae_runner, fp)

    def run(self, config, instance,
            cutoff=99999999999999.,
            seed=12345,
            instance_specific="0"):
        """
            runs target algorithm <self.ta> with configuration <config> on
            instance <instance> with instance specifics <specifics>
            for at most <cutoff> seconds and random seed <seed>

            Parameters
            ----------
                config : dictionary
                    dictionary param -> value
                instance : string
                    problem instance
                cutoff : double
                    runtime cutoff
                seed : int
                    random seed
                instance_specific: str
                    instance specific information (e.g., domain file or solution)

            Returns
            -------
                status: enum of StatusType (int)
                    {SUCCESS, TIMEOUT, CRASHED, ABORT}
                cost: float
                    cost/regret/quality (float) (None, if not returned by TA)
                runtime: float
                    runtime (None if not returned by TA)
                additional_info: dict
                    all further additional run information
        """

        param_dict = {}
        for p in config:
            if not config[p] is None:
                param_dict[p] = config[p]

        insert_time = time.time()
        insert_obj = self.collection.insert_one(
            {"data": {
                "config": param_dict,
                "instance": instance,
                "cutoff": cutoff,
                "seed": seed,
                "instance_specific": instance_specific,
            },
                "result": {
                "status": StatusType.UNKNOWN,
                "cost": 99999999999999,
                "runtime": 99999999999999,
                "additional_info": {}
            }
            })

        warn_factor = self.WARN_FACTOR
        status = StatusType.UNKNOWN
        while status == StatusType.UNKNOWN:
            time.sleep(self.SLEEP_TIME)
            dat = list(
                self.collection.find({'_id': insert_obj.inserted_id}))[0]
            status = dat["result"]["status"]

            if time.time() - insert_time > cutoff * warn_factor:
                self.logger.warn(
                    "TA has not returned since %.2f secs -- more than %d times more than expected" % (time.time() - insert_time, warn_factor))
                warn_factor *= 2
            if time.time() - insert_time > cutoff * self.ABORT_FACTOR:
                self.logger.error("TA has not returned since %.2f secs -- more than %d times more than expected" % (
                    time.time() - insert_time, self.ABORT_FACTOR))
                self.logger.error("ABORT SMAC run")
                sys.exit(44)

        return status, dat["result"]["cost"], dat["result"]["runtime"], dat["result"]["addtional_info"]
