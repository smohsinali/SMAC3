'''
Created on Nov 19, 2015

@author: lindauer
'''
import unittest
import shlex
import logging
import sys
import os
import shlex
from subprocess import Popen

from pymongo import MongoClient

from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_ta_mongodb import ExecuteTAMongoDB
from smac.tae.execute_ta_run import StatusType


class TaeMongoDBTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(stream=sys.stderr)
        self.logger = logging.getLogger()  # root logger
        self.logger.setLevel(logging.DEBUG)

    def test_run(self):
        '''
            running some simple algo via the mongodb worker interface
            ATTENTION: The DB "smac3_test" will be dropped in the beginning of the test

            Requirement
            -----------
            mongod (mongodb service) needs to be running on localhost

            pymongo needs to be installed
        '''

        DB = "smac3_test"
        collection = "tae_run_test"
        tae_pkl = "tae_runner.pkl"
        ip_mongodb = '127.0.0.1'
        port = 27017

        client = MongoClient("mongodb://%s:%d" % (ip_mongodb, port))
        client.drop_database(DB)

        cmd = "python scripts/worker.py --db %s --collection %s --tae_runner_fn %s" % (
            DB, collection, tae_pkl)
        cmd = shlex.split(cmd)
        p = Popen(cmd, preexec_fn=os.setpgrp)

        tae = ExecuteTARunOld(
            ta=shlex.split("python test/tae/dummy_ta_wrapper.py 1"), logger=None)

        mtae = ExecuteTAMongoDB(tae_runner=tae,
                                ip_mongodb=ip_mongodb,
                                port=port,
                                db=DB,
                                collection=collection)

        status, cost, runtime, ar_info = mtae.run(
            config={}, instance="", cutoff=0.1)

        assert status == StatusType.SUCCESS
        assert cost == 0.1
        assert runtime == 0.1

        status, cost, runtime, ar_info = mtae.run(
            config={}, instance="", cutoff=1)

        assert status == StatusType.SUCCESS
        assert cost == 1
        assert runtime == 1

        p.terminate()

if __name__ == "__main__":
    unittest.main()
