'''
Created on Nov 19, 2015

@author: lindauer
'''
import unittest
import shlex
import logging

from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_ta_mongodb import ExecuteTAMongoDB
from smac.tae.execute_ta_run import StatusType


class TaeMongoDBTest(unittest.TestCase):

    def test_run(self):
        '''
            running some simple algo via the mongodb worker interface
            
            Requirement
            -----------
            mongod (mongodb service) needs to be running on localhost
        '''
        tae = ExecuteTARunOld(ta=shlex.split("python dummy_ta_wrapper.py 1"), logger=None)
        mtae = ExecuteTAMongoDB(tae_runner=tae,
                                ip_mongodb='127.0.0.1',
                                port=27017,
                                db="smac3_test",
                                collection="tae_run")

        status, cost, runtime, ar_info = mtae.run(config={}, instance="", cutoff=1)
        
        
        
        assert status == StatusType.SUCCESS
        assert cost == 1.0
        assert runtime == 1.0

        print(status, cost, runtime)


if __name__ == "__main__":
    unittest.main()
