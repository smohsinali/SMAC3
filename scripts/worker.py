#!/bin/python

__author__ = "Marius Lindauer"
__license__ = "GPLv3"
__version__ = "0.0.1"

import sys
import os
import pickle
import logging
import time

from pymongo import MongoClient
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from smac.tae.execute_ta_run import StatusType


class Worker(object):

    def __init__(self,
                 tae_runner_fn,
                 ip_mongodb='127.0.0.1',
                 port=27017,
                 db="smac3",
                 collection="tae_run",
                 max_wait=120):
        '''
        Constructor

        Arguments
        ---------
        tae_runner_fn: str
            filename of pickled target algorithm runner object
        ip_mongodb: str
            ip adresse of mongodb service
        port: int
            port on <ip> 
        db: str
            database name
        collection: str
            name of collection in <db>
        max_wait: int
            maximum time to wait for new jobs in DB (secs)
        '''
        self.client = MongoClient("mongodb://%s:%d" % (ip_mongodb, port))
        self.collection = self.client[db][collection]

        self.max_wait = max_wait
        self.tae_runner_fn = tae_runner_fn
        self.tae_runner = None
        
        self.PULL_INTERVALL = 5 # sec
        
        self.logger = logging.getLogger("MongoWorker")

    def _load_tae(self):
        with open(self.tae_runner_fn, "rb") as fp:
            self.tae_runner = pickle.load(fp)
            self.tae_runner.logger = logging.getLogger("TAERunner")

    def run(self):
        '''
            wait for new jobs in database,
            runs them with provided tae_runner
            and adds result to DB
        '''
        
        st = time.time()
        while time.time() - st < self.max_wait:
            while True:
                cursor = self.collection.find({"result.status": StatusType.UNKNOWN})
                try:
                    job = next(cursor)
                    successful = False
                    try:
                        self.collection.update_one({'_id': job['_id']}, {"$set": {
                            "result.status": -2
                        },
                            #"$currentDate": {"lastModified": True}
                        })
        
                        # load tae runner the first time we find new jobs in DB
                        if not self.tae_runner:
                            self._load_tae()
        
                        status, cost, runtime, additional_info = self.tae_runner.run(
                                                                    config=job["data"]["config"],
                                                                    instance=job["data"]["instance"],
                                                                    cutoff=job["data"]["cutoff"],
                                                                    seed=job["data"]["seed"],
                                                                    instance_specific=job["data"]["instance_specific"]
                                                                                     )
                        self.logger.debug("%s, %s, %f, %f" %(job['_id'], status, cost, runtime))
        
                        self.collection.update_one({'_id': job['_id']}, {"$set": {
                            "result.status": status,
                            "result.cost": cost,
                            "result.runtime": runtime,
                            "result.additional_info": additional_info
                        },
                        #   "$currentDate": {"lastModified": True}
                        })
                        successful = True
        
                    finally:
                        if not successful:
                            # return to UNKNOWN if something went wrong
                            self.collection.update_one({'_id': job['_id']}, {"$set": {
                                "result.status": StatusType.UNKNOWN
                            },
                            #    "$currentDate": {"lastModified": True}
                            })
                        st = time.time()
                except StopIteration:
                    break
        
            self.logger.debug("No further job found ... going to sleep for %d seconds (time left before exit: %d)" 
                              %(self.PULL_INTERVALL, self.max_wait - (time.time() - st) ))
            time.sleep(self.PULL_INTERVALL)
        

        #=======================================================================
        # cursor = self.collection.find()
        # for j in cursor:
        #     print(j)
        #=======================================================================


def parse_cmd():
    '''
        parses command line arguments
    '''
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    req_opts = parser.add_argument_group("Required Options")
    req_opts.add_argument("--db", required=True,
                          help="DB name")
    req_opts.add_argument("--collection", required=True,
                          help="collection name")
    req_opts.add_argument("--tae_runner_fn", required=True,
                          help="filename of pickled tae runner object")

    opt_opts = parser.add_argument_group("Optional Options")
    opt_opts.add_argument("--ip", default="127.0.0.1",
                          help="IP (or hostname) of mongodb master")
    opt_opts.add_argument("--port", default=27017, type=int,
                          help="IP (or hostname) of mongodb master")
    opt_opts.add_argument("--max_wait", default=120, type=int,
                          help="maximal time to wait whether new jobs are in DB (sec)")
    opt_opts.add_argument("--verbose", default="INFO", choices=["INFO", "DEBUG"],
                          help="verbose level")

    args_ = parser.parse_args()
    
    logging.basicConfig(level=args_.verbose)
    root_logger = logging.getLogger()
    root_logger.setLevel(args_.verbose)
    
    return args_

if __name__ == "__main__":

    args_ = parse_cmd()
    worker = Worker(tae_runner_fn=args_.tae_runner_fn,
                    ip_mongodb=args_.ip,
                    port=args_.port,
                    db=args_.db,
                    collection=args_.collection,
                    max_wait=args_.max_wait)
    worker.run()
