import logging
import sys

class CustomLogger:
    def __init__(self):
        self.rootlogger = logging.getLogger()
        #logging.basicConfig(filename="newfile.log",
        #            format='%(asctime)s %(message)s',
        #            filemode='w')
        self.rootlogger.setLevel(logging.INFO)  
        handler = logging.StreamHandler(sys.stdout)
        #handler.setLevel(logging.DEBUG)
        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #handler.setFormatter(formatter)
        self.rootlogger.addHandler(handler)
    

    def logInfo(self,routine,message):
        self.rootlogger.info(f"{routine}::{message}")

    def logError(self,routine,message):
        self.rootlogger.error(f"{routine}::{message}")