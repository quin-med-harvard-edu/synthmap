"""Read args from a YAML file instead of specifying them at command line. This improves reusability and readability of training configurations. """

import argparse

from yaml2object import YAMLObject

class Args():
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--config',type=str,required=True,help='configuration file for test-train')
        
    def parse(self):
        args_init = self.parser.parse_args()
    
        configfile=args_init.config
        args = YAMLObject('DefaultConfig', (object,), {'source': configfile, 'namespace': 'defaults'})
        args.configfile = args_init.config

        return args