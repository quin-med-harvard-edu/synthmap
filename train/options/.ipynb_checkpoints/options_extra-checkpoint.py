"""Read args from a YAML file instead of specifying them at command line. This improves reusability and readability of training configurations. """


import argparse
import sys 

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
    
class ArgsExtra():
    
    def __init__(self):
        self.args_init = sys.argv[1:]
        self.get_config()
        self.parse_config()
        self.check_unknown_args()
        
    def parse(self):
        
        return self.args
        
        
        
    def get_config(self):
        # make sure that config is in the file 
        assert self.args_init[0] == '--config', f"First argument must be --config"
        assert self.args_init[1].endswith('.yaml') or self.args_init[1].endswith('.yml'),f"--config must be followed by path to yaml file. Found only this: {self.args_init[1]}"
        self.config = self.args_init[1]
        self.args_init = self.args_init[2:]

        
    
    def parse_config(self):

        self.args = YAMLObject('DefaultConfig', (object,), {'source': self.config, 'namespace': 'defaults'})
        self.args.configfile = self.config  

        
    def check_unknown_args(self):
        # extract the remaining parameters and make sure they are key value pairs 
        assert len(self.args_init) % 2 == 0, f"Odd number of arguments. All args must be key value pairs "

        # extract pairs 
        keys = self.args_init[::2] # even
        values = self.args_init[1::2] #odd

        # make sure keys have '--' preceding
        assert all([k[:2]=='--' for k in keys]), f"All input arguments must be key value arguments. Keys must be specified by '--' preceding the name. E.g. --mask_type improved "
        keys = [k[2:] for k in keys]

        # create dictionary of key value pairs
        d = {}
        for k,v in zip(keys,values):
            d[k] = v

        # create a list of modified keys
        modified_keys = []
            
        # check if any of the keys are inside of the yaml file and replace them 
        for k in keys:
#             if k in self.args:
            # sub 
            if d[k].lower() == 'true':
                self.args[k] = True 
            elif d[k].lower() =='false':
                self.args[k] = False
            elif d[k].lower() =='none':
                self.args[k] = None
            elif d[k][0] == '[':
                assert d[k][-1] == ']', f"Please supply correct expression. Currently: {d[k]}"
                d[k]=d[k][1:-1] # remove brackets 
                l = d[k].split(',')
                if l[0].isdigit(): 
                    assert all([i.replace(".","").isdigit() for i in l]) # check that all are integers
                l = [float(i) for i in l] if all([i.replace(".","").isdigit() for i in l]) else l 
                if isinstance(l[0], float):
                    l = [int(i) for i in l] if all([i.is_integer() for i in l]) else l
                self.args[k] = l    
            else:
                self.args[k] = d[k]
                    
            modified_keys.append(k)
                
            # REMOVED UNKNOWN ARGS DUE TO CLASH
#             else:
#                 # add the key to 'unrecognized keys'
#                 if 'unknown_args' not in vars(self.args):
#                     # add as first element in the list
#                     self.args['unknown_args'] = {}
#                     self.args['unknown_args'][k] = d[k]
#                 else:
#                     # add to list 
#                     self.args['unknown_args'][k] = d[k]
                    
#         # print warning message to user
#         if 'unknown_args' in vars(self.args):
#         if 'verbose' in vars(self.args) and self.args['verbose']!='off':
#             input(f"\nThe following input argument is not specified in configfile (.yaml). \n{self.args['unknown_args']}\nDo you want to proceed? ")
        if modified_keys:
            print("\nThe following inputs were set based on extra input arguments: ")
            for k in modified_keys:
                
                print(f"{k}: {self.args[k]}")
            if 'verbose' in vars(self.args) and self.args['verbose']!='off':
                input("Do these match to actual input arguments? Please press any key to proceed. ")
            
            
if __name__ == '__main__':

    args = ArgsExtra().parse() if len(sys.argv) > 3 else Args().parse()

    from IPython import embed; embed()

    
# BASIC TESTS 

# cd ~/w/code/mwf/synth_unet/train_unet/options/
# yaml=~/w/code/mwf/synth_unet/train_unet/configs/test_only_do_not_delete/test_only_do_not_delete.yaml
# python options_extra.py --config $yaml --argument1 45 --argument2 89 --mask_threshold True --noisevariance [45,67]
    
    
    
# def test():
#     # playing around with arg parse 
#     import argparse
#     import sys
#     def main():

#         #parser = argparse.ArgumentParser()
#         #arguments = parser.parse_args()
#         args =sys.argv
#         args = args[1:]

#         # make sure that config is in the file 
#         assert args[0] == '--config', f"First argument must be config file"
#         assert args[1].endswith('.yaml') or args[1].endswith('.yml'),f"--config must be followed by path to yaml file. Found only this: {args[1]}"
#         config = args[1]
#         args = args[2:]


#         # extract the remaining parameters and make sure they are key value pairs 
#         assert len(args) % 2 == 0, f"Odd number of arguments. All args must be key value pairs "

#         # extract pairs 
#         keys = args[::2] # even
#         values = args[1::2] #odd


#         # create dictionary of key value pairs
#         d = {}
#         for k,v in zip(keys,values):
#             d[k] = v


#         # configure yaml file 
#         ... 

#         # check if any of the keys are inside of the yaml file and replace them 
#         for k in keys:
#             if k in config:
#                 config.k = d[k]
#             else:
#                 input(f"\nThe following input argument is not specified in configfile (.yaml). Do you want to proceed? ")
#                 # add the key to 'unrecognized keys'
#                 if 'unknown_args' not in config:
#                     # add as first element in the list
#                     config['unknown_args'] = [d[k]]
#                 else:
#                     # add to list 

#                     config['unknown_args'].append(d[k])






#         test_list[::2] + test_list[1::2]

#         from IPython import embed; embed()

#         args_dict = vars(arguments)


#     """

#     Runwith:

#         cd ~/w/code/mwf/synth_unet/train_unet/
#         python s20210813_log.py try=12 second 4
#         python s20210813_log.py --config configfilename --parameter2 valueofparameter2 

#     """        