# THREE OPTIONS 

# 1. Load YAML AS DICTIONARY
import yaml 
with open('s20210311-test-ivim.yaml') as f: 
    args = yaml.safe_load(f)    
print(args)

# 2. Load YAML AS OBJECT
from yaml2object import YAMLObject
configfile="s20210311-test-ivim.yaml"
args = YAMLObject('DefaultConfig', (object,), {'source': configfile, 'namespace': 'defaults'})

# 3. LOAD JSON as OBJECT (from attention unet)
import json
import collections
def json_file_to_pyobj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json2obj(open(filename).read())
args = json_file_to_pyobj("s20210311-test-ivim.json")