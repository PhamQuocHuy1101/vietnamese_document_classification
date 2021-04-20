import yaml
import sys

sys.path.append('.')

def read_config(file_path):
    global MODEL
    global PATH
    global TRAIN
    with open(file_path, 'r') as f:
        cfg = yaml.safe_load(f)
        MODEL = cfg['MODEL'] # AU THU <3 PHAM HUY
        PATH = cfg['PATH']
        TRAIN = cfg['TRAIN']
    print("Load config finish")

# Read local config
# read_config('./config/main.yml')