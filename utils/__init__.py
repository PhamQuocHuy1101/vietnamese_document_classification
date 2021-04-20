import pickle

def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        return obj