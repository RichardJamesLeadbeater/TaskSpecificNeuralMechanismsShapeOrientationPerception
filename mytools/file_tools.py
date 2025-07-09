import pickle

def unpickle(filepath):
    with open(filepath, 'rb') as f:
        file = pickle.load(f)

    return file
