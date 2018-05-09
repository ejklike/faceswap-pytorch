import os

def mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory