import os

def mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory

def get_face_ids(input_dir='./data'):
    return sorted(os.listdir(input_dir))