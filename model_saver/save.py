import pickle
import time
import os

BASE_PATH = 'data/saved_models'
SEPARATOR = '§'


def save_model(model, name):
    """
    Saves an ML model into a file in local files
    Args:
        model(any): Fitted ML model
        name(string): Name to use for backup file
    """
    check_dir_then_create(BASE_PATH)
    file_name = f'{BASE_PATH}/{name}{SEPARATOR}{time.time()}'
    pickle.dump(model, open(file_name, 'wb'))


def load_model(name):
    """
    Load an ML previously saved in local files
    Args:
        name(string): Name to use to retrieve ML model
    """
    check_dir_then_create(BASE_PATH)
    file_names = [f for f in os.listdir(BASE_PATH) if os.path.isfile(os.path.join(BASE_PATH, f))]
    names = list(map(lambda x: x.split(SEPARATOR)[0], file_names))
    model_index = names.index(name)
    return pickle.load(open(f'{BASE_PATH}/{file_names[model_index]}', 'rb'))


def check_dir_then_create(path):
    """
    Check if path exists, if not: create every necessary directories
    Args:
        path(string): Path to check
    """
    if not os.path.exists(path):
        os.makedirs(path)


"""
Usage example:

save.save_model(myModel, 'LogReg_1')
retrieved_model = save.load_model('LogReg_1')
"""