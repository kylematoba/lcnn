import os
from typing import Dict

def get_paths() -> Dict[str, str]:
    pwd = os.getcwd() 
    
    #Parent directory
    data_home = os.path.dirname(pwd) + '/datasets' 
    cifar10 = data_home + "/cifar10"
    cifar100 = data_home + "/cifar100/cifar-100-python/"
    svhn = data_home + '/svhn'

    saved_models = os.path.join(pwd, "saved_models")
    logs = os.path.join(pwd, "logs")
    paths = {
        "cifar100": cifar100,
        "cifar10": cifar10,
        "pwd": pwd,
        "svhn": svhn,
        "logs": logs,
        "saved_models": saved_models,
    }
    return paths


if __name__ == "__main__":
    paths = get_paths()
