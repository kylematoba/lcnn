import os
from typing import Dict


def get_paths() -> Dict[str, str]:
    data_home = os.environ["DATA_HOME"]
    curvature_home = os.environ["CURVATURE_HOME"]

    cifar101 = os.path.join(curvature_home, "cifar101")
    cifar10 = data_home + "/cifar10"
    cifar100 = data_home + "/cifar100/cifar-100-python/"
    svhn = os.path.join(curvature_home, "svhn")

    saved_models = os.path.join(curvature_home, "saved_models")
    logs = os.path.join(curvature_home, "logs")
    paths = {
        "cifar100": cifar100,
        "cifar10": cifar10,
        "cifar101": cifar101,
        "curvature": curvature_home,
        "svhn": svhn,
        "logs": logs,
        "saved_models": saved_models,
    }
    return paths


if __name__ == "__main__":
    paths = get_paths()
