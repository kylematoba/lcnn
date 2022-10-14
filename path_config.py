import os
from typing import Dict


def get_paths() -> Dict[str, str]:
    database = os.environ["DATA"]
    curvature = os.environ["CURVATURE_HOME"]

    cifar101 = os.path.join(curvature, "cifar101")
    cifar10 = database + "/cifar10"
    cifar100 = database + "/cifar100"
    cifar10c = database + "/CIFAR-10-C"
    saved_models = os.path.join(curvature, "saved_models")

    svhn = os.path.join(curvature, "svhn")
    logs = os.path.join(curvature, "logs")
    paths = {
        "cifar10c": cifar10c,
        "cifar100": cifar100,
        "cifar10": cifar10,
        "cifar101": cifar101,
        "curvature": curvature,
        "svhn": svhn,
        "logs": logs,
        "saved_models": saved_models,
    }
    return paths


if __name__ == "__main__":
    paths = get_paths()
