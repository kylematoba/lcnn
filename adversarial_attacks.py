import math
import functools
import logging
from typing import Callable, Dict

import cleverhans.torch.attacks.fast_gradient_method
import cleverhans.torch.attacks.projected_gradient_descent
import cleverhans.torch.attacks.carlini_wagner_l2

import utils.logging

logging_level = 15
# Logging commands
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)
standard_streamhandler = utils.logging.get_standard_streamhandler()
logger.addHandler(standard_streamhandler)
# End Logging


def get_attacks_dict() -> Dict[str, Callable]:
    # https://cleverhans.readthedocs.io/en/v.2.1.0/source/attacks.html
    fgm_base = cleverhans.torch.attacks.fast_gradient_method.fast_gradient_method
    pgd_base = cleverhans.torch.attacks.projected_gradient_descent.projected_gradient_descent
    cwl2_base = cleverhans.torch.attacks.carlini_wagner_l2.carlini_wagner_l2

    fgm000000 = functools.partial(fgm_base, eps=.000000, norm=math.inf)
    fgm000001 = functools.partial(fgm_base, eps=.000001, norm=math.inf)
    fgm000010 = functools.partial(fgm_base, eps=.000010, norm=math.inf)
    fgm000100 = functools.partial(fgm_base, eps=.000100, norm=math.inf)
    fgm001000 = functools.partial(fgm_base, eps=.001000, norm=math.inf)
    fgm010000 = functools.partial(fgm_base, eps=.010000, norm=math.inf)
    fgm100000 = functools.partial(fgm_base, eps=.100000, norm=math.inf)

    l2_fgm000000 = functools.partial(fgm_base, eps=.000000, norm=2)
    l2_fgm000001 = functools.partial(fgm_base, eps=.000001, norm=2)
    l2_fgm000010 = functools.partial(fgm_base, eps=.000010, norm=2)
    l2_fgm000100 = functools.partial(fgm_base, eps=.000100, norm=2)
    l2_fgm001000 = functools.partial(fgm_base, eps=.001000, norm=2)
    l2_fgm010000 = functools.partial(fgm_base, eps=.010000, norm=2)

    l2_fgm100000 = functools.partial(fgm_base, eps=.100000, norm=2)
    l2_fgm200000 = functools.partial(fgm_base, eps=.100000, norm=2)
    l2_fgm300000 = functools.partial(fgm_base, eps=.100000, norm=2)

    eps_multiplier = .1
    l2_pgd_3_100000 = functools.partial(pgd_base, eps=.1, eps_iter=.1 * eps_multiplier, nb_iter=3, norm=2)
    l2_pgd_3_200000 = functools.partial(pgd_base, eps=.2, eps_iter=.2 * eps_multiplier, nb_iter=3, norm=2)
    l2_pgd_3_300000 = functools.partial(pgd_base, eps=.3, eps_iter=.3 * eps_multiplier, nb_iter=3, norm=2)

    l2_pgd_10_100000 = functools.partial(pgd_base, eps=.1, eps_iter=.1 * eps_multiplier, nb_iter=10, norm=2)
    l2_pgd_10_200000 = functools.partial(pgd_base, eps=.2, eps_iter=.2 * eps_multiplier, nb_iter=10, norm=2)
    l2_pgd_10_300000 = functools.partial(pgd_base, eps=.3, eps_iter=.3 * eps_multiplier, nb_iter=10, norm=2)

    cwpl2 = functools.partial(cwl2_base, n_classes=100)
    attacks_dict = {
        "l2_pgd_3_100000": l2_pgd_3_100000,
        "l2_pgd_3_200000": l2_pgd_3_200000,
        "l2_pgd_3_300000": l2_pgd_3_300000,
        "l2_pgd_10_100000": l2_pgd_10_100000,
        "l2_pgd_10_200000": l2_pgd_10_200000,
        "l2_pgd_10_300000": l2_pgd_10_300000,
        "l2_fgm000000": l2_fgm000000,
        "l2_fgm000001": l2_fgm000001,
        "l2_fgm000010": l2_fgm000010,
        "l2_fgm000100": l2_fgm000100,
        "l2_fgm001000": l2_fgm001000,
        "l2_fgm010000": l2_fgm010000,
        "l2_fgm100000": l2_fgm100000,
        "l2_fgm200000": l2_fgm200000,
        "l2_fgm300000": l2_fgm300000,
        "fgm000000": fgm000000,
        "fgm000001": fgm000001,
        "fgm000010": fgm000010,
        "fgm000100": fgm000100,
        "fgm001000": fgm001000,
        "fgm010000": fgm010000,
        "fgm100000": fgm100000,
        "cwl2": cwpl2,
        "zero": None
    }
    return attacks_dict


if __name__ == "__main__":
    attacks_dict = get_attacks_dict()
