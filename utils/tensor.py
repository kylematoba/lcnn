import string
import itertools
from typing import Any, List, Iterable, Tuple

import torch
import numpy as np

Partition = Any


def diag1d_to_diag3d(diag1d: torch.Tensor) -> torch.Tensor:
    assert 1 == len(diag1d.shape)
    diag3d = torch.diag_embed(torch.diag(diag1d),
                              offset=0, dim1=1, dim2=2)
    return diag3d


def diag3d_to_diag1d(diag3d: torch.Tensor) -> torch.Tensor:
    assert 3 == len(diag3d.shape)
    diag1d = torch.einsum("iii->i", diag3d)
    return diag1d


def covariant_multilinear_matrix_multiplication(a: torch.Tensor,
                                                mlist: List[torch.Tensor]) -> torch.Tensor:
    order = a.ndim
    assert order == len(mlist)
    assert order < 25, "More than 25 orders?"

    base_indices = string.ascii_letters
    indices = base_indices[:order]
    next_index = base_indices[order]

    val = a
    for idx in range(order):
        # idx = 0; print(idx)
        resp_str = indices[:idx] + next_index + indices[idx+1:]
        einsum_str = indices + f",{indices[idx]}{next_index}->{resp_str}"
        val = torch.einsum(einsum_str, val, mlist[idx])
    return val


def elem_subset(elem: List[int],
                s: List[int]) -> bool:
    # todo: in torch 1.10, there is torch.isin:
    # https://discuss.pytorch.org/t/is-there-pytorch-function-which-is-similar-to-numpy-in1d/3127/5.
    tf = np.in1d(elem, np.array(s, dtype=object)).all()
    return tf


def ge_partition(pi1: Partition,
                 pi2: Partition) -> bool:
    """
    A partition pi1 \\in P[k] is called a refinement of pi2 \\in P[k] if
    each block of pi1 is a subset of some block of pi2; conversely,
    π2 is said to be a coarsening of π1. This relationship defines
    a partial order, expressed as pi1 ≤ pi2.
    """
    # pi1_tuple = [tuple(_) for _ in pi1]
    # pi2_tuple = [tuple(_) for _ in pi2]
    #
    is_elem_subset = [elem_subset(_, pi2) for _ in pi1]
    is_ge = all(is_elem_subset)
    return is_ge


def _flatten_list_of_lists(ll: List[list]) -> list:
    flattened = [x for l in ll for x in l]
    return flattened


def is_ordered_partition_of_k(pi: Partition,
                              k: int) -> bool:
    pi_flat = _flatten_list_of_lists(pi)
    tf = pi_flat == list(range(k))
    return tf


def prod(xs: Iterable):
    p = 1
    for x in xs:
        p *= x
    return p


def pi_dim_mapping(pi: Partition,
                   in_dims: torch.Size) -> List[int]:
    out_dims = [prod(in_dims[__] for __ in _) for _ in pi]
    return out_dims


def unfold(a: torch.Tensor,
           pi: List[List[int]]) -> torch.Tensor:
    order = a.ndim
    ashapes = a.shape

    assert is_ordered_partition_of_k(pi, order)

    pi_flattened = _flatten_list_of_lists(pi)
    assert 0 in pi_flattened, "zero indexed"
    assert sorted(range(order)) == pi_flattened
    matched_sizes = [[ashapes[_] for _ in sl] for sl in pi]
    dims_out = [prod(_) for _ in matched_sizes]

    unfolded = a.reshape(*dims_out)
    return unfolded


def tuple_index_list(lst: list,
                     tup: Tuple[int]) -> List[list]:
    lowers = [0] + [_ + 1 for _ in tup]
    uppers = [_ + 1 for _ in tup] + [max(lst) + 1]

    num_tup = len(lowers)
    indexed = [lst[lowers[_]:uppers[_]] for _ in range(num_tup)]
    return indexed


def generate_all_partitions(dim: int) -> List[Partition]:
    # adapted from https://docs.python.org/3/library/itertools.html#itertools-recipes
    s = list(range(dim - 1))
    sizes = range(1, len(s) + 1)
    index_partition_chain = itertools.chain.from_iterable(itertools.combinations(s, r) for r in sizes)
    index_partition = list(index_partition_chain)

    ss = list(range(dim))
    all_partitions = [tuple_index_list(ss, i) for i in index_partition]
    return all_partitions


def bona_fide_spectral_norm(a: torch.Tensor) -> float:
    # compute using proposition 4.1: check all possible unfoldings down to 2d.
    k = a.ndim
    if k <= 2:
        sn = torch.linalg.norm(a, ord=2)
    else:
        # make recursive calls:
        all_partitions = generate_all_partitions(k)
        used_partitions = all_partitions[:-1]
        num_partitions = len(used_partitions)
        sub_spectral_norms = [None] * num_partitions

        for idx, partition in enumerate(used_partitions):
            aunfolded = unfold(a, partition)
            assert aunfolded.ndim < a.ndim
            sub_spectral_norms[idx] = bona_fide_spectral_norm(aunfolded)
        sn = min(sub_spectral_norms)
    return sn


def reduce(center: torch.Tensor,
           before: torch.Tensor,
           after: torch.Tensor) -> torch.Tensor:
    es3d_2d = "ijk,kl->ijl"
    es2d_3d = "li,ijk->ljk"

    assert after.shape[0] == center.shape[1]
    assert before.shape[1] == center.shape[0]
    assert before.shape[1] == center.shape[-1]

    val0 = torch.einsum(es2d_3d, before, center)
    val1 = torch.einsum(es3d_2d, val0, before.T)
    val2 = torch.einsum("ijk,jl->ilk", val1, after)
    return val2


if __name__ == "__main__":
    pass
