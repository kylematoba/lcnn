# Code to accompany "Efficient Training of Low-Curvature Neural Networks"

by Suraj Srinivas, Kyle Matoba, Himabindu Lakkaraju, and François Fleuret.

NeurIPS 2022.  

The [appendix](https://openreview.net/attachment?id=2B2xIJ299rx&name=supplementary_material) 
contains further proofs and analysis. 

## Configuration

Set paths to where the code lives, data is located, where you want logs to be 
written, etc. in `path_config.py` (we've done it by setting the environment
variables `CURVATURE_HOME` and `DATA_HOME`, because that's convenient for our
setup, but don't feel obliged to follow it).

There are no dependencies beyond those in the `requirements.txt`. 

Torch >= 1.11.0 is a requirement -- due to a change in the way that the parameterization is handled internally.

## Trainable centered softplus, clipped batchnorm, "real" spectral normalization. 
You might just be interested in the three main architectural novelties:
 1. [Trainable centered softplus](abc.com)
 2. [Clipped batchnorm](abc.com)
 3. ["Real spectral normalization"](abc.com)
  - (this is a minimally-modified version of the original, at https://github.com/uclaopt/Provable_Plug_and_Play/tree/master/training/model)

## Training models 
The main entry point is `train_curvature.py`. Table 1 of the paper can be generated via: 

```
python3 train_curvature.py --model-arch=resnet18 --dataset=cifar100  # Standard
python3 train_curvature.py --regularizer=curvature_proxy --model-arch=resnet18smoothconv --dataset=cifar100  # LCNNs
python3 train_curvature.py --regularizer=gnorm --model-arch=resnet18 --dataset=cifar100  # GradReg
python3 train_curvature.py --regularizer=curvature_and_gnorm --model-arch=resnet18smoothconv --dataset=cifar100  # LCNNs + GradReg

python3 train_curvature.py --regularizer=cure --model-arch=resnet18 --dataset=cifar100  # CURE
python3 train_curvature.py ????? # Softplus + Wt. Decay
python3 train_curvature.py --model-arch=resnet18 --dataset=cifar100  --adv_attack l2_pgd_3_100000  # Adversarial Training
```

Other options, e.g. other datasets, architectures, etc. are further discussed at `python3 train_curvature.py --help`.

Note that currently (October 2022 / cleverhans af09028a9b1307a432bca01f49e9c3568251aa8c)
PGD is broken for L2 attacks due to this inplace operation:
https://github.com/cleverhans-lab/cleverhans/blob/af09028a9b1307a432bca01f49e9c3568251aa8c/cleverhans/torch/utils.py#L38.
In order to run the experiment with PGD, we monkey-patched this multiply to not be in-place.

## Other experiments
To keep it simple, we have not uploaded all of the code to reproduce all of our experiments. 
If you're interested in understanding the provenance of some number in the paper that is not in Table 1
please contact Suraj or Kyle at the addresses given in [the paper](https://arxiv.org/abs/2206.07144).

## Citation
Here's a bibtex entry for the paper:

```
@misc{Srinivas2022,
  doi = {10.48550/ARXIV.2206.07144},
  url = {https://arxiv.org/abs/2206.07144},
  author = {Srinivas, Suraj and Matoba, Kyle and Lakkaraju, Himabindu and Fleuret, Francois},
  title = {Flatten the Curve: Efficiently Training Low-Curvature Neural Networks},
  year = {2022},
}

```

