# Code to accompany "Efficient Training of Low-Curvature Neural Networks"

by Suraj Srinivas, Kyle Matoba, Himabindu Lakkaraju, and François Fleuret.

NeurIPS 2023.  

The [appendix](https://openreview.net/attachment?id=2B2xIJ299rx&name=supplementary_material) 
contains furter proofs and analysis. 

## Configuration

We assume that you have set two environment variables: 
`CURVATURE_HOME`, containing this code, and 
`DATA_HOME`, which contains or will be used to store the datasets in the 
format that the Pytorch dataloader will download.   

You can do this temporarily as described [here](https://stackoverflow.com/questions/57009481/running-python-script-with-temporary-environment-variables),
and if that is not convenient, you can more directly hard code paths at `get_paths.py`.

There are no dependencies beyond those in the `requirements.txt`. 

Torch >= 1.11.0 is a requirement -- due to a change in the way that the parameterization is handled internally.

## Trainable centered softplus, clipped batchnorm, "real" spectral normalization. 
You might just be interested in the three main 
 1. [Centered softplus](abc.com)
 2. [Clipped batchnorm](abc.com)
 3. ["Real spectral normalization"](abc.com)
  - (this should just be a minimally-modified version of , at https://github.com/uclaopt/Provable_Plug_and_Play/tree/master/training/model)

## Training models 
The main entry point is `train_curvature.py`. Table 1 of the paper can be generate via: 

To train ResNet18 models, use the following commands to train LCNNs, LCNN+GradReg, and GradReg models respectively

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
please contact @suraj-srinivas at ssrinivas@seas.harvard.edu or @kylematoba at kyle.matoba@epfl.ch.

## Citation
Here's a bibtex entry for the paper:
```

```