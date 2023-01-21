# Low Curvature Neural Networks

This is the reference code for the NeurIPS 2022 paper [*Efficient Training of Low-Curvature Neural Networks*](https://openreview.net/forum?id=2B2xIJ299rx) by Suraj Srinivas, Kyle Matoba, Himabindu Lakkaraju, & François Fleuret


## Configuration

Set paths to where the code lives, data is located, where you want logs to be 
written, etc. in `path_config.py` (we've done it by setting the environment
variables `CURVATURE_HOME` and `DATA_HOME`, because that's convenient for our
setup, but don't feel obliged to follow it).


## Curvature-regularizing layers 
In case you are just interested in our architectural novelties, you can find them defined here 
- [$\beta$-centered softplus](https://github.com/kylematoba/lcnn/blob/main/models/psoftplus.py)
- [$\gamma$-Lipschitz batchnorm](https://github.com/kylematoba/lcnn/blob/main/models/layers.py#L96)


## Training LCNNs 
To train an LCNN with gradient norm regularization, just run:

```
python train_curvature.py 
      --model-arch=resnet18smoothconv 
      --regularizer=curvature_and_gnorm 
      --dataset=cifar100
```

Or to just train a vanilla LCNN, run: 

```
python train_curvature.py 
      --model-arch=resnet18smoothconv 
      --regularizer=curvature_proxy 
      --dataset=cifar100
```

Other options, e.g. other datasets, architectures, etc. are further discussed at `python3 train_curvature.py --help`. Also look at `experiments.sh` for examples of how to train other baseline models.


**Note about Cleverhans:** Note that currently (October 2022)
PGD is broken for L2 attacks due to this inplace operation:
https://github.com/cleverhans-lab/cleverhans/blob/af09028a9b1307a432bca01f49e9c3568251aa8c/cleverhans/torch/utils.py#L38.
In order to run the experiment with PGD, we monkey-patched this multiply to not be in-place.


## Research
If you found our work helpful for your research, please do consider citing us:

```
@inproceedings{
srinivas2022efficient,
title={Efficient Training of Low-Curvature Neural Networks},
author={Suraj Srinivas and Kyle Matoba and Himabindu Lakkaraju and Fran{\c{c}}ois Fleuret},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=2B2xIJ299rx}
}

```

