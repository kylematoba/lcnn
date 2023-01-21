python train_curvature.py --model-arch=resnet18 --dataset=cifar100  # Standard
python train_curvature.py --regularizer=curvature_proxy --model-arch=resnet18smoothconv --dataset=cifar100  # LCNNs
python train_curvature.py --regularizer=gnorm --model-arch=resnet18 --dataset=cifar100  # GradReg
python train_curvature.py --regularizer=curvature_and_gnorm --model-arch=resnet18smoothconv --dataset=cifar100  # LCNNs + GradReg
python train_curvature.py --regularizer=cure --model-arch=resnet18 --dataset=cifar100  # CURE
python train_curvature.py --model-arch=resnet18lowbeta --dataset=cifar100  # Softplus + Wt. Decay
python train_curvature.py --model-arch=resnet18 --dataset=cifar100  --adv_attack l2_pgd_3_100000  # Adversarial Training