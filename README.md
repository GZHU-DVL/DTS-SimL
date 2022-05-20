# ADT-SSL

**ADT-SSL: Adaptive Dual-Threshold for Semi-Supervised Learning**

*<a href="mailto:liangzechen@e.gzhu.edu.cn">Zechen Liang</a>, Yuan-Gen Wang<sup>\*</sup>, Wei Lu, Xiaochun Cao*.

## Installation

Clone this repo.

```
git clone https://github.com/GZHU-DVL/ADT-SSL.git
```

Prerequisites

- Python=3.8
- pillow
- matplotlib
- pandas
- kornia==0.5.0
- scipy
- scikit-learn
- Pytorch>=1.6.0,<=1.9.0
- Torchvision>=0.7.0,<=0.10.0
- Cudatookit=10.2

## Running

### Example

To replicate CIFAR-10 results

```
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="0" \
python main.py \
@runs/cifar10_args.txt
```

To replicate CIFAR-100 result (with distributed training)

```
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="0,1" \
python -m torch.distributed.launch \
--nproc_per_node=2 main_ddp.py \
@runs/cifar100_args.txt \
--num-epochs 5000 \
--num-step-per-epoch 512
```
