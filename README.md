# Training with Block Minifloat in Pytorch

This repository provides code to accompany the paper [A Block Minifloat Representation for Training Deep Neural Networks](https://openreview.net/forum?id=6zaTwpNSsQ2).


## Requirements
python>=3.6
pytorch>=1.1


## Usage
```bash
python main.py --data_path=. --dataset=IMAGENET --model=ResNet18LP --batch_size=256 --wd=1e-4 --lr_init=0.1 --epochs=90 \
--weight-exp=2 --weight-man=5 \
--activate-exp=2 --activate-man=5 \
--error-exp=4 --error-man=3
```

This will train a ResNet-18 model using the BM8 format specified in the paper.


