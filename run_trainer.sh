#!/usr/bin/env bash

python trainer.py \
    --num-epochs 50 \
    --model-name 'fashion_mnist_model_50e.h5' \
    --plot-path './assets/fashion_mnist_50e_loss_acc.png'
