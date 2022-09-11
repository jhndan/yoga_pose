#!/bin/sh
python train.py --epochs 300 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule-yoga.yaml --model ai85yoganet --dataset yoga_pose --confusion --param-hist --embedding --device MAX78000 "$@"
