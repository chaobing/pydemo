#!/bin/bash
python3 main.py \
  --type tensorflow\
  --layout NHWC \
  --model ./config/resnet50.pb\
  --out ./config/resnet50.info \
