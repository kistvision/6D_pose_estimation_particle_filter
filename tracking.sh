#!/bin/bash
python demo_tracking.py --dataset_pf ycb\
  --max_iteration 50 \
  --visualization True\
  --num_particles 100\
  --score_threshold=0.2\
  --trained_model=yolact/weights/yolact_resnet50_204_90000.pth --config=yolact_ycb_config --dataset=ycb_dataset --cross_class_nms=True

