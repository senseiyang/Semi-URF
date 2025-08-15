#!/bin/bash
dataset='acdc'  # acdc, promise, isic
data_mode='L'   # L, RGB
data_root='data/ACDC'   # acdc, promise, isic
batch_size=16
epochs=1
lr=0.001
crop_size=256
nclass=4
image_size=256
labeled_id_path='splits/acdc/5/labeled.txt'   #5, 10, 20
unlabeled_id_path='splits/acdc/5/unlabeled.txt'   #5, 10, 20
save_path='semi-URF'
optimizer='optimizer_Adam'    # optimizer_SGD, optimizer_Adam

# ==== Start Training ====
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_mode ${data_mode} \
  --dataset ${dataset} \
  --data_root ${data_root} \
  --batch_size ${batch_size} \
  --epochs ${epochs} \
  --lr ${lr} \
  --crop_size ${crop_size} \
  --nclass ${nclass} \
  --image_size ${image_size} \
  --labeled_data_list ${labeled_id_path} \
  --unlabeled_data_list ${unlabeled_id_path} \
  --save-path ${save_path} \
  --optimizer ${optimizer} \
