# Pytorch Train Template Demo:o:

This project is a template for training use Pytorch.

### News
:heavy_check_mark: mixed precision training
:heavy_check_mark: distributed training
:heavy_check_mark: wandb monitor

### Train

``` shell script
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch \
	--nproc_per_node=4 \
	--nnodes=1 \
	--node_rank=0 \
	--master_addr=localhost \
	--master_port=22222 \
	train.py --train_images_path data/train_data_custom/ --input_size 720 --batch_size 64
```

### Valid

Set environment variables. Run:

``` shell script
CUDA_VISIBLE_DEVICES=3 python valid.py
```

