import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import wandb

from utils import dataloader
from model import model
from utils import loss

wandb.init(project="720-train-zero-dce")



def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)



def train(config):
	# 0. set up distributed device
	rank = int(os.environ["RANK"])
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(rank % torch.cuda.device_count())
	torch.distributed.init_process_group(backend='nccl', init_method='env://') # cpu : gloo; gpu : nccl 
	device = torch.device("cuda", local_rank)

	print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

	# 1. define network
	net = model.enhance_net_nopool().cuda()
	net = net.to(device)
	# DistributedDataParallel
	net = DDP(net, device_ids=[local_rank], output_device=local_rank)
	net.apply(weights_init)

	if config.load_pretrain == True:
		print("Currently use pretrain model.")
		net.load_state_dict(torch.load(config.pretrain_dir))
	
	# 2. define dataloader
	train_dataset = dataloader.lowlight_loader(config.train_images_path, config.input_size)
	# DistributedSampler
	# we test single Machine with 4 GPUs
	# so the [batch size] for each process is 256 / 4 = 64
	sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, sampler=sampler)

	# 3. define loss and optimizer
	wandb.watch(net, log="all")

	L_color = loss.L_color()
	L_spa = loss.L_spa()
	L_exp = loss.L_exp(16,0.6)
	L_TV = loss.L_TV()

	optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	# https://blog.csdn.net/qyhaill/article/details/103043637
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

	if rank == 0:
		print("            =======  Training  ======= \n")
	
	# 4. start to train
	net.train()
	scaler = amp.GradScaler()
	for epoch in range(1, config.epochs + 1):
		train_loss = correct = total = 0
		# set sampler
		train_loader.sampler.set_epoch(epoch)

		for iteration, img_lowlight in enumerate(train_loader):
			img_lowlight = img_lowlight.cuda()
			with amp.autocast():
				enhanced_image_1, enhanced_image, A  = net(img_lowlight)
				# best_loss
				Loss_TV = 200*L_TV(A)
				loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
				loss_col = 5*torch.mean(L_color(enhanced_image))
				loss_exp = 10*torch.mean(L_exp(enhanced_image))
				loss_ =  Loss_TV + loss_spa + loss_col + loss_exp
			
			#
			wandb.log({'epoch': epoch, 'loss': loss_, 'accuracy': 0.99})
			scaler.scale(loss_).backward()
			torch.nn.utils.clip_grad_norm(net.parameters(), config.grad_clip_norm)
			scaler.step(optimizer)
			scaler.step(scheduler)
			scaler.update()
			
			# optimizer.zero_grad()
			# loss_.backward()
			# torch.nn.utils.clip_grad_norm(net.parameters(), config.grad_clip_norm)
			# optimizer.step()

			train_loss += loss_.item()
			# total += targets.size(0)
			# correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()
			if rank == 0 and ((iteration + 1) % config.log_iter) == 0 or (iteration + 1) == len(train_loader):
				print(
					"   == step: [{:3}/{}] [{}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
						iteration + 1,
						len(train_loader),
						epoch,
						config.epochs,
						train_loss / (iteration + 1),
						99.9
						# 100.0 * correct / total,
					)
				)
			if ((iteration + 1) % config.snapshot_iter) == 0:
				torch.save(net.state_dict(), config.checkpoint + "Epoch_" + str(epoch) + '.pth')

	if rank == 0:
		print("\n            =======  Training Finished  ======= \n")	




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Input Parameters
	parser.add_argument('--train_images_path', type=str, default="data/train_data/")
	parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--input_size', type=int, default=512)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--local_rank', type=int, default=0, help="DDP parameter, do not modify")
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--log_iter', type=int, default=10, help="Training log output every log_iter")
	parser.add_argument('--snapshot_iter', type=int, default=10, help="Model save every snapshot_iter")
	parser.add_argument('--checkpoint', type=str, default="checkpoint/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "checkpoint/Epoch99.pth")
	config = parser.parse_args()

	if not os.path.exists(config.checkpoint):
		os.mkdir(config.checkpoint)

	train(config)
