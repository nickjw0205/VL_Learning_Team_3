import torch
#from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
#the original concat_all_gather is abandoned because of no gradient backward
# from utils import is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

import sys
sys.path.append("..")

from sharegpt4v import RandomDataset, create_random_dataloader
from model import longclip
from model import longclip_for_soft_prompt


from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import argparse
import os
import random

# 포트 번호를 랜덤하게 선택하는 범위 설정
port_range_start = 49152  # 동적 포트 범위 시작
port_range_end = 65535    # 동적 포트 범위 끝

# 랜덤 포트 번호 생성
random_port = random.randint(port_range_start, port_range_end)

# 환경 변수 설정
os.environ["MASTER_PORT"] = str(random_port)

# 설정된 포트 번호 출력
print("MASTER_PORT is set to:", os.environ["MASTER_PORT"])
os.environ["RANK"] = "0"  # 현재 프로세스의 순위
os.environ["WORLD_SIZE"] = "1"  # 전체 프로세스의 수
# os.environ["MASTER_PORT"]="12349"
os.environ["MASTER_ADDR"]="127.0.0.1"

import subprocess
# import collections
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# import numpy as np
from datetime import datetime
from torch.cuda.amp import GradScaler
# import warnings
# warnings.filterwarnings("ignore")



class CLIP_Clean_Train():
    def __init__(self, rank,local_rank,args):
        self.rank=rank
        self.local_rank = local_rank
        self.base_model = args.base_model
        # if args.soft_prompt:
        #     print("Using soft prompt")
        #     self.model, _ = longclip_for_soft_prompt.load_from_clip(self.base_model, device='cpu', download_root=args.download_root)
        if args.soft_prompt:
            # Assuming an updated version of load_from_clip that initializes class-specific prompts
            self.model, _ = longclip_for_soft_prompt.load_from_clip(self.base_model, num_classes=100, device='cpu', download_root=args.download_root)

        else:
            print("Not using soft prompt")
            self.model, _ = longclip.load_from_clip(self.base_model, device='cpu', download_root=args.download_root)
            # print(self.model)
        
        print(self.model.positional_embedding_res.shape)
        self.model.train()
                   
        
        if args.freeze:
            for param in self.model.parameters():
                param.requires_grad = False  # 모든 파라미터 동결
            self.model.soft_prompt_embeddings.requires_grad  = True  # soft prompt만 학습
            # self.model.positional_embedding.requires_grad = True
            self.model.positional_embedding_res.requires_grad = True
        if args.freeze_vision:
                # If visual components should be frozen
            for name, param in self.model.named_parameters():
                if 'visual' in name:
                    param.requires_grad = False  # Freeze visual components
        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                # pass
        
        
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)  
        self.model = self.model.cuda()
        
        self.batch_size = args.batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_length = args.warmup_length
        if args.exp_name == "auto":
            self.logdir = f"longclip/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb"
        else:
            self.logdir = args.exp_name
        self.ckptdir = self.logdir + "/ckpt/"
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], find_unused_parameters=True)
           
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler = GradScaler()

    def train_epoch(self, dataloader, epoch, start_iter=0, args=None):
        running_loss = 0.0
        num_batches_per_epoch = len(dataloader)
        if args.soft_prompt:
            # context_length = 228
            context_length = 248
        else:
            context_length = 248
        progress_bar = tqdm(dataloader, disable=(self.rank != 0))  # tqdm progress bar 선언
        for i, (images, texts, class_id) in enumerate(progress_bar):
            # print(class_id)
            
            # print(f"Images Shape: {images.shape}")
            # print(f"Texts Type: {type(texts)}, Example: {texts[:1]}")
            # print(f"Class IDs Shape: {class_id.shape if isinstance(class_id, torch.Tensor) else type(class_id)}")
            
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue
            # images = images.cuda()
            # images_short = images.clone()
            texts = longclip.tokenize(texts, truncate=True, context_length=context_length).cuda()
            # short_text = longclip.tokenize(short_text, truncate=True).cuda()
            self.scheduler(step)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = self.model(images, texts, class_id, self.rank)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss = loss.item()
            progress_bar.set_postfix(loss=running_loss)  # tqdm에 loss 값 업데이트
            if i % num_batches_per_epoch == 0:
                torch.save(self.model.module.state_dict(), './checkpoints/'+str(self.rank)+args.exp_name+str(i)+'.pt')
                
   
    def train(self, resume=False, warmup_length=200, args=None):

        trainset = RandomDataset()
        batch_sampler = create_random_dataloader(trainset, batch_size=self.batch_size, num_workers=4)

        # DataLoader에서 batch_sampler만 사용
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_sampler=batch_sampler,  # 배치 샘플러 사용
            num_workers=32,
            pin_memory=True
        )

        # Initialize the learning rate scheduler
        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=self.num_epoch * len(train_loader))

        start_epoch = 0
        resume_iter = 0

        # Training loop
        for epoch in range(start_epoch, self.num_epoch):
            self.train_epoch(train_loader, epoch, start_iter=resume_iter, args=args)
            if self.rank == 0:
                name = "longclip.pt"
                torch.save(self.model.module.state_dict(), os.path.join('./checkpoints', f'{self.rank}{args.exp_name}{name}'))

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29522"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(device=f'cuda:{rank % num_gpus}')
    return rank, rank % num_gpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=1e-6, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--base_model", default="ViT-L/14", help="CLIP Base Model")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size per gpu."#112
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--resume",
        default=False,
        action='store_true',
        help="resume training from checkpoint."
    )

    parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")
    
    # custom
    parser.add_argument("--freeze", default=False, type=bool, help="CLIP Base Model download root")
    parser.add_argument("--freeze-vision", default=False, type=bool, help="CLIP Base Model download root")
    parser.add_argument("--soft-prompt", default=False, type=bool, help="CLIP Base Model download root")
       
    
    args = parser.parse_args()
    rank,local_rank = setup_distributed()
    print("DDP Done")
    # args.soft_prompt = False

    trainer = CLIP_Clean_Train(
        rank=rank,
        local_rank=local_rank, 
        args=args
        )
    trainer.train(resume=args.resume, warmup_length=args.warmup_length, args=args)
