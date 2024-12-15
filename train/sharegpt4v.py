import json
import cv2
from PIL import Image
import clip
from collections import defaultdict
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Sampler, Dataset
from itertools import chain, cycle, islice
import os
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import RandomSampler, BatchSampler
data4v_root = '/data/jwsuh/construction/Zero-shot_Image_Classification/dataset/construction/PBL/LongCLIP_for_train/data'
# json_name = '41_to_44.json'
json_name = 'image_data.json'
image_root = '/data/jwsuh/construction/Zero-shot_Image_Classification/dataset/construction/PBL/LongCLIP_for_train/data/'

class share4v_val_dataset(data.Dataset):
    def __init__(self):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[:self.total_len]
        _ , self.preprocess = clip.load("ViT-L/14")
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption


class share4v_train_dataset(data.Dataset):
    def __init__(self):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)
        _ , self.preprocess = clip.load("ViT-L/14")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        if caption is None:
            print(self.image_root + self.json_data[index]['image'])
        caption = caption.replace("\n", " ")
        
        # caption_short = caption.split(". ")[0]
        
        image_name = self.image_root + self.json_data[index]['image']
        # print(image_name)
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption, ""
        # return image_tensor, caption, caption_short
        
        
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # 전역 변수 참조
        global data4v_root, json_name, image_root

        self.image_root = image_root
        # JSON 데이터 로드
        with open(os.path.join(data4v_root, json_name), 'r', encoding='utf8') as file:
            self.data = json.load(file)
        
        # CLIP 모델 로드 및 이미지 전처리 함수 설정
        _, self.preprocess = clip.load("ViT-L/14")
        
        # 클래스 정보 매핑 생성
        self.class_to_indices = defaultdict(list)
        self.class_id_map = {}
        for idx, item in enumerate(self.data):
            # 이미지 경로에서 클래스 키 추출
            class_key = item['image'].split('/')[0]  # 클래스 키 추출 (디렉토리 이름 기준)
            if class_key not in self.class_id_map:
                self.class_id_map[class_key] = len(self.class_id_map)  # 클래스 ID 부여
            self.class_to_indices[class_key].append(idx)
        
        # 클래스 ID 매핑 저장
        output_path = os.path.join(data4v_root, "class_id_map.json")
        with open(output_path, 'w') as f:
            json.dump(self.class_id_map, f, indent=4, ensure_ascii=False)
        print(f"Class ID map saved to {output_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print(index)
        # index가 텐서 또는 리스트로 전달될 경우 첫 번째 값을 사용
        if isinstance(index, torch.Tensor):
            if index.numel() == 1:  # 텐서 크기가 1인 경우 스칼라로 변환
                index = index.item()
            else:
                raise ValueError(f"Index Tensor has unexpected shape: {index.shape}")
        elif isinstance(index, list):
            index = index[0]  # 리스트의 첫 번째 요소 사용

        # 데이터 로드
        item = self.data[index]
        image_path = os.path.join(self.image_root, item['image'])
        image = Image.open(image_path).convert("RGB")  # RGB 변환
        image_tensor = self.preprocess(image)
        caption = item['conversations'][1]['value'].replace("\n", " ")
        class_id = self.class_id_map[item['image'].split('/')[0]]
        return image_tensor, caption, class_id

def create_random_dataloader(dataset, batch_size, num_workers=0):
    sampler = RandomSampler(dataset)  # 랜덤 샘플링
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    return batch_sampler