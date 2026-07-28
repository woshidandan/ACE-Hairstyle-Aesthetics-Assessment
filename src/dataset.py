import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import random


class HairFaceDataset(Dataset):
    def __init__(self, data_info, transform=None, augment_multiplier=5, random_seed=None):

        self.data_info = data_info
        self.transform = transform
        self.augment_multiplier = augment_multiplier

        if random_seed is not None:
            random.seed(random_seed)
        self.data_indices = list(range(len(data_info))) * augment_multiplier

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        actual_idx = self.data_indices[idx]
        info = self.data_info[actual_idx]
        front_image_path = info['images'][0]  # 0_0.png
        back_image_path = info['images'][6]  # 0_6.png
        left_candidates = info['images'][1:6]  # 0_1.png - 0_5.png
        right_candidates = info['images'][7:12]  # 0_7.png - 0_11.png
        left_image_path = random.choice(left_candidates)
        right_image_path = random.choice(right_candidates)
        selected_image_paths = [front_image_path, back_image_path, left_image_path, right_image_path]
        images = []
        angle_labels = []
        for i, image_path in enumerate(selected_image_paths):
            angle_label = int(image_path.split('_')[-1].split('.')[0])
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
            angle_labels.append(angle_label)

        images = torch.stack(images)  # [4, C, H, W]
        angle_labels = torch.tensor(angle_labels, dtype=torch.long)  # [4]
        score = torch.tensor(info['score'], dtype=torch.float32)
        return images, score, angle_labels


def load_data(root_dir, json_file, test_size=0.2, random_state=None):

    with open(json_file, 'r') as f:
        data = json.load(f)

    data_info = []
    for index in data:
        score = float(data[index]['score'])
        images = [os.path.join(root_dir, f"{index}_{i}.png") for i in range(12)]
        data_info.append({
            'images': images,
            'score': score
        })

    train_info, val_info = train_test_split(
        data_info,
        test_size=test_size,
        random_state=random_state,
    )
    return train_info, val_info
