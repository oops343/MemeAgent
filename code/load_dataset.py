import os
import json

from torch.utils.data import Dataset

## 定义数据集
class goat_bench_datasets(Dataset):
    def jsonl_load(self, jsonl_path):
        with open(jsonl_path, 'r') as f:
            jsonl_data = [json.loads(line) for line in f]
        return jsonl_data

    def __init__(self, root_path):
        self.root_path = root_path
        self.anno_path = os.path.join(self.root_path, 'test.jsonl')
        self.anno_datas = self.jsonl_load(self.anno_path)
        self.img_dir = os.path.join(self.root_path, 'images')

    def __len__(self):
        return len(self.anno_datas)
    
    def __getitem__(self, idx):
        anno = self.anno_datas[idx]
        return {
            'id': anno['id'],
            'img_path': os.path.join(self.img_dir, anno['img']),
            'label': anno['label'],
            'text': anno['text']
        } 
