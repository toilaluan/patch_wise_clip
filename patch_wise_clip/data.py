from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import CLIPProcessor
import random
import torch
import math
import json


class ClipDataset(Dataset):
    def __init__(self, pretrained_clip_id: str, transform=None):
        self.ds = load_dataset("pixparse/cc3m-wds", split="train", num_proc=32)
        self.processor = CLIPProcessor.from_pretrained(pretrained_clip_id)
        self.target_size = self.processor.image_processor.size["shortest_edge"]
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        try:
            x = self.ds[index]
        except Exception as e:
            print(f"Error at index {index}: {e}")
            return self.__getitem__(random.randint(0, len(self.ds)))

        image = x["jpg"].convert("RGB")
        width, height = image.size
        ratio = width / height
        scale = math.sqrt(width * height / (self.target_size**2))
        out = self.processor(
            text=x["txt"],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        pixel_values = self.transform(image)
        input_ids = out["input_ids"].squeeze(0)
        attention_mask = out["attention_mask"].squeeze(0)
        meta_tensor = torch.tensor([ratio, scale])
        return pixel_values, input_ids, attention_mask, meta_tensor


class ImageNetDataset(Dataset):
    """Returns only image tensor + label (no per-example caption duplication)."""

    def __init__(self, pretrained_clip_id: str, transform=None):
        self.ds = load_dataset("timm/imagenet-1k-wds", split="validation", num_proc=32)
        self.target = 224
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        try:
            item = self.ds[idx]
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.ds)))

        img = item["jpg"].convert("RGB")
        w, h = img.size
        img = self.transform(img)
        ratio, scale = w / h, math.sqrt(w * h / (self.target**2))

        return img, item["cls"], torch.tensor([ratio, scale])


if __name__ == "__main__":
    ds = ClipDataset(pretrained_clip_id="openai/clip-vit-base-patch32")
    item = ds[0]
    print(item)
    print(item["pixel_values"].shape, item["input_ids"].shape)

    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=2)

    item = next(iter(dl))

    print(item)
    print(item["pixel_values"].shape, item["input_ids"].shape)
