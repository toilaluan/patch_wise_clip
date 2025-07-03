from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPImageProcessor
import random
import torch
import math
import json

IMAGENET_CAPTIONS = json.load(open("imagenet_map.txt"))


class ClipDataset(Dataset):
    def __init__(self, pretrained_clip_id: str, is_train=True):
        self.ds = load_dataset(
            "pixparse/cc3m-wds", split="train", num_proc=16
        )
        print(len(self.ds), is_train)
        self.processor = CLIPProcessor.from_pretrained(pretrained_clip_id)
        self.target_size = self.processor.image_processor.size["shortest_edge"]
        print(f"Target size: {self.target_size}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        try:
            x = self.ds[index]
        except Exception as e:
            print(f"Error at index {index}: {e}")
            return self.__getitem__(random.randint(0, len(self.ds)))

        image = x["jpg"]
        width, height = image.size
        ratio = width / height
        scale = math.sqrt(width * height / (self.target_size**2))
        out = self.processor(
            images=x["jpg"],
            text=x["txt"],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        for k, v in out.items():
            out[k] = v.squeeze(0)
        out["meta_tensor"] = torch.tensor([ratio, scale])
        return out




class ImageNetDataset(Dataset):
    """Returns only image tensor + label (no per-example caption duplication)."""

    def __init__(self, processor: CLIPImageProcessor):
        self.ds = load_dataset("timm/imagenet-1k-wds", split="validation", num_proc=16)
        self.proc = processor
        self.target = self.proc.image_processor.size["shortest_edge"]

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        try:
            item = self.ds[idx]
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.ds)))

        img = item["jpg"]
        w, h = img.size
        ratio, scale = w / h, math.sqrt(w * h / (self.target**2))

        pix = self.proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return {
            "pixel_values": pix,
            "label": int(item["cls"]),
            "meta_tensor": torch.tensor([ratio, scale]),
        }


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
