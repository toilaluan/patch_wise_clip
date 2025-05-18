from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPImageProcessor
import random
import torch
import math


class ClipDataset(Dataset):
    def __init__(self, pretrained_clip_id: str, is_train=True):
        self.ds = load_dataset(
            "svjack/conceptual_captions_3m_en_tiny", split="train", num_proc=16
        )
        print(len(self.ds), is_train)
        self.ds = self.ds.filter(
            lambda x: x["image"].size[0] >= 112 and x["image"].size[1] >= 112,
            num_proc=16,
        )
        self.processor = CLIPProcessor.from_pretrained(pretrained_clip_id)
        self.target_size = self.processor.image_processor.size["shortest_edge"]
        print(f"Target size: {self.target_size}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        x = self.ds[index]
        image = x["image"]
        width, height = image.size
        ratio = width / height
        scale = math.sqrt(width * height / (self.target_size**2))
        out = self.processor(
            images=x["image"],
            text=x["caption"],
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
    def __init__(self, processor: CLIPImageProcessor):
        self.ds = load_dataset("frgfm/imagenette", "320px", split="validation")
        self.processor = processor
        self.target_size = self.processor.image_processor.size["shortest_edge"]
        self.texts, self.labels = self.get_texts_labels()

    def __len__(self):
        return len(self.ds)

    def get_texts_labels(self):
        labels = []
        texts = []
        texts = self.ds.features["label"].names
        labels = list(range(len(texts)))
        return texts, labels

    def __getitem__(self, index):
        x = self.ds[index]
        image = x["image"]
        label = int(x["label"])

        width, height = image.size
        ratio = width / height
        scale = math.sqrt(width * height / (self.target_size**2))

        captions = [f"An image of a {text}" for text in self.texts]
        out = self.processor(
            images=image,
            text=captions,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        # out: {'pixel_values': (num_classes, 3, 224, 224), 'input_ids': (num_classes, 77), ...}
        # For images, pixel_values is repeated per class. We'll use only the first copy.
        for k, v in out.items():
            out[k] = v.squeeze(0) if v.dim() == 3 else v  # (num_classes, ...) or (num_classes, 77)
        return {
            "pixel_values": out["pixel_values"][0],  # use only first (since image repeats)
            "input_ids_all": out["input_ids"],       # (num_classes, 77)
            "attention_mask_all": out["attention_mask"],  # (num_classes, 77)
            "meta_tensor": torch.tensor([ratio, scale]),
            "label": label
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
