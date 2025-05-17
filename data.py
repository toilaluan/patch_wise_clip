from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPImageProcessor
import random
import torch


class ClipDataset(Dataset):
    def __init__(self, pretrained_clip_id: str, is_train=True):
        if is_train:
            split = "train[:98%]"
        else:
            split = "train[99%:]"
        self.ds = load_dataset(
            "svjack/conceptual_captions_3m_en_tiny", split=split, num_proc=16
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
        scale = self.target_size / width
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
