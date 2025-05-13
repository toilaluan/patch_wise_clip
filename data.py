from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import CLIPProcessor
import random


class ClipDataset(Dataset):
    def __init__(self, pretrained_clip_id: str, is_train=True):
        self.ds = load_dataset("nlphuji/flickr30k", split="test")
        if is_train:
            self.ds = self.ds.select(range(15000))
        else:
            self.ds = self.ds.select(range(20000, 20100))
        self.ds = iter(self.ds.shuffle())
        self.processor = CLIPProcessor.from_pretrained(pretrained_clip_id)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        item = next(self.ds)
        image = item["image"]
        text = random.choice(item["caption"])
        inputs = self.processor(
            images=[image],
            text=[text],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        return inputs


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
