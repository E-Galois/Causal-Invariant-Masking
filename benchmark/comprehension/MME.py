from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image


class MME(Dataset):

    def __init__(self):
        super().__init__()
        self.ds = load_dataset("lmms-lab/MME")

    def __len__(self):
        return len(self.ds["test"])

    def __getitem__(self, idx):
        row = self.ds["test"][idx]
        image = row["image"]
        max_dim = 1024
        w, h = image.size
        if max(w, h) > max_dim:
            scale_factor = max_dim / max(w, h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        res = {
            "idx": idx,
            "data": {
                "image": image,
                "text": {"question": row["question"], "answer": row["answer"]},
            },
        }
        return res
