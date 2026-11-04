from datasets import load_dataset
from torch.utils.data import Dataset


class POPE(Dataset):

    def __init__(self, key="adversarial"):
        super().__init__()
        self.ds = load_dataset("lmms-lab/POPE", name="Full")
        self.key = key

    def __len__(self):
        return len(self.ds[self.key])

    def __getitem__(self, idx):
        row = self.ds[self.key][idx]
        res = {
            "idx": idx,
            "data": {
                "image": row["image"],
                "text": {
                    "question": row["question"] + " Answer only with 'yes' or 'no'.",
                    "answer": row["answer"],
                },
            },
        }
        return res
