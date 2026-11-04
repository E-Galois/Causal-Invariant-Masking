from datasets import load_dataset
from torch.utils.data import Dataset


class VQAv2(Dataset):

    def __init__(self, prompt=None):
        super().__init__()
        self.ds = load_dataset("lmms-lab/VQAv2")
        self.prompt = (
            "Answer this question in only a few words: "
            if prompt == "vs"
            else "Answer this question shortly. "
        )

    def __len__(self):
        return len(self.ds["validation"])

    def __getitem__(self, idx):
        row = self.ds["validation"][idx]
        res = {
            "idx": idx,
            "data": {
                "image": row["image"],
                "text": {
                    "question": self.prompt + row["question"],
                    "answer": row["multiple_choice_answer"],
                },
            },
        }
        return res
