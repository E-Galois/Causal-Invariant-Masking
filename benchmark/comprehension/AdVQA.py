from torch.utils.data import Dataset
from collections import Counter
from .AdVQABuilder import AdVQAGeneratorBasedBuilder


class AdVQA(Dataset):

    def __init__(self, prompt=None):
        super().__init__()
        self.prompt = (
            "Answer this question in only a few words: "
            if prompt == "vs"
            else "Answer this question shortly. "
        )
        builder = AdVQAGeneratorBasedBuilder()
        builder.download_and_prepare()
        self.ds = builder.as_dataset()

    def __len__(self):
        return len(self.ds["validation"])

    def __getitem__(self, idx):
        row = self.ds["validation"][idx]
        all_answers = [item["answer"] for item in row["answers"]]
        answer_counts = Counter(all_answers)
        most_common_answer, count = answer_counts.most_common(1)[0]
        res = {
            "idx": idx,
            "data": {
                "image": row["image"],
                "text": {
                    "question": self.prompt + row["question"],
                    "answer": most_common_answer,
                },
            },
        }
        return res
