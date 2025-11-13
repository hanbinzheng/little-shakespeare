import torch
from pathlib import Path
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer

vocab_path = Path("../tokenizer/vocab.json")
mergrs_path = Path("../tokenizer/merges.txt")
data_path = Path("../data/shakespeare.txt")

class Shakspeare(Dataset):
    def __init__(self, data_path=data_path,
                 vocab_path=vocab_path, merges_path=merges_path, seq_len=128):
        super().__init__()

        self.seq_len = seq_len
        self.tokenizer = ByteLevelBPETokenizer(
            vocab=str(vocab_path),
            merges=str(merges_path)
        )
        text = data_path.read_text(encoding="utf-8")

        self.encoded = self.tokenizer.encode(text)
        self.num_samples = len(self.encoded.ids) - seq_len

    def __getitem__(self, idx):
        train_input = torch.tensor(
            self.encoded.ids[idx : idx+self.seq_len], dtype=torch.long)
        train_target = torch.tensor(
            self.encoded.ids[idx+1 : idx+1+self.seq_len], dtype=torch.long)
        return train_input, train_target

    def __len__(self):
        return self.num_samples
