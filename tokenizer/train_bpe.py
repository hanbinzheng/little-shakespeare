from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

data_path = Path("../data/shakespeare.txt")

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[str(data_path)],
    vocab_size=5000,
    min_frequency=2,  # frequency under this will be omitted
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]  # required
)

tokenizer.save_model(str(Path(".")))

print("BPE tokenizer training done")
