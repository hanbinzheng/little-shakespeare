# Little Shakespeare

A decoder-only (mini-gpt) transformer to generate shakespeare-style sentence, with the purpose of practicing transformer architecture

```txt
transformer/
│
├── data/
│   └── shakespeare.txt
│
├── tokenizer/
│   ├── train_bpe.py
│   ├── vocab.json
│   └── merges.txt
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
│
├── config.py
└── README.md
```

This repository is tested on Ubuntu 24.04

```shell
conda create -n little-shakespeare python=3.10
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tokenizers tqdm
```
