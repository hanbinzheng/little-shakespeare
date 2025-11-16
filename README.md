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
└── README.md
```

This repository is tested on Ubuntu 24.04


### Environment Setup
```bash
conda create -n little-shakespeare python=3.10
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tokenizers tqdm accelerate tensorboard
```

### Train the Model
```bash
conda activate little-shakespeare
mkdir checkpoint
cd src
python train.py
```

### Play and have fun
```bash
python inference.py
```

### Some result.....

没训好瞎训了一会儿，蠢蠢的还挺可爱的

```text
>> hi
,
Right son of children, you shall know your son-in-law
>> to be
 to be
pinched with modesty
>> lord
ing
>> to be
 I, my lord,
>> father
,
And, I fear, with all my heart’s forces
>> god
s;
For I have the very ballad will be very hot,
And do all that dim and have yielded up
To th’ volume of authority
>>
```
