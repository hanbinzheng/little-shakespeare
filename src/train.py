import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import Shakespeare
from model import DecoderOnlyTransformer


vocab_path = Path("../tokenizer/vocab.json")
merges_path = Path("../tokenizer/merges.txt")
data_path = Path("../data/shakespeare.txt")
model_savepath = Path("../checkpoint/model.pth")

def cross_entropy_loss(target, logits):
    """
    implement the corss entropy loss

    Args:
        target (torch.Tensor): (bs, seq_len)
        logits (torch.Tensor): model output (bs, seq_len, vocab_size)

    Returns:
        loss (torch.Tensor): (, )
    """
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction='mean'
    )
    return loss

def shakespeare_dataloader(data_path=data_path, vocab_path=vocab_path,
                           merges_path=merges_path, seq_len=128,
                           batch_size=128, shuffle=True, num_workers=4, pin_memory=True):
    dataset = Shakespeare(
        data_path=data_path,
        vocab_path=vocab_path,
        merges_path=merges_path,
        seq_len=seq_len
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader

def train(data_path=data_path, vocab_path=vocab_path,
          merges_path=merges_path, batch_size=128, shuffle=True,
          num_workers=4, pin_memory=True, max_seq_len=128, vocab_size=5000,
          n_block=8, d_embed=256, d_hidden=512, d_ff=512, n_head=4, dropout=0.1,
          num_epoch=16, lr=1e-4, device='cuda', model_savepath=model_savepath):
    # get shakespeare dataloader
    dataloader = shakespeare_dataloader(
        data_path=data_path,
        vocab_path=vocab_path,
        merges_path=merges_path,
        seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # get model
    model = DecoderOnlyTransformer(
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        n_block=n_block,
        d_embed=d_embed,
        d_hidden=d_hidden,
        d_ff=d_ff,
        n_head=n_head,
        dropout=dropout
    ).to(device)

    # get optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
         for input_ids, target_ids in dataloader:
             input_ids = input_ids.to(device)
             target_ids = target_ids.to(device)

             logits = model(input_ids)            # (bs, seq, vocab)
             loss = cross_entropy_loss(target_ids, logits)
             print(f"epoch {epoch + 1}, loss: {loss.item()}")

             optimizer.zero_grad()
             loss.backward()
             optimizer.step()

    torch.save(model.state_dict(), model_savepath)


if __name__ == '__main__':
    train(batch_size=256)
