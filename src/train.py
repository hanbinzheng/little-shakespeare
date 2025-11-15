import torch
import os
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import Shakespeare
from model import DecoderOnlyTransformer
import inference
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tokenizers import ByteLevelBPETokenizer


vocab_path = Path("../tokenizer/vocab.json")
merges_path = Path("../tokenizer/merges.txt")
data_path = Path("../data/shakespeare.txt")
model_savepath = Path("../checkpoint/model.pth")
log_dir = os.path.abspath("./logs")

tokenizer = ByteLevelBPETokenizer(
    vocab=str(vocab_path),
    merges=str(merges_path)
)

LOG_FILE = 'log.txt'
LR_T_0_EPOCHS = 2
LR_T_MULT = 2
LR_ETA_MIN = 1e-6
GRADIENT_CLIP_VALUE = 1.0
TEST_STEP = 1000

def write_log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

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

def model_info(model):
    MiB = 1024 * 1024
    size = 0
    total_params = sum(p.numel() for p in model.parameters())
    d_type = next(model.parameters()).dtype
    for param in model.parameters():
        size += param.nelement() * param.element_size()

    model_info_str = str(model)
    size_info = f"The model size is: {size / MiB:.2f} MiB"
    dtype_info = f"Data type is: {d_type}"
    total_params_info = f"Total Parameters: {total_params:,}"

    write_log("=== Model Summary ===")
    write_log(model_info_str)
    write_log(size_info)
    write_log(dtype_info)
    write_log(total_params_info)

    return model_info_str, size_info, dtype_info, total_params_info

def shakespeare_dataloader(data_path=data_path, vocab_path=vocab_path,
                           merges_path=merges_path, seq_len=128,
                           batch_size=128, shuffle=True, num_workers=16, pin_memory=True):
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

def test_inference(model, device='cuda'):
    model.eval()

    prompt_str = "To be or not to be"
    prompt_ids = inference.prompt_to_token(prompt_str, tokenizer, device)

    response_ids = inference.inference(model, prompt_ids)
    response_str = inference.response_to_string(
        response_ids[:, prompt_ids.shape[-1]:], tokenizer, device
    )

    model.train()
    return response_str

def train(data_path=data_path, vocab_path=vocab_path,
          merges_path=merges_path, batch_size=128, shuffle=True,
          num_workers=16, pin_memory=True, max_seq_len=128, vocab_size=5000,
          n_block=12, d_embed=256, d_hidden=512, d_ff=1024, n_head=4, dropout=0.1,
          num_epoch=16, lr=1e-4, device='cuda', model_savepath=model_savepath):

    accelerator = Accelerator(
            log_with="tensorboard",
            project_dir=log_dir
    )

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
    ).to(accelerator.device)
    model_info(model)

    # get optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # get scheduler
    steps_per_epoch = len(dataloader)
    T_0_steps = steps_per_epoch * LR_T_0_EPOCHS
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0_steps,
        T_mult=LR_T_MULT,
        eta_min=LR_ETA_MIN
    )

    # prepare
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # train loop begin here
    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epoch}",
            disable=not accelerator.is_main_process
        )
        for input_ids, target_ids in progress_bar:
             logits = model(input_ids)            # (bs, seq, vocab)
             loss = cross_entropy_loss(target_ids, logits)

             accelerator.backward(loss)
             accelerator.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)

             optimizer.step()
             scheduler.step()
             optimizer.zero_grad()

             current_loss = loss.item()
             progress_bar.set_postfix({
                "Loss": f"{current_loss:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.6e}"
             })

             if accelerator.is_main_process:
                 write_log(f"epoch: {epoch}, step: {global_step}, loss: {current_loss}, lr: {scheduler.get_last_lr()[0]}")

                 if global_step % TEST_STEP == 0:
                     unwrapped_model = accelerator.unwrap_model(model)
                     response_str = test_inference(unwrapped_model, device=accelerator.device)
                     write_log(f"epoch: {epoch}, step: {global_step}, sample_generation: {response_str}")
             global_step += 1

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), f"../checkpoint/epoch_{epoch+1}.pth")
            accelerator.print(f"\nModel saved after Epoch {epoch + 1}")

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), model_savepath)
        accelerator.print("\nFinal Model saved.")

if __name__ == '__main__':
    train(batch_size=216, num_epoch=16)
