import torch
import torch.nn as nn
from pathlib import Path
from model import DecoderOnlyTransformer
from tokenizers import ByteLevelBPETokenizer


model_savepath = Path("../checkpoint/model.pth")
vocab_path = Path("../tokenizer/vocab.json")
merges_path = Path("../tokenizer/merges.txt")

TOP_K = 50
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 100
STOP_TOKEN_IDS = [18, 35, 5, 65] # 18: '.', 35: '?', 5: '!', 65: ']'

def load_model(model, model_savepath=model_savepath, device='cuda'):
    state_dict = torch.load(model_savepath, map_location=device, weights_only=False)

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model.to(device)

    return model

@torch.no_grad()
def inference(model, prompt_ids, max_new_tokens=MAX_NEW_TOKENS,
              top_k=TOP_K, temperature=TEMPERATURE):
    """
    function to implement inference

    Args:
        model (nn.Module): DecoderOnlyTransformer
        prompt_ids (torch.Tensor): encoded token as prompt, (bs, seq_len)
        max_new_tokens (int): max number of tokens to generate
        top_k (int): number of top tokens to sample from
        temperature (float): controls the randomness of the sampling

    Returns:
        response (torch.Tensor): model response, (bs, < max_seq_len, vocab_size)
    """
    model.eval()
    max_seq_len = model.max_seq_len
    generated = prompt_ids.clone()  # (bs, deq_len)

    for _ in range(max_new_tokens):
        model_input = generated[:, -max_seq_len:]
        logits = model(model_input)  # (bs, seq_len, vocab_size)
        next_logits = logits[:, -1, :]  # (bs, vocab_size)
        next_logits = next_logits / temperature  # introduce temperature

        # top k sampling
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -float('inf')

        probs = nn.functional.softmax(next_logits, dim=-1)  # (bs, vocab_size)
        next_token = torch.multinomial(probs, num_samples=1)  # (bs, 1)

        # check for stop condition
        if next_token.item() in STOP_TOKEN_IDS:
            break
        generated = torch.cat([generated, next_token], dim=-1)  # (bs, seq_len + 1)

    return generated

def prompt_to_token(prompt_str, tokenizer, device='cuda'):
    """
    function to change prompt to tokens

    Args:
        prompt_str (string): the input prompt
        tokenizer (ByteLevelTokenizer): tokenizer

    Returns:
        prompt_ids: token for prompt
    """
    tokens = tokenizer.encode(prompt_str).ids
    prompt_ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (seq_len, )
    return prompt_ids.reshape(1, -1)  # (1, seq_len)

def response_to_string(response_ids, tokenizer, device='cuda'):
    """
    function to change the token response to string

    Args:
        response_ids (torch.Tensor): token response, (1, seq_len)
        tokenizer (ByteLevelTokenizer): tokenizer

    Returns:
        response_str (string)
    """
    response = response_ids.squeeze(0).tolist()
    response_str = tokenizer.decode(response)
    return response_str

def terminal_toy(model, tokenizer, device='cuda'):
    print("Welcome to little-shakespeare! Type 'exit()' to quit.")

    while True:
        prompt_str = input(">> ")  # 读取用户输入

        if prompt_str.strip() == "exit()":
            print("Exiting...")
            break

        prompt_ids = prompt_to_token(prompt_str, tokenizer, device)
        response_ids = inference(model, prompt_ids)  # (1, max_seq_len)
        response_str = response_to_string(
            response_ids[:, prompt_ids.shape[-1]:], tokenizer, device
        )

        print(response_str)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DecoderOnlyTransformer(
        max_seq_len=128,
        vocab_size=5000,
        n_block=12,
        d_embed=256,
        d_hidden=512,
        d_ff=1024,
        n_head=4,
        dropout=0.1
    ).to(device)
    model = load_model(model, model_savepath, device)
    tokenizer = ByteLevelBPETokenizer(
        vocab=str(vocab_path),
        merges=str(merges_path)
    )

    terminal_toy(model, tokenizer, device)
