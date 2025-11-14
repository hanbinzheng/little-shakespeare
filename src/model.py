import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_inout, d_hidden, n_head, dropout):
        assert d_hidden % n_head == 0
        super().__init__()
        self.d_inout = d_inout
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.W_q = nn.Linear(d_inout, d_hidden)
        self.W_k = nn.Linear(d_inout, d_hidden)
        self.W_v = nn.Linear(d_inout, d_hidden)
        self.W_o = nn.Linear(d_hidden, d_inout)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, Q, K, V, valid):
        """
        implementation of Scaled-Dot-Product-Attention with mask

        Args:
            Q (torch.Tensor): querry, (bs, n_q, d_q)
            K (torch.Tensor): key, (bs, n_k, d_k=d_q)
            V (torch.Tensor): value, (bs, n_v=n_k, d_v)
            valid (torch.Tensor): (bs, n_q) for each query,
                                 describe the number of valid key for the query

        Returns:
            results (torch.Tensor): (bs, n_q, d_v)
        """

        # add the mask for softmax
        def sequence_mask(score, valid, value=-float("inf")):
            # score: (bs * n_q, n_k), valid: (bs * n_q, )
            shape = score.shape
            device = score.device
            seq = torch.arange(shape[1], device=device)  # (n_k, )
            mask = seq.unsqueeze(0) < valid.unsqueeze(-1)  # (bs * n_q, n_k)
            masked_score = score.masked_fill(~mask, value)
            return masked_score

        score = torch.bmm(
            Q, K.transpose(1, 2)) / math.sqrt(K.shape[-1])  # (bs, n_q, n_k)
        shape = score.shape

        # masked score
        masked_score = sequence_mask(
            score=score.reshape(-1, shape[-1]),
            valid=valid.reshape(-1),
        ).reshape(shape)  # (bs, n_q, n_k)

        # softmax weights and results
        weights = nn.functional.softmax(masked_score, dim=-1)  # (bs, n_q, n_k)
        weights = self.dropout(weights)
        results = torch.bmm(weights, V)

        return results

    @staticmethod
    def transfer_forward(matrix, h):
        """
        function to transfer matrix for multihead attention

        Args:
            matrix (torch.Tensor): Q/K/V, (bs, n, d)
            h (int): number of heads

        Returns:
            matrix (torch.Tensor): Q/K/V, (bs * h, n, d / h)
        """
        shape = matrix.shape
        matrix = matrix.reshape(shape[0], shape[1], h, -1)  # (bs, n, h, d / h)
        matrix = matrix.permute(0, 2, 1, 3)  # (bs, h, n, d / h)
        matrix = matrix.reshape(-1, shape[1], shape[2] // h)  # (bs * h, n, d / h)
        return matrix

    @staticmethod
    def transfer_backward(matrix, h):
        """
        function to transfer matrix back

        Args:
            matrix (torch.Tensor): (bs * h, n, d / h)
            h (int): number of heads

        Returns:
            matrix (torch.Tensor): (bs, n, d)
        """
        shape = matrix.shape
        matrix = matrix.reshape(-1, h, shape[1], shape[2])  # (bs, h, n, d / h)
        matrix = matrix.permute(0, 2, 1, 3)  # (bs, n, h, d / h)
        matrix = matrix.reshape(shape[0] // h, shape[1], -1)  # (bs, n, d)
        return matrix

    def multihead_attention(self, Q, K, V, valid):
        """
        implementation of MultiHead Scaled-Dot-Product-Attention with mask

        Args:
            Q (torch.Tensor): querry, (bs, n_q, d_q)
            K (torch.Tensor): key, (bs, n_k, d_k=d_q)
            V (torch.Tensor): value, (bs, n_v=n_k, d_v)
            valid (torch.Tensor): (bs, n_q) for each query,
                                 describe the number of valid key for the query

        Returns:
            results (torch.Tensor): (bs, n_q, d_v)
        """
        # prepare for attention
        h = self.n_head
        Q = self.transfer_forward(Q, h)  # (bs * h, n_q, d_q / h)
        K = self.transfer_forward(K, h)  # (bs * h, n_k, d_k / h)
        V = self.transfer_forward(V, h)  # (bs * h, n_k, d_v / h)
        valid = valid.repeat(h, 1)  # (bs * h, n_q)

        # implement attention mechanism
        results = self.attention(Q, K, V, valid)  # (bs * h, n_q, d_v / h)
        results = self.transfer_backward(results, h)  # (bs, n_q, d_v)

        return results

    def forward(self, x):
        """
        Multihead-Self-Scaled-Dot-Product-Attention

        Args:
            x (torch.Tensor): from previous layer, (bs, n, d)
        Returns:
            x (torch.Tensor): output for next layer, (bs, n, d)
        """
        shape = x.shape
        device = x.device
        Q = self.W_q(x)  # (bs, n, d_hidden)
        K = self.W_k(x)  # (bs, n, d_hidden)
        V = self.W_v(x)  # (bs, n, d_hidden)
        valid = torch.arange(1, shape[1]+1, device=device).repeat(shape[0], 1)  # (bs, n)

        result = self.multihead_attention(Q, K, V, valid)  # (bs, n, d_hidden)
        result = self.W_o(result)  # (bs, n, d_inout)
        return result

class TransformerBlock(nn.Module):
    def __init__(self, d_inout, d_hidden, d_ff, n_head, dropout):
        super().__init__()
        assert d_hidden % n_head == 0
        self.d_inout = d_inout
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.d_ff = d_ff
        self.ffn = nn.Sequential(
            nn.Linear(d_inout, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_inout)
        )
        self.attn = MultiHeadSelfAttention(d_inout, d_hidden, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_inout)
        self.norm2 = nn.LayerNorm(d_inout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (bs, n, d_inout)

        Returns:
            x (torch.Tensor): (bs, n, d_inout)
        """
        # attention and residual norm
        attn_out = self.attn(x)  # (bs, n, d_inout)
        x = x + self.dropout(attn_out)  # (bs, d, d_inout)
        x = self.norm1(x)  # (bs, n, d_inout)

        # feed forward and residual norm
        ffn_out = self.ffn(x)  # (bs, n, d_inout)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)  # (bs, n, d_inout)

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        assert d_embed % 2 == 0
        self.d_embed = d_embed

    def forward(self, seq_len, device='cuda', max_timescale=1e4):
        """
        P_{pos, 2i} = sin(pos * (1 / max_timescale)^{2i / d_embed})
        P_{pos, 2i+1} = cos(pos * (1 / max_timescale)^{2i+1 / d_embed})

        Args:
            seq_len (int): number of positions to embedding
            device (str): the target device
            max_timescale (float): the scale for frequency in embedding

        Returns:
            pos_embed (torch.Tensor): (seq_len, d_embed)
        """
        d = self.d_embed
        pos = torch.arange(0, seq_len, device=device)  # (seq_len)
        div_term = torch.exp(
            - torch.arange(0, d, 2, device=device) * math.log(max_timescale) / d)  # (d // 2, )
        args = div_term.unsqueeze(0) * pos.unsqueeze(-1)  # (seq_len, d)
        pos_embed = torch.zeros(seq_len, d, device=device)
        pos_embed[:, 0::2] = torch.sin(args)
        pos_embed[:, 1::2] = torch.cos(args)
        return pos_embed

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, max_seq_len, vocab_size, n_block, d_embed, d_hidden, d_ff, n_head, dropout):
        assert d_embed % n_head == 0
        super().__init__()

        # store model information
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.n_block = n_block
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.d_ff = d_ff
        self.n_head = n_head

        # model initialization
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.pos_embed = PositionalEmbedding(d_embed)
        blocks = [TransformerBlock(
            d_embed, d_hidden, d_ff, n_head, dropout) for _ in range(n_block)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        decoder only transformer architecture

        Args:
            x (torch.Tensor): sentence token, (bs, seq_len)

        Returns:
            logits (torch.Tensor): logits result, (bs, seq_len, vocab_size)
        """
        seq_len = x.shape[-1]
        assert seq_len <= self.max_seq_len
        device = x.device

        # initial embedding and positional embedding
        x_embed = self.embed(x) * math.sqrt(self.d_embed)  # (bs, seq_len, d_embed)
        pos_embed = self.pos_embed(seq_len, device=device)  # (seq_len, d_embed)
        x_embed = self.dropout(x_embed) + pos_embed.unsqueeze(0)  # (bs, seq_len, d_embed)

        # transformer block
        for block in self.blocks:
            x_embed = block(x_embed)  # (bs, seq_len, d_embed)
        x_out = self.norm(x_embed)  # (bs, seq_len, d_embed)

        # final logits
        logits = x_out @ self.embed.weight.T  # (bs, seq_len, vocab_size)

        return logits
