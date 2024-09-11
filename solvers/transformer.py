import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embedding_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Self-attention layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Cross-attention layers
        self.cross_q_proj = nn.Linear(d_model, d_model)
        self.cross_k_proj = nn.Linear(d_model, d_model)
        self.cross_v_proj = nn.Linear(d_model, d_model)
        self.cross_o_proj = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        use_cache=False,
        cache=None,
    ):
        # Self-attention
        if use_cache:
            if cache is None:
                k = self.k_proj(tgt)
                v = self.v_proj(tgt)
            else:
                k, v = cache
                # Only process the last token
                tgt = tgt[:, -1:]
                k = torch.cat([k, self.k_proj(tgt)], dim=1)
                v = torch.cat([v, self.v_proj(tgt)], dim=1)

            q = self.q_proj(tgt[:, -1:])
            tgt2 = self._attention(q, k, v, tgt_mask)
            cache = (k, v)
        else:
            q = self.q_proj(tgt)
            k = self.k_proj(tgt)
            v = self.v_proj(tgt)
            tgt2 = self._attention(q, k, v, tgt_mask)
            cache = None

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        q = self.cross_q_proj(tgt)
        k = self.cross_k_proj(memory)
        v = self.cross_v_proj(memory)
        tgt2 = self._attention(q, k, v)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, cache

    def _attention(self, q, k, v, attn_mask=None):
        q = einops.rearrange(q, "b l (h d) -> b h l d", h=self.nhead)
        k = einops.rearrange(k, "b l (h d) -> b h l d", h=self.nhead)
        v = einops.rearrange(v, "b l (h d) -> b h l d", h=self.nhead)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask[None, None] != 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        # Compute output
        output = torch.matmul(attn, v)
        output = einops.rearrange(output, "b h l d -> b l (h d)")

        return self.o_proj(output)


class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(embedding_dim, num_heads, dim_feedforward, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.num_layers = num_decoder_layers

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        use_cache=False,
        layer_caches=None,
    ):
        output = tgt
        new_caches = []

        for i, layer in enumerate(self.layers):
            cache = layer_caches[i] if layer_caches is not None else None
            output, new_cache = layer(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache)
            new_caches.append(new_cache)

        output = self.norm(output)

        return output, new_caches
