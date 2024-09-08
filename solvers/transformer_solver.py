import torch
import torch.nn as nn
import torch.nn.functional as F
from solvers.interface import TSPSolver
from tsp_problem import TSPBatch
import math


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embedding_dim, num_heads, dim_feedforward, dropout, batch_first=True, norm_first=True
            ),
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(embedding_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

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
        if use_cache and cache is not None:
            k, v = cache
            if k is None:
                k = v = tgt
            else:
                k = torch.cat([k, tgt], dim=1)
                v = torch.cat([v, tgt], dim=1)

            tgt2, _ = self.self_attn(tgt, k, v, attn_mask=tgt_mask)
            cache = (k, v)
        else:
            tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
            cache = None

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2, _ = self.multihead_attn(
            tgt,
            memory,
            memory,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, cache


class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerDecoder, self).__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                embedding_dim, num_heads, dim_feedforward, dropout, batch_first=True, norm_first=True
            ),
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(embedding_dim),
        )
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

        if layer_caches is None:
            layer_caches = [None] * self.num_layers

        new_caches = []
        output = self.decoder(output, memory, tgt_mask=tgt_mask)
        # for i, mod in enumerate(self.decoder.layers[1:]):
        # output, new_cache = mod(
        #     output,
        #     memory,
        #     tgt_mask=tgt_mask,
        #     use_cache=use_cache,
        #     cache=layer_caches[i],
        # )
        # new_caches.append(new_cache)

        return output, new_caches


class TransformerSolver(nn.Module, TSPSolver):
    def __init__(
        self,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
    ):
        super(TransformerSolver, self).__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.encoder = TransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)
        self.encoder_output_proj = nn.Linear(d_model, d_model)
        self.decoder_output_proj = nn.Linear(d_model, d_model)
        self.num_heads = nhead
        self.d_model = d_model
        # Add new layers for additive attention
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.attention_v = nn.Linear(d_model, 1)

        # Add start and end tokens
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, batch: TSPBatch):
        points = batch.points
        solutions = batch.solutions
        padding_mask = batch.padding_mask
        device = points.device

        # Project the points to the embedding space
        embedded_points = self.input_proj(points)

        # Encode the points
        encoded_messages = self.encoder(embedded_points)

        # Generate the square mask
        mask = nn.Transformer.generate_square_subsequent_mask(points.size(1) + 1, device=device)

        # Create an index tensor from the solutions
        solution_indices = solutions[:, :-1].unsqueeze(-1).expand(-1, -1, embedded_points.size(-1))

        # Gather the points based on the solution
        ordered_embedded_points = torch.gather(embedded_points, 1, solution_indices)

        # Add start and end tokens
        batch_size = ordered_embedded_points.size(0)
        start_tokens = self.start_token.expand(batch_size, -1, -1)
        ordered_embedded_points = torch.cat([start_tokens, ordered_embedded_points], dim=1)

        # Decode the encoded messages
        decoded_messages, _ = self.decoder(ordered_embedded_points, encoded_messages, mask)

        # Calculate the attention scores using additive attention
        # Shape: [batch_size, decoder_points, 1, d_model]
        query = self.query_proj(decoded_messages).unsqueeze(2)
        # Shape: [batch_size, 1, encoder_points, d_model]
        key = self.key_proj(encoded_messages).unsqueeze(1)
        # Shape: [batch_size, num_points, num_points]
        attention_scores = self.attention_v(torch.tanh(query + key)).squeeze(-1)
        # Calculate the loss between the attention scores and the solutions
        loss = (
            torch.nn.functional.cross_entropy(attention_scores[:, :-1, :].transpose(-1, -2), solutions[:, :-1], reduction="none")
            * padding_mask
        )
        loss = loss.sum() / padding_mask.sum()
        return loss

    def solve(self, batch: TSPBatch):
        points = batch.points
        device = points.device
        batch_size, num_points, _ = points.shape

        # Project the points to the embedding space
        embedded_points = self.input_proj(points)

        # Encode the points
        encoded_messages = self.encoder(embedded_points)

        # Initialize the solution
        solution = torch.zeros(batch_size, num_points + 1, dtype=torch.long, device=device)
        solution[:, 0] = batch.solutions[:, 0]  # Start from the first point (depot)

        with torch.no_grad():
            # Initialize KV caches for the decoder
            kv_caches = [None] * self.decoder.num_layers

            # Start token
            current_token = self.start_token.expand(batch_size, -1, -1)

            for i in range(num_points):
                # Decode the encoded messages with KV caches
                decoded_messages, kv_caches = self.decoder(
                    current_token, encoded_messages, use_cache=True, layer_caches=kv_caches
                )

                # Calculate the attention scores for the last decoded step using additive attention
                query = self.query_proj(decoded_messages[:, -1, :]).unsqueeze(1)
                key = self.key_proj(encoded_messages)
                attention_scores = self.attention_v(torch.tanh(query.unsqueeze(2) + key.unsqueeze(1))).squeeze(-1)

                # Mask out already visited cities
                attention_scores.scatter_(2, solution[:, :i].unsqueeze(1), float("-inf"))

                # Get the next city
                _, next_city = attention_scores[:, 0, :].max(dim=-1)
                solution[:, i] = next_city

                # Update current token
                current_token = torch.cat(
                    [current_token, embedded_points[torch.arange(batch_size), next_city].unsqueeze(1)], dim=1
                )
            solution[:, -1] = solution[:, 0]
        return solution

    def generate_square_subsequent_mask(self, batch_size, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        # Expand the mask to match the batch size
        mask = mask.unsqueeze(0).expand(batch_size * self.num_heads, -1, -1)
        return mask
