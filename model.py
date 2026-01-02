import math

import torch
from torch import nn
from torch.nn import functional as F

from tokenize import Tokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:len(x), :]


class TransformerAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_matrix = nn.Linear(d_model, d_model)
        self.key_matrix = nn.Linear(d_model, d_model)
        self.value_matrix = nn.Linear(d_model, d_model)
        self.final_projection = nn.Linear(d_model, d_model)

    def forward(self, sequence, attention_mask=None):
        sequence_length, model_dim = sequence.shape
        q = self.query_matrix(sequence)
        k = self.key_matrix(sequence)
        v = self.value_matrix(sequence)

        q = q.view(sequence_length, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(sequence_length, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(sequence_length, self.num_heads, self.head_dim).transpose(0, 1)

        q_k = torch.matmul(q, k.transpose(-1, -2))
        q_k = q_k/math.sqrt(self.head_dim)

        if attention_mask is not None:
            q_k = q_k + attention_mask
        attention_score = F.softmax(q_k, dim=-1)
        q_k_v = torch.matmul(attention_score, v)

        q_k_v = q_k_v.transpose(0, 1).contiguous()
        q_k_v = q_k_v.view(sequence_length, self.num_heads*self.head_dim)

        attention_output = self.final_projection(q_k_v)


        return attention_output


class NeuralNet(nn.Module):
    def __init__(self, d_model, hidden_layer_dim):
        super().__init__()
        self.d_model = d_model
        self.hidden_layer_dim = hidden_layer_dim
        self.fc1 = nn.Linear(in_features=self.d_model, out_features=self.hidden_layer_dim)
        self.fc2 = nn.Linear(in_features=self.hidden_layer_dim, out_features=self.d_model)

    def forward(self, attention_output):
        fc1_output = self.fc1(attention_output)
        fc1_output = F.relu(fc1_output)
        fc2_output = self.fc2(fc1_output)
        return fc2_output
    

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_layer_dim):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_layer_dim = hidden_layer_dim

        self.transformer_attention = TransformerAttention(d_model=d_model, num_heads=num_heads)
        self.neural_net = NeuralNet(d_model=d_model, hidden_layer_dim=hidden_layer_dim)

        self.layer_normalization_attention = nn.LayerNorm(self.d_model)
        self.layer_normalization_neuralnet = nn.LayerNorm(self.d_model)

    @staticmethod
    def generate_attention_mask(sequence_length, device):
        mask = torch.triu(torch.ones(sequence_length, sequence_length, device=device), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def forward(self, sequence):
        sequence_length, _ = sequence.size()
        attention_mask = self.generate_attention_mask(sequence_length, sequence.device)
        attention_mask = attention_mask.unsqueeze(0)
        attention = self.transformer_attention(sequence, attention_mask)
        layer_normed_attention = self.layer_normalization_attention(sequence + attention)
        network_output = self.neural_net(layer_normed_attention)
        layer_normed_network_output = self.layer_normalization_neuralnet(layer_normed_attention + network_output)
        return layer_normed_network_output


class TransformersDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, hidden_layer_dim, num_blocks):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_layer_dim = hidden_layer_dim
        self.num_blocks = num_blocks
        self.positional_encoding_layer = PositionalEncoding(d_model, max_len=1000)
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.decoder_blocks = nn.ModuleList([TransformerDecoderBlock(self.d_model, self.num_heads, self.hidden_layer_dim) for _ in range(self.num_blocks)])

        self.projection_layer = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, sequence):
        sequence_embedding = self.embedding_layer(sequence)
        final_embedding = self.positional_encoding_layer(sequence_embedding)
        decoder_block_output = final_embedding
        for block in self.decoder_blocks:
            decoder_block_output = block(decoder_block_output)
        logits = self.projection_layer(decoder_block_output)
        return logits
