import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
import math
import copy

class CrossAttention(nn.Sequential):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        transformer_emb_size = hidden_dim
        transformer_n_layer = 4
        transformer_intermediate_size = hidden_dim
        transformer_num_attention_heads = 4
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.encoder = Encoder_1d(transformer_n_layer,
                                 transformer_emb_size,
                                 transformer_intermediate_size,
                                 transformer_num_attention_heads,
                                 transformer_attention_probs_dropout,
                                 transformer_hidden_dropout_rate)

    def forward(self, emb, ex_e_mask, device1):
        global device
        device = device1

        encoded_layers, attention_scores = self.encoder(emb, ex_e_mask)
        return encoded_layers, attention_scores

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class CrossFusion(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(CrossFusion, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear layers for MHC
        self.query_mhc = nn.Linear(hidden_size, self.all_head_size)
        self.key_mhc = nn.Linear(hidden_size, self.all_head_size)
        self.value_mhc = nn.Linear(hidden_size, self.all_head_size)

        # Linear layers for peptide
        self.query_pep = nn.Linear(hidden_size, self.all_head_size)
        self.key_pep = nn.Linear(hidden_size, self.all_head_size)
        self.value_pep = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # MHC (sequence + graph)
        mhc_hidden = hidden_states[0]
        mhc_mask = attention_mask[0]

        # Peptide (sequence + graph)
        pep_hidden = hidden_states[1]
        pep_mask = attention_mask[1]

        mhc_mask = mhc_mask.unsqueeze(1).unsqueeze(2)
        mhc_mask = ((1.0 - mhc_mask) * -10000.0).to(device)

        pep_mask = pep_mask.unsqueeze(1).unsqueeze(2)
        pep_mask = ((1.0 - pep_mask) * -10000.0).to(device)

        # MHC query, key, value
        mixed_query_layer_mhc = self.query_mhc(mhc_hidden)
        mixed_key_layer_mhc = self.key_mhc(mhc_hidden)
        mixed_value_layer_mhc = self.value_mhc(mhc_hidden)

        query_layer_mhc = self.transpose_for_scores(mixed_query_layer_mhc)
        key_layer_mhc = self.transpose_for_scores(mixed_key_layer_mhc)
        value_layer_mhc = self.transpose_for_scores(mixed_value_layer_mhc)

        # Peptide query, key, value
        mixed_query_layer_pep = self.query_pep(pep_hidden)
        mixed_key_layer_pep = self.key_pep(pep_hidden)
        mixed_value_layer_pep = self.value_pep(pep_hidden)

        query_layer_pep = self.transpose_for_scores(mixed_query_layer_pep)
        key_layer_pep = self.transpose_for_scores(mixed_key_layer_pep)
        value_layer_pep = self.transpose_for_scores(mixed_value_layer_pep)

        # Peptide as query, MHC as key/value
        attention_scores_pep = torch.matmul(query_layer_pep, key_layer_mhc.transpose(-1, -2))
        attention_scores_pep = attention_scores_pep / math.sqrt(self.attention_head_size)
        attention_scores_pep = attention_scores_pep + mhc_mask
        attention_probs_pep = nn.Softmax(dim=-1)(attention_scores_pep)
        attention_probs_pep = self.dropout(attention_probs_pep)

        context_layer_pep = torch.matmul(attention_probs_pep, value_layer_mhc)
        context_layer_pep = context_layer_pep.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_pep = context_layer_pep.size()[:-2] + (self.all_head_size,)
        context_layer_pep = context_layer_pep.view(*new_context_layer_shape_pep)

        # MHC as query, Peptide as key/value
        attention_scores_mhc = torch.matmul(query_layer_mhc, key_layer_pep.transpose(-1, -2))
        attention_scores_mhc = attention_scores_mhc / math.sqrt(self.attention_head_size)
        attention_scores_mhc = attention_scores_mhc + pep_mask
        attention_probs_mhc = nn.Softmax(dim=-1)(attention_scores_mhc)
        attention_probs_mhc = self.dropout(attention_probs_mhc)

        context_layer_mhc = torch.matmul(attention_probs_mhc, value_layer_pep)
        context_layer_mhc = context_layer_mhc.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_mhc = context_layer_mhc.size()[:-2] + (self.all_head_size,)
        context_layer_mhc = context_layer_mhc.view(*new_context_layer_shape_mhc)

        # Output of cross fusion
        context_layer = [context_layer_mhc, context_layer_pep]
        # Attention of cross fusion
        attention_probs = [attention_probs_mhc, attention_probs_pep]

        return context_layer, attention_probs

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense_mhc = nn.Linear(hidden_size, hidden_size)
        self.dense_pep = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states_mhc = self.dense_mhc(hidden_states[0])
        hidden_states_mhc = self.dropout(hidden_states_mhc)
        hidden_states_mhc = self.LayerNorm(hidden_states_mhc + input_tensor[0])

        hidden_states_pep = self.dense_pep(hidden_states[1])
        hidden_states_pep = self.dropout(hidden_states_pep)
        hidden_states_pep = self.LayerNorm(hidden_states_pep + input_tensor[1])
        return [hidden_states_mhc, hidden_states_pep]

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = CrossFusion(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_scores = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_scores

class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense_mhc = nn.Linear(hidden_size, hidden_size)
        self.dense_pep = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states_mhc = self.dense_mhc(hidden_states[0])
        hidden_states_mhc = F.relu(hidden_states_mhc)

        hidden_states_pep = self.dense_pep(hidden_states[1])
        hidden_states_pep = F.relu(hidden_states_pep)

        return [hidden_states_mhc, hidden_states_pep]

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense_mhc = nn.Linear(hidden_size, hidden_size)
        self.dense_pep = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states_mhc = self.dense_mhc(hidden_states[0])
        hidden_states_mhc = self.dropout(hidden_states_mhc)
        hidden_states_mhc = self.LayerNorm(hidden_states_mhc + input_tensor[0])

        hidden_states_pep = self.dense_pep(hidden_states[1])
        hidden_states_pep = self.dense_pep(hidden_states[1])
        hidden_states_pep = self.dropout(hidden_states_pep)
        hidden_states_pep = self.LayerNorm(hidden_states_pep + input_tensor[1])
        return [hidden_states_mhc, hidden_states_pep]

class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_scores = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_scores

class Encoder_1d(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size,
                 num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_1d, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads,
                        attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

        # Modality embedding: 0 for sequence, 1 for graph
        self.mod = nn.Embedding(2, hidden_size)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        # Add modality embeddings
        # For MHC sequence
        seq_mhc_emb = torch.tensor([0]).expand(hidden_states[0].size()[0], hidden_states[0].size()[1]).to(device)
        seq_mhc_emb = self.mod(seq_mhc_emb)
        hidden_states[0] = hidden_states[0] + seq_mhc_emb

        # For MHC graph
        graph_mhc_emb = torch.tensor([1]).expand(hidden_states[1].size()[0], hidden_states[1].size()[1]).to(device)
        graph_mhc_emb = self.mod(graph_mhc_emb)
        hidden_states[1] = hidden_states[1] + graph_mhc_emb

        # For peptide sequence
        seq_pep_emb = torch.tensor([0]).expand(hidden_states[2].size()[0], hidden_states[2].size()[1]).to(device)
        seq_pep_emb = self.mod(seq_pep_emb)
        hidden_states[2] = hidden_states[2] + seq_pep_emb

        # For peptide graph
        graph_pep_emb = torch.tensor([1]).expand(hidden_states[3].size()[0], hidden_states[3].size()[1]).to(device)
        graph_pep_emb = self.mod(graph_pep_emb)
        hidden_states[3] = hidden_states[3] + graph_pep_emb

        # Concatenate sequence and graph embeddings for MHC and peptide
        mhc_hidden = torch.cat((hidden_states[0], hidden_states[1]), dim=1)
        pep_hidden = torch.cat((hidden_states[2], hidden_states[3]), dim=1)

        mhc_mask = torch.cat((attention_mask[0], attention_mask[1]), dim=1)
        pep_mask = torch.cat((attention_mask[2], attention_mask[3]), dim=1)

        hidden_states = [mhc_hidden, pep_hidden]
        attention_mask = [mhc_mask, pep_mask]

        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_scores = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        return all_encoder_layers, attention_scores
