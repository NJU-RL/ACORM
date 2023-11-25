import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, att_dim, att_out_dim, soft_temperature, dim_q, dim_k, dim_v):
        super(MultiHeadAttention, self).__init__()
        assert (att_dim % n_heads) == 0, "n_heads must divide att_dim"
        self.att_dim = att_dim
        self.att_out_dim = att_out_dim
        self.head_att_dim = att_dim // n_heads
        self.n_heads = n_heads
        self.temperature = self.head_att_dim ** 0.5 / soft_temperature

        self.fc_q = nn.Linear(dim_q, self.att_dim, bias=False)
        self.fc_k = nn.Linear(dim_k, self.att_dim, bias=False)
        self.fc_v = nn.Linear(dim_v, self.att_dim)
        self.fc_final = nn.Linear(self.att_dim, self.att_out_dim)

    def forward(self, q, k, v):
        # q.shape = (batch, N, dim)
        batch_size = q.shape[0]
        # shape = (batch*N, att_dim)->(batch, N, heads, head_att_dim)->(batch, heads, N, head_att_dim)
        q = self.fc_q(q.view(-1, q.shape[2])).view(batch_size, -1, self.n_heads, self.head_att_dim).transpose(1, 2)
        # shape = (batch*N, att_dim)->(batch, N, heads, head_att_dim)->(batch, heads, head_att_dim, N)
        k_T = self.fc_k(k.view(-1, k.shape[2])).view(batch_size, -1, self.n_heads, self.head_att_dim).permute(0,2,3,1)
        v = self.fc_v(v.view(-1, v.shape[2])).view(batch_size, -1, self.n_heads, self.head_att_dim).transpose(1, 2)
        alpha = F.softmax(torch.matmul(q/self.temperature, k_T), dim=-1)  # shape = (batch, heads, N, N)
        # shape = (batch, heads, N, head_att_dim)->(batch, N, heads, head_att_dim)->(batch, N, att_dim)
        result = torch.matmul(alpha, v).transpose(1, 2).reshape(batch_size, -1, self.att_dim)
        result = self.fc_final(result)  # shape = (batch, N, att_out_dim)
        return result
