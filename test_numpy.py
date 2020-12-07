# import numpy as np

# arr = np.random.rand(2, 4, 4)
# a_del = np.delete(arr, (1, 1), 1)

# print(arr)
# print(arr.shape)

# print(a_del)
# print(a_del.shape)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

vocab_size = 3
emb_size = 1
hidden_size = 1
num_layers = 1
num_direction = 1
batch_size = 2

torch.manual_seed(42)

emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
emb.weight = nn.Parameter(torch.tensor([[float(i)] for i in range(vocab_size)]))
rnn = nn.LSTM(input_size=emb_size,
              hidden_size=hidden_size,
              num_layers=num_layers,
              bidirectional=num_direction == 2)

print('# ----- Input -----')
X = [[1, 1], [2, 2, 2]]
X_len = torch.tensor([len(x) for x in X])
X = [torch.tensor(x) for x in X]
print(f'ORIGINAL\n{X}')
print(f'LENGTH\n{X_len}')
padded_X = pad_sequence(X)
print('type:', type(padded_X))
print(f'PADDED\n{padded_X}')
print()

print('# ----- Output (without packing) -----')
emb_X = emb(padded_X)
print("emb_x", emb_X.shape)
rnn_out, hidden = rnn(emb_X)
# rnn_out, seq_len = pad_packed_sequence(rnn_out)

print(f'RNN_OUT\n{rnn_out}')
print(f'HIDDEN[0]\n{hidden[0]}')
print()

print('# ----- Output (with packing) -----')
packed_emb_X = pack_padded_sequence(emb_X, X_len, enforce_sorted=False)
print(packed_emb_X)
packed_rnn_out, hidden = rnn(packed_emb_X)
rnn_out, seq_len = pad_packed_sequence(packed_rnn_out)
print(f'RNN_OUT\n{rnn_out}')
print(rnn_out.shape)
print(f'HIDDEN[0]\n{hidden[0]}')

