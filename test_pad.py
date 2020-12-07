import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
seq = torch.tensor(
    [[[1,2,3], [4,5,6], [0,0,0]], 
    [[3,4,5], [0,0,0], [0,0,0]],
    [[4,5,6], [7,8,9], [10,11,12]],
    [[13,14,15], [16,17,18], [19,20,21]]])
lens = [2, 1, 3, 3]
print(seq)
packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
print(packed)
seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
print(seq_unpacked)
print(lens_unpacked)
