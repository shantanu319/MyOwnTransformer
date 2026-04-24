import numpy as np
import torch


BIN_DTYPE = np.uint16


def load_bin(path):
    return np.memmap(path, dtype=BIN_DTYPE, mode='r')


def read_corpus(filename, tokenizer):
    seq = []
    with open(filename, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return seq


def data_feeder(data, batch_size, seq_len, device):
    total = len(data)
    num_sequences = total // seq_len

    for start_seq in range(0, num_sequences, batch_size):
        end_seq = start_seq + batch_size
        if end_seq > num_sequences:
            break
        slice_data = data[start_seq * seq_len: end_seq * seq_len]
        batch = torch.tensor(np.asarray(slice_data), dtype=torch.long, device=device)
        batch = batch.view(batch_size, seq_len)
        yield batch[:, :-1], batch[:, 1:]
