import torch


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
    data = data[:num_sequences * seq_len]
    data = torch.tensor(data, dtype=torch.long, device=device)

    data = data.view(num_sequences, seq_len)

    for i in range(0, num_sequences, batch_size):
        x = data[i:i + batch_size]
        if x.size(0) < batch_size:
            break
        yield x[:, :-1], x[:, 1:]
