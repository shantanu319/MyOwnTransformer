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


def load_tinystories(tokenizer, split='train', max_docs=None):
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    tokens = []
    for i, row in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        ids = tokenizer(row['text'])['input_ids']
        tokens.extend(ids)
        tokens.append(eos_id)
    return tokens


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
