import torch
import struct
import fire
FLOAT32 = 0
INT8 = 1


def get_metadata(w):
    with torch.no_grad():
        n_embd = w['emb.weight'].shape[1]
        n_layer = 0
        keys = list(w.keys())

        for x in keys:
            w[x].requires_grad = False
            if x == 'emb.weight' or 'ln0' in x:
                continue

            block_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
            n_layer = max(n_layer, block_id + 1)
    return n_embd, n_layer


def save_metadata(f, n_embd, n_layer):
    f.write(struct.pack('i', n_embd))
    f.write(struct.pack('i', n_layer))


def serialize_fp32(f, key, w):
    f.write(struct.pack('i', len(key)))
    f.write(bytes(key, 'ascii'))
    w[key]: torch.Tensor = w[key]
    if '.time_' in key:
        w[key] = w[key].squeeze()
    if '.time_decay' in key:
        w[key] = torch.exp(-torch.exp(w[key])).reshape(-1, 1, 1)
    if '.time_first' in key:
        w[key] = torch.exp(w[key]).reshape(-1, 1, 1)
    f.write(struct.pack('i', FLOAT32))
    ndim = w[key].ndim
    f.write(struct.pack('i', ndim))
    for i in w[key].shape:
        f.write(struct.pack('i', i))
    d = w[key].detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    f.write(b)


def export_checkpoint(data_file, target_file):
    data = torch.load(data_file, map_location='cpu')
    n_embd, n_layer = get_metadata(data)
    with open(target_file, 'wb') as f:
        save_metadata(f, n_embd, n_layer)
        for k in data.keys():
            serialize_fp32(f, k, data)


def export_vocab(vocab_file, target_file):
    idx2token = {}
    sorted = []  # must be already sorted
    lines = open(vocab_file, "r", encoding="utf-8").readlines()
    for l in lines:
        idx = int(l[:l.index(' ')])
        x = eval(l[l.index(' '):l.rindex(' ')])
        x = x.encode("utf-8") if isinstance(x, str) else x
        assert isinstance(x, bytes)
        assert len(x) == int(l[l.rindex(' '):])
        sorted += [x]
        idx2token[idx] = x

    token2idx = {}
    for k, v in idx2token.items():
        token2idx[v] = int(k)

    # precompute some tables for fast matching
    table = [[[] for j in range(256)] for i in range(256)]
    good = [set() for i in range(256)]
    wlen = [0 for i in range(256)]

    for i in reversed(range(len(sorted))):  # reverse order - match longer tokens first
        s = sorted[i]
        if len(s) >= 2:
            s0 = int(s[0])
            s1 = int(s[1])
            table[s0][s1] += [s]
            wlen[s0] = max(wlen[s0], len(s))
            good[s0].add(s1)
    with open(target_file, 'wb') as f:
        f.write(struct.pack('i', len(idx2token)))
        for i in idx2token:
            f.write(struct.pack('i', len(idx2token[i])))
            f.write(idx2token[i])

        for i in table:
            for j in i:
                f.write(struct.pack('i', len(j)))
                for k in j:
                    f.write(struct.pack('i', len(k)))
                    f.write(k)

        for i in wlen:
            f.write(struct.pack('i', i))

        for i in good:
            f.write(struct.pack('i', len(i)))
            for j in i:
                f.write(struct.pack('i', j))


if __name__ == '__main__':
    fire.Fire({'export_checkpoint': export_checkpoint,
               'export_vocab': export_vocab})
