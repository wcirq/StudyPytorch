# -*- encoding: utf-8 -*-
"""
@File    : card_sorting.py
@Time    : 2023/5/26 16:19
@Author  : wcirq
@Software: PyCharm
@Link: https://zhuanlan.zhihu.com/p/339207092
"""
import copy

import numpy as np
import torch.nn as nn
import time
import torch
from matplotlib import pyplot as plt

from transformer.losses import SimpleLossCompute
from transformer.model import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, EncoderDecoder, \
    Encoder, Decoder, EncoderLayer, DecoderLayer, Embeddings, Generator, Variable, subsequent_mask, LabelSmoothing

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y.long(), batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    # for i in range(nbatches):
    #     data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
    #     data[:, 0] = 1
    #     src = Variable(data, requires_grad=False)
    #     tgt = Variable(data, requires_grad=False)
    #     yield Batch(src, tgt, 0)

    for i in range(nbatches):
        sample = np.random.randint(1, V, size=(batch, 10))
        inputs = torch.from_numpy(sample)
        outputs = torch.from_numpy(np.sort(sample, axis=1))
        src = Variable(inputs, requires_grad=False)
        tgt = Variable(outputs, requires_grad=False)
        yield Batch(src, tgt, 0)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def test(model):
    model.eval()
    src = Variable(torch.LongTensor(np.random.randint(1, 11, (1, 10))))
    src_mask = Variable(torch.ones(1, 1, 10))
    print(src, "->", greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

def mian():
    # tmp_model = make_model(10, 10, 2)

    # opts = [NoamOpt(512, 1, 4000, None),
    #         NoamOpt(512, 1, 8000, None),
    #         NoamOpt(256, 1, 4000, None)]
    # plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    # plt.legend(["512:4000", "512:8000", "256:4000"])
    #
    # plt.show()

    V = 10
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9))

    datas = data_gen(V, 30, 20)

    for epoch in range(2000):
        model.train()
        run_epoch(data_gen(V, 100, 20), model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 20, 5), model,
                        SimpleLossCompute(model.generator, criterion, None)))

        test(model)


if __name__ == '__main__':
    mian()