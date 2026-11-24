"""DER++ mechanism adapted to V4's dual inputs and federated client lifetime.

Reference: https://github.com/aimagelab/mammoth/blob/master/models/derpp.py
Online pre-update logits, independent replay draws, and stream reservoir are
preserved. V4's optimizer, cosine margin, freezing and FedAvg are shared.
"""
import random

import torch
import torch.nn.functional as F

from run_federated_cl_v4 import V4Client


class ReferenceDERPPClient(V4Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_examples = 0

    def draw(self, size):
        items = random.sample(self.replay_buffer, min(size, len(self.replay_buffer)))
        stat = torch.stack([r[0] for r in items]).to(self.device)
        seq = torch.stack([r[1] for r in items]).to(self.device)
        labels = torch.tensor([r[2] for r in items], device=self.device)
        logits = torch.stack([r[3] for r in items]).to(self.device)
        return stat, seq, labels, logits

    def train(self, model, loader, task_id, logger):
        if self.components['dp']:
            raise ValueError('This matched comparison requires DP disabled for both methods')
        model.to(self.device).train()
        freeze = task_id > 0 and self.args.freeze_backbone
        optimizer = torch.optim.AdamW(self._trainable_parameters(model, freeze),
                                      lr=self.args.lr, weight_decay=1e-4)
        resources = {'optimizer_updates': 0, 'replay_batches': 0, 'replay_samples': 0}
        for _ in range(self.args.local_epochs):
            for stat, seq, labels in loader:
                stat, seq, labels = stat.to(self.device), seq.to(self.device), labels.to(self.device)
                outputs = model(stat, seq)
                loss = F.cross_entropy(model(stat, seq, targets=labels), labels)
                if self.replay_buffer:
                    rs, rq, _, old = self.draw(self.args.batch_size)
                    loss = loss + self.args.der_alpha * F.mse_loss(model(rs, rq), old)
                    resources['replay_batches'] += 1
                    resources['replay_samples'] += len(rs)
                    rs, rq, targets, _ = self.draw(self.args.batch_size)
                    loss = loss + self.args.der_beta * F.cross_entropy(model(rs, rq, targets=targets), targets)
                    resources['replay_batches'] += 1
                    resources['replay_samples'] += len(rs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                resources['optimizer_updates'] += 1
                for s, q, target, output in zip(stat, seq, labels, outputs.detach()):
                    index = self.seen_examples if self.seen_examples < self.args.replay_capacity else random.randrange(self.seen_examples + 1)
                    self.seen_examples += 1
                    if index >= self.args.replay_capacity:
                        continue
                    item = (s.detach().cpu().clone(), q.detach().cpu().clone(), int(target), output.cpu().clone())
                    if index == len(self.replay_buffer):
                        self.replay_buffer.append(item)
                    else:
                        self.replay_buffer[index] = item
                if self.args.fixed_updates and resources['optimizer_updates'] >= self.args.fixed_updates:
                    break
            if self.args.fixed_updates and resources['optimizer_updates'] >= self.args.fixed_updates:
                break
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        keys = list(state) if not freeze else [k for k in state if 'adapter' in k or 'classifier' in k]
        return state, keys, resources

    def store_replay(self, samples, label_map, model=None):
        # The buffer is updated online inside train, as in the reference.
        return
