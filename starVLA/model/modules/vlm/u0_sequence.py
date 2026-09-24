"""Stateful observation/action sequence contract shared by training and serving."""
import hashlib
import json

import numpy as np
import torch


def contract_hash(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def state_bins(state, stats):
    state = np.asarray(state, dtype=np.float64)
    lo, hi = np.asarray(stats['q01']), np.asarray(stats['q99'])
    if state.shape != (8,) or lo.shape != (8,) or hi.shape != (8,) or not np.isfinite(state).all():
        raise ValueError('Expected finite raw state [8]')
    span = hi - lo
    unit = np.clip((state - lo) / np.where(span > 1e-8, span, 1), 0, 1)
    return np.where(span > 1e-8, np.minimum(np.floor(unit * 256), 255), 128).astype(np.int64)


class SequencePacker:
    def __init__(self, interface, contract):
        self.interface, self.contract = interface, contract
        self.h = int(contract['sequence_h'])
        if self.h < 1 or int(contract['action_horizon']) <= 0 or contract['state_dim'] != 8:
            raise ValueError('Invalid U0 sequence contract')
        self.tok = interface.tokenizer
        self.start = self.tok.convert_tokens_to_ids(self.tok.bss_token)
        self.end = self.tok.convert_tokens_to_ids(self.tok.ess_token)

    def text(self, value):
        if '<|' in value or '|>' in value:
            raise ValueError('Condition text contains reserved control tokens')
        ids = self.tok.encode(value, add_special_tokens=False)
        if any(t in self.interface.action_token_ids for t in ids):
            raise ValueError('Condition text contains IDs reserved for FAST actions')
        return ids

    def observation(self, state, codes, instruction=None):
        bins = ' '.join(map(str, state_bins(state, self.contract['state_statistics'])))
        grids = np.asarray(codes)
        size = self.interface.image_size // 16
        if grids.shape != (2, size, size):
            raise ValueError('Expected two camera grids at each observation')
        images = [t for grid in grids for t in self.interface.format_image_codes(grid)]
        if instruction is not None:
            ids = [self.tok.bos_token_id] + self.text(self.contract['prompt_template'].format(instruction=instruction, state=bins))
            ids += images
        else:
            ids = images + self.text(' State: ' + bins)
        return ids + self.text(' ASSISTANT: ') + [self.start]

    def action(self, tokens):
        if not tokens or any(not isinstance(t, (int, np.integer)) or not 0 <= t < len(self.interface.action_token_ids) for t in tokens):
            raise ValueError('Invalid FAST action token sequence')
        expected = self.contract.get('action_token_length')
        if expected is not None and len(tokens) != expected:
            raise ValueError('Invalid fixed-length action token sequence')
        return [self.interface.action_token_ids[int(t)] for t in tokens] + [self.end]

    def prompt(self, instruction, state, codes, history=()):
        if len(history) >= self.h:
            raise ValueError('History exceeds sequence_h')
        ids = []
        for i, entry in enumerate(history):
            ids += self.observation(entry['state'], entry['image_codes'], instruction if i == 0 else None)
            ids += self.action(entry.get('action_tokens', entry.get('fast_tokens')))
        ids += self.observation(state, codes, instruction if not history else None)
        return ids

    def training_inputs(self, examples, action_tokens):
        rows, labels = [], []
        for example, segments in zip(examples, action_tokens):
            states, codes = example['sequence_state'], example['sequence_image_codes']
            if len(states) != self.h or len(codes) != self.h or len(segments) != self.h:
                raise ValueError('Incomplete U0 sequence')
            row, target = [], []
            for i in range(self.h):
                obs = self.observation(states[i], codes[i], example['lang'] if i == 0 else None)
                row += obs
                target += [t if i > 0 and 151854 <= t < 282926 else -100 for t in obs]
                act = self.action(segments[i])
                row += act
                target += act
            row.append(self.tok.eos_token_id)
            target.append(self.tok.eos_token_id)
            if len(row) > self.interface.max_length:
                raise ValueError('U0 sequence exceeds max_length; refusing truncation')
            rows.append(row)
            labels.append(target)
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        targets = torch.full_like(ids, -100)
        for i, (row, label) in enumerate(zip(rows, labels)):
            ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
            targets[i, :len(row)] = torch.tensor(label)
        return {k: v.to(self.interface.input_device) for k, v in dict(input_ids=ids, attention_mask=mask, labels=targets).items()}
