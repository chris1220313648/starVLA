"""Frozen author-released LIBERO OAT, with a single raw-action normalization contract."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file

from .oat_vendor.codecs.latents.oat.modeling import RegisterQueryEncoder, SinglePassDecoder
from .oat_vendor.nn.quantization.fsq import FiniteScalarQuantization


class OATNormalization:
    def __init__(self, path):
        data = torch.load(Path(path) / 'norm_stats.pt', map_location='cpu', weights_only=True)
        if data['modes'] != {'action': 'min_max'}:
            raise ValueError('OAT requires its released min/max normalization')
        self.lo = data['stats']['action']['min'].numpy().astype(np.float32)
        self.hi = data['stats']['action']['max'].numpy().astype(np.float32)
        if self.lo.shape != (7,) or np.any(self.hi <= self.lo):
            raise ValueError('Invalid OAT action statistics')

    def normalize(self, actions):
        a = np.asarray(actions, dtype=np.float32).copy()
        if a.shape[-1] != 7 or not np.isfinite(a).all() or not np.isin(a[..., 6], [0, 1]).all():
            raise ValueError('Expected finite LIBERO actions with binary open gripper')
        a[..., 6] = 1 - 2 * a[..., 6]
        return (2 * (a - self.lo) / (self.hi - self.lo) - 1).astype(np.float32)

    def unnormalize(self, actions):
        a = np.asarray(actions, dtype=np.float32)
        if a.shape[-1] != 7 or not np.isfinite(a).all():
            raise ValueError('Invalid normalized OAT actions')
        a = (a + 1) * .5 * (self.hi - self.lo) + self.lo
        a[..., 6] = (1 - a[..., 6]) * .5
        return a.astype(np.float32)


class OATCodec(nn.Module):
    horizon, action_dim, token_count, codebook_size = 32, 7, 16, 1920

    def __init__(self, path):
        super().__init__()
        path = Path(path)
        cfg = json.loads((path / 'config.json').read_text())
        if (cfg['action_horizon'], cfg['action_dim'], cfg['latent_horizon'], cfg['quantizer_levels']) != (32, 7, 16, [8, 8, 6, 5]):
            raise ValueError('Unsupported OAT checkpoint geometry')
        shared = dict(sample_dim=7, sample_horizon=32, emb_dim=cfg['emb_dim'],
                      head_dim=cfg['head_dim'], pdropout=cfg['dropout'], latent_dim=cfg['latent_dim'])
        self.encoder = RegisterQueryEncoder(**shared, depth=cfg['encoder_depth'], register_schedule=(1,) * 16)
        self.decoder = SinglePassDecoder(**shared, depth=cfg['decoder_depth'],
                                        latent_dropout_mode=cfg['latent_dropout_mode'], latent_horizon=16)
        self.quantizer = FiniteScalarQuantization(levels=cfg['quantizer_levels'])
        self.load_state_dict(load_file(str(path / 'model.safetensors')), strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, actions):
        x = torch.as_tensor(np.asarray(actions), device=next(self.parameters()).device, dtype=torch.float32)
        if x.ndim != 3 or x.shape[1:] != (32, 7) or not torch.isfinite(x).all():
            raise ValueError('OAT expects finite [B,32,7] normalized actions')
        with torch.autocast(x.device.type, enabled=False):
            _, tokens = self.quantizer(self.encoder(x))
        return tokens.cpu().tolist()

    @torch.no_grad()
    def decode(self, tokens):
        raw = np.asarray(tokens)
        if raw.ndim != 2 or raw.shape[1] != 16 or not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0) or np.any(raw >= 1920):
            raise ValueError('OAT expects exactly 16 integer tokens in [0,1920)')
        x = torch.as_tensor(raw, device=next(self.parameters()).device, dtype=torch.long)
        with torch.autocast(x.device.type, enabled=False):
            result = self.decoder(self.quantizer.indices_to_embedding(x))
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite OAT decoded actions')
        return result.cpu().numpy()


def oat_token_constraint(action_ids, end_id):
    if len(action_ids) != 1920:
        raise ValueError('OAT requires 1920 action IDs')
    def allowed(generated):
        if len(generated) > 16 or any(t not in action_ids for t in generated):
            raise ValueError('Invalid OAT generation prefix')
        return [end_id] if len(generated) == 16 else action_ids
    return allowed
