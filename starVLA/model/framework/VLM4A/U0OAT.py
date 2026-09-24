"""U0 h=2 autoregressive OAT actions and future-image supervision."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.U0Fast import U0Fast, U0FastDefaultConfig
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.vlm.U0 import U0Interface
from starVLA.model.modules.vlm.u0_sequence import SequencePacker, contract_hash
from starVLA.model.modules.action_model.oat_ActionHeader import OATCodec, OATNormalization, oat_token_constraint
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register('u0oat')
@FRAMEWORK_REGISTRY.register('U0OAT')
class U0OAT(U0Fast):
    def __init__(self, config=None, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = merge_framework_config(U0FastDefaultConfig, config)
        cfg = self.config.framework
        self.config.framework.name = 'U0OAT'
        contract = OmegaConf.to_container(cfg.u0.sequence_contract, resolve=True)
        if contract_hash(contract) != cfg.u0.sequence_contract_sha256:
            raise ValueError('OAT sequence contract hash mismatch')
        if (contract.get('codec'), contract['sequence_h'], contract['action_horizon'], contract.get('action_token_length'), contract.get('action_token_count')) != ('oat', 2, 32, 16, 1920):
            raise ValueError('Invalid OAT sequence contract')
        if (int(cfg.u0.sequence_h), int(cfg.action_model.action_horizon), int(cfg.action_model.action_dim), int(cfg.u0.action_token_count)) != (2, 32, 7, 1920):
            raise ValueError('OAT configuration differs from contract')
        for filename, digest in contract['encoder_sha256'].items():
            with Path(filename).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError(f'Encoder identity mismatch: {filename}')
        self.u0_interface = U0Interface(self.config)
        self.register_buffer('action_token_ids', torch.tensor(self.u0_interface.action_token_ids))
        self.action_horizon, self.action_dim = 32, 7
        self.sequence_packer = SequencePacker(self.u0_interface, contract)
        # Input processor stays outside distributed trainable parameters and BF16 casts.
        object.__setattr__(self, 'oat_codec', OATCodec(cfg.action_model.oat_tokenizer_path))
        self.oat_normalization = OATNormalization(cfg.action_model.oat_tokenizer_path)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        parameter = next(self.oat_codec.parameters())
        target = fn(torch.empty(0, device=parameter.device, dtype=parameter.dtype))
        self.oat_codec.to(device=target.device)
        return self

    def forward(self, examples=None, **kwargs):
        actions = np.asarray([e['sequence_action'] for e in examples], dtype=np.float32)
        if actions.shape[1:] != (2, 32, 7) or not np.isfinite(actions).all():
            raise ValueError('Expected OAT normalized [B,2,32,7] actions')
        flat = self.oat_codec.encode(actions.reshape(-1, 32, 7))
        tokens = [flat[i:i+2] for i in range(0, len(flat), 2)]
        inputs = self.sequence_packer.training_inputs(examples, tokens)
        result = self.u0_interface.action_loss(inputs, self.sequence_packer.contract['future_image_weight'])
        result['sequence_tokens'] = inputs['attention_mask'].sum(-1).max().detach()
        if not torch.isfinite(result['action_loss']):
            raise RuntimeError('Nonfinite U0OAT loss')
        return result

    @torch.inference_mode()
    def predict_action(self, examples=None, **kwargs):
        if isinstance(examples, dict):
            examples = [examples]
        images, instructions = self._observations(examples)
        prompts = []
        for e, views in zip(examples, images):
            history = []
            for entry in e.get('history', []):
                tokens = entry['action_tokens']
                self.sequence_packer.action(tokens)
                grids = [self.u0_interface.encode_image_codes(v).cpu().numpy() for v in to_pil_preserve(entry['image'])]
                history.append(dict(state=entry['state'], image_codes=grids, action_tokens=tokens))
            grids = [self.u0_interface.encode_image_codes(v).cpu().numpy() for v in views]
            prompts.append(self.sequence_packer.prompt(e['lang'], e['state'], grids, history))
        constraint = oat_token_constraint(self.u0_interface.action_token_ids, self.sequence_packer.end)
        tokens = self.u0_interface.generate_fast_tokens(images, instructions, action_constraint=constraint, prompt_ids=prompts)
        return {'normalized_actions': self.oat_codec.decode(tokens), 'action_tokens': tokens}

    def unnormalize_actions(self, actions):
        return self.oat_normalization.unnormalize(actions)
