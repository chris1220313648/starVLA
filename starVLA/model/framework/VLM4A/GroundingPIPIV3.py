"""GroundingPI adapter for the unchanged QwenPI_v3 action architecture."""
import copy
import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.framework.VLM4A._groundingpi_config import grounding_config
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register('GroundingPIPIV3')
class GroundingPIPIV3(Qwen_PI_v3):
    def __init__(self, config=None, **kwargs):
        super().__init__(grounding_config(config, 'GroundingPIPIV3'), **kwargs)
        if not self.config.datasets.vla_data.get('normalize_state_on_server', False):
            raise ValueError('GroundingPIPIV3 requires normalize_state_on_server=true')
        self._health_samples = None
        self._dead_checks = 0
        self._gradient_norms = {}
        # Representative gradients before optimizer clipping/partitioning.
        watched = {
            'backbone': self.qwen_vl_interface.model.get_input_embeddings().weight,
            'projection': self.project_layers[-1][1].weight,
            'expert': self.action_model.model.transformer_blocks[0].attn1.to_q.weight,
            'decoder': self.action_model.action_decoder.layer1.weight,
        }
        for name, param in watched.items():
            def record(grad, name=name):
                self._gradient_norms[name] = float(grad.detach().float().norm())
            param.register_hook(record)

    def _encode_vl_hidden_states(self, batch_images, instructions):
        states, mask = super()._encode_vl_hidden_states(batch_images, instructions)
        dtype = next(self.action_model.parameters()).dtype
        return [state.to(dtype=dtype) for state in states], mask

    def forward(self, examples=None, **kwargs):
        if self.training and self._health_samples is None:
            self._health_samples = copy.deepcopy(examples[:2])
        return super().forward(examples=examples, **kwargs)

    @torch.no_grad()
    def training_health_check(self):
        samples = self._health_samples
        if not samples:
            raise RuntimeError('No fixed training samples for health check')
        device = next(self.parameters()).device
        modes = {module: module.training for module in self.modules()}
        positive = []
        def inspect(module, inputs, output):
            x = output[:, -self.action_horizon:]
            if not torch.isfinite(x).all():
                raise FloatingPointError('Non-finite decoder activation')
            positive.append(float((x > 0).float().mean()))
        handle = self.action_model.action_decoder.layer1.register_forward_hook(inspect)
        try:
            self.eval()
            with torch.random.fork_rng(devices=[device.index]):
                def predict(items):
                    torch.random.default_generator.manual_seed(7)
                    torch.cuda.manual_seed(7)
                    return self.predict_action(items)['normalized_actions']
                original = predict(samples)
                changes = {}
                for case in ('image', 'lang', 'state'):
                    altered = copy.deepcopy(samples)
                    for item in altered:
                        if case == 'image':
                            item['image'] = [Image.new('RGB', im.size) for im in item['image']]
                        elif case == 'lang':
                            item['lang'] = 'do nothing and keep the robot still'
                        else:
                            item['state'] = np.zeros_like(item['state'])
                    pred = predict(altered)
                    changes[case] = float(np.max(np.abs(pred-original)))
                if not np.isfinite(original).all():
                    raise FloatingPointError('Non-finite health prediction')
                target = np.stack([s['action'] for s in samples])
                dead = max(positive) == 0 and max(changes.values()) == 0
                self._dead_checks = self._dead_checks + 1 if dead else 0
                return dict(decoder_positive_fraction=min(positive),
                            condition_max_abs_change=changes,
                            normalized_action_mae=float(np.abs(original-target).mean()),
                            representative_grad_norms=dict(self._gradient_norms),
                            consecutive_dead_checks=self._dead_checks,
                            stop=self._dead_checks >= 2)
        finally:
            handle.remove()
            for module, training in modes.items():
                module.training = training
