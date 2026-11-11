"""PI05 block attention with a pretrained GroundingPI prefix and an adaRMS expert.

The prefix is bidirectional, cannot see actions, and supplies its native rotated
K/V at every layer. The action stream attends to prefix + action K/V. Both full
training and cached inference use the same layer operations below.
"""
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.VLM4A._groundingpi_config import grounding_config
from starVLA.model.framework.VLM4A.PI0 import _norm, _gated_residual
from starVLA.model.modules.action_model.OpenPI_ActionHead import GemmaDims, OpenPI05ActionHead
from starVLA.model.modules.vlm.GroundingPI import GroundingPIInterface
from starVLA.model.modules.action_model.fast_ActionHeader import Fast_Action_Tokenizer
from starVLA.model.tools import FRAMEWORK_REGISTRY


def state_prompt(instruction, state, state_dim):
    state = np.asarray(state, dtype=np.float32)
    if state.shape not in ((state_dim,), (1, state_dim)) or not np.isfinite(state).all():
        raise ValueError(f'Expected finite normalized state of shape ({state_dim},) or (1,{state_dim})')
    if np.any(np.abs(state) > 1.00001):
        raise ValueError('PI05 state must be normalized to [-1,1] before tokenization')
    bins = np.digitize(np.clip(state.reshape(-1), -1, 1), np.linspace(-1, 1, 257)[:-1]) - 1
    task = str(instruction).strip().replace('_', ' ').replace('\n', ' ')
    return f'Task: {task}, State: {" ".join(map(str, bins.tolist()))};\nAction: '


@FRAMEWORK_REGISTRY.register('GroundingPIPI05')
class GroundingPIPI05(baseframework):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = grounding_config(config, 'GroundingPIPI05')
        # Reading these contracts also retains them in accessed-only checkpoint
        # configs, which the policy server uses before constructing the model.
        data_cfg = self.config.datasets.vla_data
        if not data_cfg.get('include_state', True) or not data_cfg.get('normalize_state_on_server', True):
            raise ValueError('GroundingPIPI05 requires state input and server-side state normalization')
        cfg = self.config.framework.action_model
        self.action_dim = int(cfg.get('action_dim', 7))
        self.action_horizon = int(cfg.get('action_horizon', 16))
        self.state_dim = int(cfg.get('state_dim', 8))
        self.num_inference_steps = int(cfg.get('num_inference_steps', 10))
        self.gradient_checkpointing = bool(self.config.framework.qwenvl.enable_gradient_checkpointing)
        self.knowledge_insulation = bool(self.config.framework.get("knowledge_insulation", False))
        self.cot_loss_weight = float(self.config.framework.get('cot_loss_weight', 0.0))
        if not np.isfinite(self.cot_loss_weight) or self.cot_loss_weight < 0:
            raise ValueError('cot_loss_weight must be finite and nonnegative')
        if self.cot_loss_weight and self.knowledge_insulation:
            raise ValueError('CoT supervision and knowledge insulation cannot be combined')
        self.cot_training_mode = str(self.config.framework.get('cot_training_mode', 'legacy'))
        if self.cot_training_mode not in ('legacy', 'shared_canvas_v1', 'history_canvas_v1', 'separate_head_v1'):
            raise ValueError('Unsupported cot_training_mode')
        self.config.datasets.vla_data.get('history_frames', 0)
        self.config.datasets.vla_data.get('history_stride', 40)
        self.cot_prompt = str(self.config.framework.get('cot_prompt', ''))
        if self.cot_training_mode in ('shared_canvas_v1', 'history_canvas_v1') and (not self.cot_loss_weight or not self.cot_prompt.strip()):
            raise ValueError('Shared canvas CoT requires a positive loss weight and a prompt')
        self.state_representation = str(cfg.get('state_representation', 'discrete_text'))
        if self.state_representation not in ('discrete_text', 'continuous_expert_token'):
            raise ValueError('Unsupported state_representation')
        self.continuous_state = self.state_representation == 'continuous_expert_token'
        if self.continuous_state:
            # Read serving contracts so accessed-only checkpoints retain them.
            self.observation_image_size = data_cfg.get('obs_image_size', [384, 320])
            self.image_layout = data_cfg.get('image_layout', 'head_top_wrists_bottom')
            self.quaternion_order = data_cfg.get('quaternion_order', 'xyzw')
            if self.quaternion_order not in ('xyzw', 'wxyz'):
                raise ValueError('Unsupported quaternion_order')
        if self.continuous_state and self.knowledge_insulation:
            raise ValueError('Continuous expert state tokens do not yet support knowledge insulation')
        self.qwen_vl_interface = GroundingPIInterface(self.config)
        # Text supervision needs the pretrained vocabulary head.
        if self.knowledge_insulation:
            self.fast_tokenizer = Fast_Action_Tokenizer(cfg.get("fast_tokenizer", "playground/Pretrained_models/fast"))
            self.fast_tokenizer.fast_tokenizer.time_horizon = self.action_horizon
            self.fast_tokenizer.fast_tokenizer.action_dim = self.action_dim
            if self.fast_tokenizer.fast_tokenizer.vocab_size != self.config.framework.qwenvl.action_vocab_size:
                raise ValueError("FAST vocabulary size must match action_vocab_size")
        elif not self.cot_loss_weight:
            self.qwen_vl_interface.model.lm_head = nn.Identity()
        text = self.language_model.config
        dims = GemmaDims(width=int(cfg.get('expert_width', 1024)), depth=text.num_hidden_layers,
                         mlp_dim=int(cfg.get('expert_mlp_dim', 4096)),
                         num_heads=text.num_attention_heads, num_kv_heads=text.num_key_value_heads,
                         head_dim=text.head_dim, vocab_size=1)
        self.action_model = OpenPI05ActionHead(dims, action_dim=self.action_dim,
                                               action_horizon=self.action_horizon, max_state_dim=self.state_dim)
        if self.continuous_state:
            self.action_model.state_encoder = nn.Sequential(
                nn.Linear(self.state_dim, dims.width), nn.ReLU(), nn.Linear(dims.width, dims.width))
        self.action_model.gemma_expert.lm_head = nn.Identity()
        # FP32 master weights in the small expert; BF16 attention under autocast.
        # The final prefix layer's output and final norm cannot affect action K/V.
        # Keep their pretrained parameters but exclude them from the optimizer.
        last = self.language_model.layers[-1]
        for module in (last.self_attn.q_proj, last.self_attn.q_norm, last.self_attn.o_proj,
                       last.post_attention_layernorm, last.mlp, self.language_model.norm):
            if not self.knowledge_insulation and not self.cot_loss_weight:
                module.requires_grad_(False)
        self.validate_attention_layout()
        self._gradient_norms = {}
        self._updated = {}
        watched = {'backbone': self.language_model.layers[0].self_attn.k_proj.weight,
                   'expert': self.action_model.layers[0].self_attn.q_proj.weight,
                   'time': self.action_model.time_mlp_in.weight,
                   'decoder': self.action_model.action_out_proj.weight}
        if self.continuous_state:
            watched['state'] = self.action_model.state_encoder[0].weight
        if self.knowledge_insulation:
            watched['lm_head'] = self.qwen_vl_interface.model.lm_head.weight
        self._watched = watched
        self._initial_samples = {k: p.detach().flatten()[:256].float().cpu().clone() for k, p in watched.items()}
        for name, parameter in watched.items():
            def record(grad, name=name):
                self._gradient_norms[name] = float(grad.detach().float().norm())
            parameter.register_hook(record)

    @property
    def language_model(self):
        return self.qwen_vl_interface.model.model.language_model

    def validate_attention_layout(self):
        if len(self.language_model.layers) != len(self.action_model.layers):
            raise ValueError('Prefix and action expert depths must match')
        for prefix, expert in zip(self.language_model.layers, self.action_model.layers):
            for name in ('q_proj', 'k_proj', 'v_proj'):
                if getattr(prefix.self_attn, name).out_features != getattr(expert.self_attn, name).out_features:
                    raise ValueError(f'Prefix/expert {name} attention dimensions do not match')
            if prefix.self_attn.head_dim != expert.self_attn.head_dim:
                raise ValueError('Prefix/expert head dimensions do not match')

    def _prefix_prompts(self, examples):
        prompts = ([f'Task: {str(e["lang"]).strip()};\nAction: ' for e in examples]
                   if self.continuous_state else
                   [state_prompt(e['lang'], e['state'], self.state_dim) for e in examples])
        if self.cot_training_mode in ('shared_canvas_v1', 'history_canvas_v1'):
            prompts = [prompt.removesuffix('Action: ') + self.cot_prompt for prompt in prompts]
        return prompts

    def _prefix_inputs(self, examples, solutions=None):
        if not examples:
            raise ValueError('At least one example is required')
        if any(not e['image'] for e in examples):
            raise ValueError('Each example requires images')
        images = to_pil_preserve([e['image'] for e in examples])
        if self.cot_training_mode in ('shared_canvas_v1', 'history_canvas_v1'):
            count = 5 if self.cot_training_mode == 'history_canvas_v1' else 1
            if any(len(views) != count or any(im.size != (320, 384) for im in views) for views in images):
                raise ValueError(f'Canvas CoT requires {count} 320x384 composites per sample')
        elif self.cot_training_mode == 'separate_head_v1':
            if any(len(views) != 3 or any(image.size != (320, 240) for image in views) for views in images):
                raise ValueError('Separate-view CoT requires three 320x240 images per sample')
        kwargs = dict(solutions=solutions, allow_missing_solutions=True) if solutions is not None else {}
        return self.qwen_vl_interface.build_qwenvl_inputs(images, self._prefix_prompts(examples), **kwargs)

    def prepare_prefix(self, examples):
        inputs = self._prefix_inputs(examples)
        return self._embed_inputs(inputs), inputs.attention_mask.bool()

    def _embed_inputs(self, inputs):
        core = self.qwen_vl_interface.model.model
        embeddings = core.get_input_embeddings()(inputs.input_ids)
        features = torch.cat(core.get_image_features(inputs.pixel_values, inputs.image_grid_thw), dim=0)
        features = features.to(embeddings)
        image_mask, _ = core.get_placeholder_mask(inputs.input_ids, embeddings, image_features=features)
        return embeddings.masked_scatter(image_mask, features)

    def _qkv(self, layer, hidden, cond, positions, *, prefix):
        normalized, gate = _norm(layer.input_layernorm, hidden, cond)
        attn = layer.self_attn
        shape = (*hidden.shape[:-1], -1, attn.head_dim)
        q = attn.q_proj(normalized.to(attn.q_proj.weight.dtype)).view(shape)
        k = attn.k_proj(normalized.to(attn.k_proj.weight.dtype)).view(shape)
        v = attn.v_proj(normalized.to(attn.v_proj.weight.dtype)).view(shape)
        if prefix:
            q, k = attn.q_norm(q), attn.k_norm(k)
        q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2),
                                   *self.language_model.rotary_emb(q, positions))
        return q, k, v.transpose(1, 2), gate

    @staticmethod
    def _attend(q, k, v, valid_keys):
        # Explicit mask, never causal: block visibility is enforced by the K/V
        # supplied to each stream. repeat_kv also works on older SDPA versions.
        groups = q.shape[1] // k.shape[1]
        out = F.scaled_dot_product_attention(q, repeat_kv(k.to(q.dtype), groups),
                                             repeat_kv(v.to(q.dtype), groups),
                                             attn_mask=valid_keys[:, None, None, :] if valid_keys.ndim == 2 else valid_keys, dropout_p=0.0)
        return out.transpose(1, 2).flatten(2)

    @staticmethod
    def _finish(layer, hidden, attended, gate, cond):
        out = layer.self_attn.o_proj(attended.to(layer.self_attn.o_proj.weight.dtype))
        hidden = _gated_residual(hidden, out, gate)
        normalized, gate = _norm(layer.post_attention_layernorm, hidden, cond)
        out = layer.mlp(normalized.to(layer.mlp.up_proj.weight.dtype))
        return _gated_residual(hidden, out, gate)

    def _prefix_layer(self, index, prefix, mask, positions):
        layer = self.language_model.layers[index]
        q, k, v, gate = self._qkv(layer, prefix, None, positions, prefix=True)
        # Last prefix output is unused: actions read the pre-attention K/V.
        if index < len(self.language_model.layers) - 1:
            prefix = self._finish(layer, prefix, self._attend(q, k, v, mask), gate, None)
        return prefix, k, v

    def _action_layer(self, index, suffix, cond, positions, prefix_k, prefix_v, mask):
        layer = self.action_model.layers[index]
        q, k, v, gate = self._qkv(layer, suffix, cond, positions, prefix=False)
        k, v = torch.cat((prefix_k.to(k.dtype), k), dim=2), torch.cat((prefix_v.to(v.dtype), v), dim=2)
        valid = torch.cat((mask, torch.ones(suffix.shape[:2], dtype=torch.bool, device=mask.device)), dim=1)
        return self._finish(layer, suffix, self._attend(q, k, v, valid), gate, cond)

    def _joint_layer(self, index, prefix, suffix, cond, mask, prefix_pos, action_pos):
        prefix, k, v = self._prefix_layer(index, prefix, mask, prefix_pos)
        suffix = self._action_layer(index, suffix, cond, action_pos, k, v, mask)
        return prefix, suffix

    def _positions(self, mask):
        prefix = (mask.long().cumsum(-1) - 1).clamp_min(0)
        action = mask.sum(-1)[:, None] + torch.arange(self.action_horizon + int(self.continuous_state), device=mask.device)[None]
        return prefix, action

    def velocity(self, prefix, mask, noisy_actions, time, cache=None, state=None):
        suffix, _, _, cond = self.action_model.embed_action_suffix(state, noisy_actions, time)
        prefix_pos, action_pos = self._positions(mask)
        for index in range(len(self.action_model.layers)):
            if cache is not None:
                if self.training and self.gradient_checkpointing:
                    suffix = checkpoint(partial(self._action_layer, index), suffix, cond, action_pos, *cache[index], mask, use_reentrant=False)
                else:
                    suffix = self._action_layer(index, suffix, cond, action_pos, *cache[index], mask)
            elif self.training and self.gradient_checkpointing:
                prefix, suffix = checkpoint(partial(self._joint_layer, index), prefix, suffix, cond,
                                             mask, prefix_pos, action_pos, use_reentrant=False)
            else:
                prefix, suffix = self._joint_layer(index, prefix, suffix, cond, mask, prefix_pos, action_pos)
        suffix, _ = _norm(self.action_model.norm, suffix, cond)
        return self.action_model.decode_action_velocity(suffix)

    def prefix_cache(self, prefix, mask):
        positions, _ = self._positions(mask)
        cache = []
        for index in range(len(self.language_model.layers)):
            prefix, k, v = self._prefix_layer(index, prefix, mask, positions)
            cache.append((k, v))
        return tuple(cache)

    def _ki_layer(self, index, hidden, attention, positions):
        layer = self.language_model.layers[index]
        q, k, v, gate = self._qkv(layer, hidden, None, positions, prefix=True)
        hidden = self._finish(layer, hidden, self._attend(q, k, v, attention), gate, None)
        return hidden, k, v

    @staticmethod
    def ki_attention(mask, labels):
        prefix = mask & labels.eq(-100)
        length = mask.shape[1]
        causal = torch.arange(length, device=mask.device)[None, :] <= torch.arange(length, device=mask.device)[:, None]
        # Prefix is bidirectional; teacher-forced answers can never enter prefix.
        allowed = prefix[:, None, :] | (~prefix[:, :, None] & causal[None])
        return allowed[:, None] & mask[:, None, None, :], prefix

    def _fast_ce(self, hidden, labels, normalized=False, allow_empty=False):
        targets = labels[:, 1:]
        valid = targets.ne(-100)
        hidden = hidden[:, :-1][valid]
        if not normalized:
            hidden = self.language_model.norm(hidden)
        targets = targets[valid]
        head = self.qwen_vl_interface.model.lm_head
        if targets.numel() == 0:
            if not allow_empty:
                raise ValueError('Empty assistant supervision')
            # Keep the text-only parameters in the graph on unlabelled ranks.
            return hidden.float().sum() * 0 + head.weight.reshape(-1)[0].float() * 0
        def chunk_loss(h, y):
            return F.cross_entropy(head(h.to(head.weight.dtype)).float(), y, reduction='sum')
        loss = hidden.new_zeros((), dtype=torch.float32)
        for h, y in zip(hidden.split(128), targets.split(128)):
            loss = loss + checkpoint(chunk_loss, h, y, use_reentrant=False)
        return loss / targets.numel()

    def ki_forward(self, examples, actions, noise=None, time=None):
        ids = self.fast_tokenizer.encoder_action2fastoken(actions.detach().cpu().numpy())
        vocab = int(self.config.framework.qwenvl.action_vocab_size)
        if any(not seq or any(int(i) < 0 or int(i) >= vocab for i in seq) for seq in ids):
            raise ValueError('Invalid FAST action tokens')
        solutions = [''.join(f'<robot_action_{int(i)}>' for i in seq) for seq in ids]
        prompts = [state_prompt(e['lang'], e['state'], self.state_dim) for e in examples]
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            to_pil_preserve([e['image'] for e in examples]), prompts, solutions=solutions)
        hidden = self._embed_inputs(inputs)
        mask, labels = inputs.attention_mask.bool(), inputs.labels
        attention, prefix_mask = self.ki_attention(mask, labels)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        cache = []
        for index in range(len(self.language_model.layers)):
            if self.training and self.gradient_checkpointing:
                hidden, k, v = checkpoint(partial(self._ki_layer, index), hidden, attention, positions, use_reentrant=False)
            else:
                hidden, k, v = self._ki_layer(index, hidden, attention, positions)
            # KI boundary: continuous actions receive conditioning, never gradients.
            cache.append((k.detach(), v.detach()))
        fast_loss = self._fast_ce(hidden, labels)
        noisy, target, _, time = self.action_model.prepare_flow_targets(actions, noise, time)
        velocity = self.velocity(None, prefix_mask, noisy, time, cache=cache)
        fm_loss = F.mse_loss(velocity.float(), target.float())
        return {'action_loss': fm_loss, 'fast_token_loss': fast_loss, 'total_loss': fm_loss + fast_loss}

    def _cot_joint_layer(self, index, hidden, suffix, cond, attention, prefix_mask, text_pos, action_pos):
        hidden, k, v = self._ki_layer(index, hidden, attention, text_pos)
        # Actions only read public-prefix K/V; teacher-forced answers stay masked.
        suffix = self._action_layer(index, suffix, cond, action_pos, k, v, prefix_mask)
        return hidden, suffix

    def _cot_loss(self, hidden, labels, examples):
        ce = self._fast_ce(hidden, labels, allow_empty=True)
        tokens = labels[:, 1:].ne(-100).sum().to(dtype=torch.float32)
        total_tokens = tokens.detach().clone()
        samples = tokens.new_tensor(sum(bool(e.get('cot')) for e in examples))
        world = 1
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(total_tokens)
            torch.distributed.all_reduce(samples)
            world = torch.distributed.get_world_size()
        # DDP averages gradients; scale local sums to a global token mean.
        local_loss = ce * tokens * world / total_tokens.clamp_min(1)
        if world > 1:
            reported = local_loss.detach().clone()
            torch.distributed.all_reduce(reported)
            local_loss = local_loss + (reported / world - local_loss.detach())
        return local_loss, samples

    def _cot_velocity(self, examples, noisy, time):
        if self.cot_training_mode not in ('shared_canvas_v1', 'history_canvas_v1', 'separate_head_v1'):
            raise ValueError('CoT training requires an explicit coordinate-aware mode')
        coordinate_frame = ('composite_320x384_v1' if self.cot_training_mode in ('shared_canvas_v1', 'history_canvas_v1')
                            else 'head_camera_320x240_v1')
        if any(e.get('cot') and e.get('cot_coordinate_frame') != coordinate_frame for e in examples):
            raise ValueError(f'CoT labels must use {coordinate_frame} coordinates')
        inputs = self._prefix_inputs(examples, solutions=[e.get('cot') for e in examples])
        hidden = self._embed_inputs(inputs)
        mask, prefix_mask = inputs.attention_mask.bool(), inputs.prefix_mask.bool()
        attention, _ = self.ki_attention(mask, inputs.labels)
        # For missing answers all real tokens belong to the public prefix.
        if not torch.equal(prefix_mask, mask & inputs.labels.eq(-100)):
            raise ValueError('Assistant boundary and public prefix disagree')
        text_pos = (mask.long().cumsum(-1) - 1).clamp_min(0)
        _, action_pos = self._positions(prefix_mask)
        suffix, _, _, cond = self.action_model.embed_action_suffix(self._continuous_state(examples), noisy, time)
        for index in range(len(self.language_model.layers)):
            args = (hidden, suffix, cond, attention, prefix_mask, text_pos, action_pos)
            if self.training and self.gradient_checkpointing:
                hidden, suffix = checkpoint(partial(self._cot_joint_layer, index), *args, use_reentrant=False)
            else:
                hidden, suffix = self._cot_joint_layer(index, *args)
        suffix, _ = _norm(self.action_model.norm, suffix, cond)
        velocity = self.action_model.decode_action_velocity(suffix)
        cot_loss, count = self._cot_loss(hidden, inputs.labels, examples)
        return velocity, cot_loss, count

    def _continuous_state(self, examples):
        if not self.continuous_state:
            return None
        state = np.stack([np.asarray(e['state'], dtype=np.float32).reshape(-1) for e in examples])
        if state.shape != (len(examples), self.state_dim) or not np.isfinite(state).all():
            raise ValueError(f'Expected finite state of shape (B,{self.state_dim})')
        return torch.as_tensor(state, device=next(self.parameters()).device, dtype=torch.float32)

    def forward(self, examples, noise=None, time=None, **kwargs):
        actions = np.stack([e['action'] for e in examples])
        expected = (len(examples), self.action_horizon, self.action_dim)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f'Expected finite actions with shape {expected}')
        actions = torch.as_tensor(actions, device=next(self.parameters()).device, dtype=torch.float32)
        if self.knowledge_insulation:
            return self.ki_forward(examples, actions, noise, time)
        noisy, target, _, time = self.action_model.prepare_flow_targets(actions, noise, time)
        if self.cot_loss_weight:
            velocity, cot_loss, cot_samples = self._cot_velocity(examples, noisy, time)
        else:
            prefix, mask = self.prepare_prefix(examples)
            velocity = self.velocity(prefix, mask, noisy, time, state=self._continuous_state(examples))
        if self.continuous_state:
            valid = torch.as_tensor(np.stack([e['action_mask'] for e in examples]), device=actions.device, dtype=torch.bool)
            if valid.shape != actions.shape[:2] or not valid.any():
                raise ValueError('Expected a nonempty action_mask of shape (B,H)')
            loss = F.mse_loss(velocity.float()[valid], target.float()[valid])
        else:
            loss = F.mse_loss(velocity.float(), target.float())
        output = {'action_loss': loss}
        if self.cot_loss_weight:
            output.update(cot_loss=cot_loss, cot_samples=cot_samples,
                          total_loss=loss + self.cot_loss_weight * cot_loss)
        return output

    @torch.no_grad()
    def predict_action(self, examples, noise=None, num_inference_steps=None, **kwargs):
        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        if steps < 1:
            raise ValueError('num_inference_steps must be positive')
        device = next(self.parameters()).device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            prefix, mask = self.prepare_prefix(examples)
            cache = self.prefix_cache(prefix, mask)
            state = self._continuous_state(examples)
            shape = (len(examples), self.action_horizon, self.action_dim)
            actions = torch.randn(shape, device=device) if noise is None else torch.as_tensor(noise, device=device, dtype=torch.float32).clone()
            if actions.shape != shape or not torch.isfinite(actions).all():
                raise ValueError(f'Expected finite noise with shape {shape}')
            for step in range(steps):
                time = torch.full((len(examples),), 1 - step / steps, device=device, dtype=torch.float32)
                actions = actions - self.velocity(None, mask, actions, time, cache=cache, state=state).float() / steps
        return {'normalized_actions': actions.float().cpu().numpy()}

    @torch.no_grad()
    def training_health_check(self):
        for name, parameter in self._watched.items():
            changed = not torch.equal(parameter.detach().flatten()[:256].float().cpu(), self._initial_samples[name])
            self._updated[name] = self._updated.get(name, False) or changed
        values = list(self._gradient_norms.values())
        result = {'representative_grad_norms': dict(self._gradient_norms),
                  'representative_parameters_updated': dict(self._updated),
                  'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                  'stop': not all(np.isfinite(v) for v in values)}
        if torch.distributed.is_initialized():
            ranks = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ranks, result)
            result = dict(result, ranks=ranks)
        return result
