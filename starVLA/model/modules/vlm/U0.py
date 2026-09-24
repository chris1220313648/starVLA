"""U0-4B backbone with the existing Emu/FAST image and prompt contract."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from .Emu3_5 import ACTION_TOKEN_COUNT, ACTION_TOKEN_START_ID, Emu3_5Interface
from .emu3_5.vision_tokenizer import build_vision_tokenizer
from .u0.configuration_unis import UNISConfig
from .u0.modeling_unis import UNISForCausalLM
from .u0.tokenization_unis import UNISTokenizer


def fast_token_constraint(bpe_tokenizer, action_ids, eos_id, coefficient_count):
    """Allow exactly H*D UTF-8 coefficients, including split-byte BPE tokens.

    This constrains FAST's serialization shape, never the predicted values.
    A malformed stream must not reach FAST's silent zero-action fallback.
    """
    if coefficient_count <= 0:
        raise ValueError("FAST coefficient count must be positive")
    byte_decoder = {char: byte for byte, char in bytes_to_unicode().items()}
    pieces = {
        token_id: bytes(byte_decoder[c] for c in bpe_tokenizer.convert_ids_to_tokens(i))
        for i, token_id in enumerate(action_ids)
    }

    def state(raw):
        try:
            return len(raw.decode("utf-8")), False
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data":
                return None
            return len(raw[:exc.start].decode("utf-8")), True

    def allowed(generated_ids):
        raw = b"".join(pieces[t] for t in generated_ids)
        current = state(raw)
        if current == (coefficient_count, False):
            return [eos_id]
        result = []
        for token_id, piece in pieces.items():
            candidate = state(raw + piece)
            if candidate is not None:
                count, pending = candidate
                if count < coefficient_count or (count == coefficient_count and not pending):
                    result.append(token_id)
        if not result:
            raise RuntimeError("No valid FAST continuation for the configured action shape")
        return result

    return allowed


def action_cross_entropy(hidden, labels, head, chunk_size=64, future_image_weight=None):
    """Causal full-vocabulary CE, materializing logits only for supervised tokens."""
    if chunk_size <= 0 or hidden.shape[:2] != labels.shape:
        raise ValueError("Invalid loss chunk size or hidden/label shape")
    targets = labels[:, 1:].reshape(-1)
    valid = targets != -100
    states = hidden[:, :-1].reshape(-1, hidden.shape[-1])[valid]
    targets = targets[valid]
    if not targets.numel():
        raise ValueError("U0Fast batch contains no supervised action tokens")

    def ce(x, y):
        logits = head(x).float()
        return F.cross_entropy(logits, y, reduction="none"), logits.detach().argmax(-1).eq(y)

    losses, correct = [], []
    for x, y in zip(states.split(chunk_size), targets.split(chunk_size)):
        loss, match = (checkpoint(ce, x, y, use_reentrant=False)
                       if torch.is_grad_enabled() and x.requires_grad else ce(x, y))
        losses.append(loss)
        correct.append(match)
    losses, correct = torch.cat(losses), torch.cat(correct)
    if future_image_weight is None:
        return losses.mean()
    visual = (targets >= 151854) & (targets < 282926)
    action = (targets >= 149595) & (targets < 151643)
    if not visual.any() or not action.any() or future_image_weight <= 0:
        raise ValueError('Interleaved loss requires action and future image targets')
    text_ce, image_ce = losses[~visual].mean(), losses[visual].mean()
    return dict(action_loss=text_ce + future_image_weight * image_ce,
                action_ce=text_ce.detach(), future_image_ce=image_ce.detach(),
                action_token_count=action.sum(), future_image_token_count=visual.sum(),
                action_accuracy=correct[action].float().mean())


class U0Interface(Emu3_5Interface):
    """Reuse token packing, with a native UNIS model and full-parameter training."""

    def __init__(self, config):
        nn.Module.__init__(self)
        cfg = config.framework.u0
        self.image_size = int(cfg.image_size)
        self.max_length = int(cfg.max_length)
        self.max_new_tokens = int(cfg.max_new_tokens)
        self.loss_chunk_tokens = int(cfg.loss_chunk_tokens)
        if self.image_size <= 0 or self.image_size % 16:
            raise ValueError("U0 image_size must be a positive multiple of 16")
        if min(self.max_length, self.max_new_tokens, self.loss_chunk_tokens) <= 0:
            raise ValueError("U0 sequence and loss limits must be positive")
        backend = cfg.attn_implementation
        if backend not in ("eager", "flash_attention_2"):
            raise ValueError("U0Fast supports eager/flash_attention_2; upstream SDPA omits Q/K normalization")
        path = Path(cfg.get("tokenizer_path", cfg.base_vlm))
        self.tokenizer = UNISTokenizer(
            vocab_file=str(path / "unis.tiktoken"),
            special_tokens_file=str(path / "unis_vision_tokens.txt"),
        )
        self._set_special_tokens()
        model_cfg = UNISConfig.from_pretrained(cfg.base_vlm, local_files_only=True)
        start = int(cfg.get("action_token_start_id", ACTION_TOKEN_START_ID))
        count = int(cfg.get('action_token_count', ACTION_TOKEN_COUNT))
        if not 0 < count <= ACTION_TOKEN_COUNT:
            raise ValueError('Invalid U0 action token count')
        self.action_token_ids = list(range(start, start + count))
        if not set(self.action_token_ids).issubset(self.tokenizer.mergeable_ranks.values()):
            raise ValueError("FAST action IDs must be existing text tokens")
        if len(self.tokenizer) != model_cfg.vocab_size:
            raise ValueError("U0 model and tokenizer vocabulary sizes differ")
        self._ACTION_TOKEN_MIN, self._ACTION_TOKEN_MAX = start, self.action_token_ids[-1]
        self.model, loading = UNISForCausalLM.from_pretrained(
            cfg.base_vlm, config=model_cfg, torch_dtype=torch.bfloat16,
            attn_implementation=backend, low_cpu_mem_usage=True,
            local_files_only=True, output_loading_info=True,
        )
        if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"U0 pretrained weights do not match the model: {loading}")
        self.model.requires_grad_(True)
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        self.model.config._pre_quantization_dtype = torch.bfloat16
        self.vision_tokenizer_path = str(cfg.vision_tokenizer)
        vq_model = None
        if not cfg.get("use_cached_vision", False):
            vq_model = build_vision_tokenizer("ibq", cfg.vision_tokenizer, device="cpu")
            vq_model.requires_grad_(False).eval()
        # IBQ is an inference-only input processor, not a distributed parameter.
        # ZeRO-3 otherwise mixes its FP32 weights with BF16 LM all-gather buffers.
        object.__setattr__(self, "vq_model", vq_model)

    def train(self, mode=True):
        super().train(mode)
        if self.vq_model is not None:
            self.vq_model.eval()
        return self

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Follow device moves, but never round-trip IBQ weights through BF16:
        # that would change image IDs between training and deployment.
        if self.vq_model is None:
            return self
        parameter = next(self.vq_model.parameters())
        target = fn(torch.empty(0, device=parameter.device, dtype=parameter.dtype))
        self.vq_model.to(device=target.device)
        return self

    def action_loss(self, inputs, future_image_weight=None):
        labels = inputs["labels"]
        hidden = self.model.get_decoder()(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            use_cache=False, return_dict=True,
        ).last_hidden_state
        return action_cross_entropy(hidden, labels, self.model.get_output_embeddings(), self.loss_chunk_tokens, future_image_weight)

    def build_prefix_inputs(self, images, instructions, image_tokens=None):
        """Build U0 observation-only inputs for non-autoregressive action heads."""
        if not isinstance(images, list) or not isinstance(instructions, list):
            raise TypeError("images and instructions must be lists")
        if len(images) != len(instructions) or not images:
            raise ValueError("images and instructions must be nonempty lists of equal length")
        if image_tokens is not None and len(image_tokens) != len(images):
            raise ValueError("Cached image batch size mismatch")

        rows = []
        for index, (sample_images, instruction) in enumerate(zip(images, instructions)):
            if not isinstance(instruction, str):
                raise TypeError("U0 instructions must be strings")
            prompt = self.build_prompt_ids(
                sample_images,
                instruction,
                None if image_tokens is None else image_tokens[index],
            )
            if len(prompt) > self.max_length:
                raise ValueError(f"U0 prefix length {len(prompt)} exceeds max_length={self.max_length}")
            rows.append(prompt)

        width = max(map(len, rows))
        input_ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, :len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, :len(row)] = 1
        return {
            "input_ids": input_ids.to(self.input_device),
            "attention_mask": attention_mask.to(self.input_device),
        }

    def encode_prefix_hidden(self, inputs):
        """Encode an observation prefix and return all decoder hidden states."""
        if set(inputs) != {"input_ids", "attention_mask"}:
            raise ValueError("U0 prefix inputs must contain only input_ids and attention_mask")
        return self.model.get_decoder()(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    @torch.no_grad()
    def encode_image_codes(self, image):
        if self.vq_model is None:
            vq = build_vision_tokenizer("ibq", self.vision_tokenizer_path, device=self.input_device)
            object.__setattr__(self, "vq_model", vq.requires_grad_(False).eval())
        with torch.autocast(device_type=next(self.vq_model.parameters()).device.type, enabled=False):
            return super().encode_image_codes(image)

    @torch.no_grad()
    def encode_image(self, image):
        return self.format_image_codes(self.encode_image_codes(image))

    @torch.inference_mode()
    def generate_fast_tokens(self, images, instructions, action_constraint=None, prompt_ids=None):
        if not images or len(images) != len(instructions):
            raise ValueError("Expected matching nonempty images/instructions batches")
        ess = self.tokenizer.convert_tokens_to_ids(self.tokenizer.ess_token)
        allowed = self.action_token_ids + [ess]
        results = []
        for index, (views, instruction) in enumerate(zip(images, instructions)):
            prompt = self.build_prompt_ids(views, instruction) if prompt_ids is None else prompt_ids[index]
            if len(prompt) + self.max_new_tokens > self.max_length:
                raise ValueError("U0 generation prompt plus token budget exceeds max_length")
            ids = torch.tensor([prompt], device=self.input_device)
            output = self.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
                do_sample=False, max_new_tokens=self.max_new_tokens,
                eos_token_id=ess, pad_token_id=self.tokenizer.pad_token_id,
                prefix_allowed_tokens_fn=lambda _batch, _ids: (
                    action_constraint(_ids[len(prompt):].tolist()) if action_constraint else allowed
                ),
            )[0, len(prompt):].tolist()
            if len(output) < 2 or output[-1] != ess or any(t not in self.action_token_ids for t in output[:-1]):
                raise RuntimeError(f"U0 generated an empty, invalid or unterminated FAST sequence: {output}")
            results.append([t - self._ACTION_TOKEN_MIN for t in output[:-1]])
        return results
