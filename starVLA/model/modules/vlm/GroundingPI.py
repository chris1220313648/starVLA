"""GroundingPI's native multimodal packing behind the StarVLA VLM interface."""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature


def expand_image_tokens(prompt, counts, image_token="<|image_pad|>"):
    parts = prompt.split(image_token)
    if len(parts) != len(counts) + 1 or any(n <= 0 for n in counts):
        raise ValueError("GroundingPI image placeholders do not match image grids")
    return parts[0] + "".join(image_token * n + tail for n, tail in zip(counts, parts[1:]))


class GroundingPIInterface(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = config.framework.qwenvl
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.base_vlm, trust_remote_code=True, dtype=torch.bfloat16,
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
        )
        self.processor = AutoProcessor.from_pretrained(cfg.base_vlm, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        self.model.config.hidden_size = self.model.config.text_config.hidden_size
        self.model.config.use_cache = False
        if cfg.get("enable_gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.action_token = "<robot_action_query>"
        name = config.framework.name
        fast_supervision = name == "GroundingPIFast" or bool(config.framework.get("knowledge_insulation", False))
        tokens = []
        if name == "GroundingPIOFT":
            tokens = [self.action_token]
        elif fast_supervision:
            tokens = [f"<robot_action_{i}>" for i in range(cfg.get("action_vocab_size", 2048))]
        tokenizer = self.processor.tokenizer
        if tokens:
            tokenizer.add_tokens(tokens, special_tokens=True)
            self.model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            if any(tokenizer.encode(token, add_special_tokens=False) != [idx] for token, idx in zip(tokens, ids)):
                raise ValueError("Action tokens must each encode to one token")
            if fast_supervision:
                if ids != list(range(ids[0], ids[0] + len(ids))):
                    raise ValueError("FAST action tokens must form a contiguous ordered vocabulary")
                self._ACTION_TOKEN_MIN, self._ACTION_TOKEN_MAX = ids[0], ids[-1]

    def forward(self, **kwargs):
        if kwargs.get("labels") is None:
            kwargs.setdefault("logits_to_keep", 1)
        return self.model(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def build_qwenvl_inputs(self, images, instructions, solutions=None, allow_missing_solutions=False, **kwargs):
        if not images or len(images) != len(instructions) or (solutions is not None and len(solutions) != len(images)):
            raise ValueError("Images, instructions and solutions must batch-align")
        flat_images = [image for views in images for image in views]
        image_inputs = self.processor.image_processor(images=flat_images, return_tensors="pt") if flat_images else {}
        merge = self.processor.image_processor.merge_size ** 2
        sizes = [int(grid.prod()) for grid in image_inputs.get("image_grid_thw", [])]
        if any(n % merge for n in sizes):
            raise ValueError("Image grid must be divisible by merge_size squared")
        counts = [n // merge for n in sizes]
        texts, prefix_lengths = [], []
        offset = 0
        tokenizer = self.processor.tokenizer
        for i, (views, instruction) in enumerate(zip(images, instructions)):
            template = self.config.datasets.vla_data.get("CoT_prompt", "{instruction}")
            content = [{"type": "image", "image": "provided"} for _ in views]
            content.append({"type": "text", "text": template.replace("{instruction}", instruction)})
            messages = [{"role": "user", "content": content}]
            render_kwargs = dict(tokenize=False, enable_thinking=False)
            prefix = self.processor.apply_chat_template(messages, add_generation_prompt=True, **render_kwargs)
            view_counts = counts[offset:offset + len(views)]
            offset += len(views)
            prefix = expand_image_tokens(prefix, view_counts)
            if solutions is None:
                texts.append(prefix)
                continue
            if solutions[i] is None:
                if not allow_missing_solutions:
                    raise ValueError("Missing assistant solution")
                prefix_lengths.append(len(tokenizer.encode(prefix, add_special_tokens=False)))
                texts.append(prefix)
                continue
            if not isinstance(solutions[i], str) or not solutions[i].strip():
                raise ValueError("Empty assistant solution")
            messages.append({"role": "assistant", "content": [{"type": "text", "text": solutions[i]}]})
            full = self.processor.apply_chat_template(messages, add_generation_prompt=False, **render_kwargs)
            full = expand_image_tokens(full, view_counts)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            full_ids = tokenizer.encode(full, add_special_tokens=False)
            if full_ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError("Assistant training prefix does not match generation prefix")
            prefix_lengths.append(len(prefix_ids))
            texts.append(full)
        if offset != len(counts):
            raise ValueError("Unused processed image grids")
        inputs = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
        if solutions is not None:
            labels = inputs["input_ids"].clone()
            prefix_mask = inputs["attention_mask"].bool().clone()
            for i, prefix_len in enumerate(prefix_lengths):
                pad_len = labels.shape[1] - int(inputs["attention_mask"][i].sum())
                labels[i, :pad_len + prefix_len] = -100
                prefix_mask[i, pad_len + prefix_len:] = False
            labels[inputs["attention_mask"] == 0] = -100
            for i, solution in enumerate(solutions):
                if solution is not None and not labels[i].ne(-100).any():
                    raise ValueError("Empty assistant supervision")
            inputs["labels"] = labels
            if allow_missing_solutions:
                inputs["prefix_mask"] = prefix_mask
        return BatchFeature(data={**inputs, **image_inputs}).to(self.model.device)
