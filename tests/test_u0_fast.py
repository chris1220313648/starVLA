from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file, save_file

from starVLA.model.framework.VLM4A.U0Fast import U0Fast
from starVLA.model.modules.vlm.U0 import U0Interface, action_cross_entropy, fast_token_constraint
from starVLA.model.modules.vlm.u0.configuration_unis import UNISConfig
from starVLA.model.modules.vlm.u0.modeling_unis import UNISForCausalLM


def test_chunked_loss_and_gradients():
    torch.manual_seed(1)
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    head = torch.nn.Linear(8, 13, bias=False)
    labels = torch.tensor([[-100, -100, 3, 4, 5, -100, -100], [-100, 1, 2, 3, 4, 5, 6]])
    expected = torch.nn.functional.cross_entropy(
        head(hidden[:, :-1]).reshape(-1, 13), labels[:, 1:].reshape(-1), ignore_index=-100,
    )
    actual = action_cross_entropy(hidden, labels, head, 2)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual, (hidden, head.weight), retain_graph=True)
    ge = torch.autograd.grad(expected, (hidden, head.weight))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)
    with pytest.raises(ValueError, match="no supervised"):
        action_cross_entropy(hidden, torch.full_like(labels, -100), head)


def tiny_config():
    # Independent head_dim is essential: U0-4B does not use hidden_size / heads.
    return UNISConfig(vocab_size=282926, hidden_size=16, intermediate_size=32,
                      head_dim=8, num_attention_heads=4, num_key_value_heads=2,
                      num_hidden_layers=1, attention_dropout=0, attn_implementation="eager")


def test_full_training_packing_generation_and_restore(tmp_path):
    vocab = Path("starVLA/model/modules/vlm/emu3_5/tokenizer_emu3_ibq").resolve()
    (tmp_path / "unis.tiktoken").symlink_to(vocab / "emu3.tiktoken")
    (tmp_path / "unis_vision_tokens.txt").symlink_to(vocab / "emu3_vision_tokens.txt")
    cfg = OmegaConf.create({"framework": {"u0": {
        "gradient_checkpointing": False, "attn_implementation": "eager",
        "tokenizer_path": str(tmp_path),
    }}})
    config = tiny_config()

    def load(*args, **kwargs):
        return UNISForCausalLM(config), {}

    def action_model(_cfg):
        module = torch.nn.Module()
        module.fast_tokenizer = SimpleNamespace()
        return module

    with patch("starVLA.model.modules.vlm.U0.UNISConfig.from_pretrained", return_value=config), \
         patch("starVLA.model.modules.vlm.U0.UNISForCausalLM.from_pretrained", side_effect=load), \
         patch("starVLA.model.modules.vlm.U0.build_vision_tokenizer", return_value=torch.nn.Linear(1, 1)), \
         patch("starVLA.model.framework.VLM4A.U0Fast.get_action_model", side_effect=action_model):
        model = U0Fast(cfg).train()
        interface = model.u0_interface
        assert all(p.requires_grad for p in interface.model.parameters())
        assert not any(p.requires_grad for p in interface.vq_model.parameters())
        assert not interface.vq_model.training
        assert not any("vq_model" in name for name, _ in model.named_parameters())
        assert interface.model.model.layers[0].self_attn.q_proj.weight.shape == (32, 16)
        assert interface.action_token_ids == list(range(149595, 151643))
        inputs = interface.build_training_inputs([[], []], ["open drawer", "move cup"], [[0, 17, 2047], [3]])
        with patch.object(interface, "encode_image", return_value=[1, 2, 3]) as encode:
            online = interface.build_training_inputs([[object()], [object()]], ["open", "move"], [[0], [3]])
            assert encode.call_count == 2
        with patch.object(interface, "encode_image", side_effect=AssertionError("online encoding called")):
            cached = interface.build_training_inputs([[], []], ["open", "move"], [[0], [3]],
                                                     image_tokens=[[[1, 2, 3]], [[1, 2, 3]]])
        for key in online:
            torch.testing.assert_close(online[key], cached[key], rtol=0, atol=0)
        from starVLA.dataloader.u0_vision_cache import extract_codebook_grids
        codes = np.arange(196, dtype=np.int32).reshape(14, 14)
        formatted = interface.format_image_codes(codes)
        restored_codes = extract_codebook_grids(np.asarray([[formatted]], dtype=np.int32), interface.tokenizer, (14, 14))
        np.testing.assert_array_equal(restored_codes[0, 0], codes)
        assert interface.format_image_codes(restored_codes[0, 0]) == formatted
        corrupt = np.asarray([[formatted]], dtype=np.int32)
        corrupt[0, 0, 0] = 0
        with pytest.raises(ValueError, match="Malformed"):
            extract_codebook_grids(corrupt, interface.tokenizer, (14, 14))
        assert inputs["labels"][0][inputs["labels"][0] != -100].tolist() == [149595, 149612, 151642, 151747, 151850]
        assert (inputs["labels"][inputs["attention_mask"] == 0] == -100).all()
        with pytest.raises(ValueError, match="reserved for FAST"):
            interface.build_prompt_ids([], interface.tokenizer.decoder[149614].decode())
        before = interface.model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        optimizer = torch.optim.SGD(interface.model.parameters(), lr=0.01)
        loss = interface.action_loss(inputs)
        loss.backward()
        assert torch.isfinite(loss)
        assert interface.model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
        optimizer.step()
        assert not torch.equal(before, interface.model.model.layers[0].self_attn.q_proj.weight)
        model.eval()
        with torch.no_grad():
            expected = interface.model(**inputs).logits
        saved = model.checkpoint_state_dict(model.state_dict())
        assert not any("vq_model" in k or "lora" in k for k in saved)
        save_file(saved, tmp_path / "model.safetensors")
        restored = U0Fast(cfg).eval()
        restored.load_checkpoint_state_dict(load_file(tmp_path / "model.safetensors"))
        with torch.no_grad():
            torch.testing.assert_close(expected, restored.u0_interface.model(**inputs).logits, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="mapping"):
            restored.load_checkpoint_state_dict({**saved, "action_token_ids": saved["action_token_ids"] + 1})
        with pytest.raises(RuntimeError, match="Missing key"):
            restored.load_checkpoint_state_dict({k: v for k, v in saved.items() if k != "u0_interface.model.lm_head.weight"})
        prompt = interface.build_prompt_ids([], "open drawer")
        with patch.object(interface.model, "generate", return_value=torch.tensor([prompt + [149595, 149612, 151747]])):
            assert interface.generate_fast_tokens([[]], ["open drawer"]) == [[0, 17]]
        with patch.object(interface.model, "generate", return_value=torch.tensor([prompt + [149595]])):
            with pytest.raises(RuntimeError, match="unterminated"):
                interface.generate_fast_tokens([[]], ["open drawer"])
        with pytest.raises(ValueError, match="action shape"):
            model([{"image": [Image.new("RGB", (16, 16))], "lang": "open", "action": np.zeros((8, 14))}])
        example = {"image": [Image.new("RGB", (16, 16))], "lang": "open"}
        model.action_model.fast_tokenizer.bpe_tokenizer = SimpleNamespace(
            decode=lambda row: "x" * 56, convert_ids_to_tokens=lambda i: "x",
        )
        model.action_model.fast_tokenizer.decode = lambda rows: np.ones((len(rows), 8, 7))
        # Cached training batches must reach generation with the same prompt as raw views.
        cached_examples = [{"image": [], "image_codes": [codes, codes], "lang": lang}
                           for lang in ("open", "move")]
        with patch.object(interface, "encode_image", return_value=formatted):
            expected_prompts = [interface.build_prompt_ids([object(), object()], e["lang"])
                                for e in cached_examples]
        def generate_cached(input_ids, **kwargs):
            assert input_ids[0].tolist() == expected_prompts[len(seen)]
            seen.append(input_ids[0].tolist())
            return torch.cat([input_ids, input_ids.new_tensor([[149595, 149612, 151747]])], dim=1)
        seen = []
        with patch.object(interface, "encode_image", side_effect=AssertionError("online encoding called")), \
             patch.object(interface.model, "generate", side_effect=generate_cached):
            result = model.predict_action(cached_examples)["normalized_actions"]
        assert seen == expected_prompts
        np.testing.assert_array_equal(result, np.ones((2, 8, 7)))
        with pytest.raises(ValueError, match="cached prediction"):
            model.predict_action([cached_examples[0], example])
        with pytest.raises(ValueError, match="cached prediction"):
            model.predict_action({"image_codes": [], "lang": "open"})
        with patch.object(interface, "generate_fast_tokens", return_value=[[0, 1]]):
            assert model.predict_action(example)["normalized_actions"].shape == (1, 8, 7)
            model.action_model.fast_tokenizer.bpe_tokenizer.decode = lambda row: "x"
            with pytest.raises(RuntimeError, match="coefficients"):
                model.predict_action(example)
        vq_before = interface.vq_model.weight.detach().clone()
        model.bfloat16()
        assert interface.vq_model.weight.dtype == torch.float32
        torch.testing.assert_close(vq_before, interface.vq_model.weight, rtol=0, atol=0)
        cfg.framework.u0.use_cached_vision = True
        with patch("starVLA.model.modules.vlm.U0.build_vision_tokenizer", side_effect=AssertionError("IBQ loaded during cached training")):
            cached_model = U0Fast(cfg).train().bfloat16()
            assert cached_model.u0_interface.vq_model is None
            cached_model.action_model.encoder_action2fastoken = lambda actions: [[0] for _ in actions]
            result = cached_model([{"image_codes": np.zeros((1, 14, 14), dtype=np.int32), "lang": "open", "action": np.zeros((8, 7))}])
            assert torch.isfinite(result["action_loss"])
            result["action_loss"].backward()


def test_cached_generation_matches_full_prefix():
    torch.manual_seed(2)
    model = UNISForCausalLM(tiny_config()).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        first = model(ids[:, :3], use_cache=True)
        cached = model(ids[:, 3:], past_key_values=first.past_key_values, use_cache=True)
        full = model(ids, use_cache=False)
        torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], atol=1e-6, rtol=1e-5)
        output = model.generate(ids, use_cache=True, max_new_tokens=3, do_sample=False)
        assert output.shape == (1, 7)


def test_fast_constraint_handles_utf8_boundaries_and_exact_shape():
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    encode_byte = bytes_to_unicode()
    raw_tokens = [b"A", b"BC", b"\xc4", b"\xa2", b"XYZ"]
    tokens = ["".join(encode_byte[b] for b in raw) for raw in raw_tokens]
    bpe = SimpleNamespace(convert_ids_to_tokens=lambda i: tokens[i])
    allowed = fast_token_constraint(bpe, [10, 11, 12, 13, 14], 99, 2)
    assert allowed([]) == [10, 11, 12]
    assert allowed([12]) == [13]  # Complete a split UTF-8 coefficient first.
    assert allowed([11]) == [99]
    assert allowed([12, 13, 10]) == [99]
    assert 11 not in allowed([10])  # Two more coefficients would overflow.
