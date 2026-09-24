from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from starVLA.model.framework.VLM4A.EmuFast import EmuFast
from starVLA.model.modules.vlm.Emu3_5 import Emu3_5Interface
from starVLA.model.modules.vlm.emu3_5 import Emu3Config, Emu3ForCausalLM
from starVLA.model.modules.vlm.emu3_5.modeling_emu3 import _get_usable_cache_length


def test_text_action_mapping_and_checkpoint_restore(tmp_path):
    tokenizer_dir = Path("starVLA/model/modules/vlm/emu3_5/tokenizer_emu3_ibq")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        special_tokens_file=tokenizer_dir / "emu3_vision_tokens.txt",
        trust_remote_code=True,
    )
    # Use the real vocabulary and PEFT layers with a tiny CPU transformer.
    config = Emu3Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                       attn_implementation="eager")
    base = Emu3ForCausalLM(config).state_dict()

    def load_base(*args, **kwargs):
        model = Emu3ForCausalLM(config)
        model.load_state_dict(base)
        return model

    def action_model(_config):
        model = torch.nn.Module()
        model.fast_tokenizer = SimpleNamespace()
        return model

    cfg = OmegaConf.create({"framework": {"emu": {
        "gradient_checkpointing": False, "attn_implementation": "eager",
        "lora": {"r": 2, "alpha": 4, "dropout": 0.0},
    }}})
    with patch("starVLA.model.modules.vlm.Emu3_5.Emu3Config.from_pretrained", return_value=config), \
         patch("starVLA.model.modules.vlm.Emu3_5.Emu3ForCausalLM.from_pretrained", side_effect=load_base), \
         patch("starVLA.model.modules.vlm.Emu3_5.build_vision_tokenizer", return_value=torch.nn.Linear(1, 1)), \
         patch("starVLA.model.framework.VLM4A.EmuFast.get_action_model", side_effect=action_model), \
         patch.object(Emu3ForCausalLM, "resize_token_embeddings", side_effect=AssertionError("must not resize")):
        model = EmuFast(cfg).eval()
        interface = model.emu_vl_interface
        assert interface.action_token_ids == list(range(149595, 151643))
        assert len(interface.tokenizer) == 282926
        assert not set(interface.action_token_ids).intersection(tokenizer.special_tokens.values())
        inputs = interface.build_training_inputs([[]], ["open the middle drawer"], [[0, 17, 2047]])
        labels = inputs["labels"][0]
        assert labels[labels != -100].tolist() == [149595, 149612, 151642, 151747, 151850]
        with pytest.raises(ValueError, match="reserved for FAST"):
            interface.build_prompt_ids([], tokenizer.decoder[149614].decode())

        prompt = interface.build_prompt_ids([], "open the middle drawer")
        output = torch.tensor([prompt + [149595, 149612, 151642, 151747]])
        with patch.object(interface.model, "generate", return_value=output) as generate:
            assert interface.generate_fast_tokens([[]], ["open the middle drawer"]) == [[0, 17, 2047]]
            assert generate.call_args.kwargs["prefix_allowed_tokens_fn"](0, None) == list(range(149595, 151643)) + [151747]

        # Emulate learned adapters, then restore into a freshly initialized adapter.
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.add_(0.01)
        with torch.no_grad():
            before = interface.model(**inputs).logits
        path = tmp_path / "adapter.safetensors"
        save_file(model.checkpoint_state_dict(model.state_dict()), path)
        saved = load_file(path)
        assert saved["action_token_ids"].tolist() == list(range(149595, 151643))
        restored = EmuFast(cfg).eval()
        restored.load_checkpoint_state_dict(saved)
        with torch.no_grad():
            after = restored.emu_vl_interface.model(**inputs).logits
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="mapping"):
            restored.load_checkpoint_state_dict({k: v for k, v in saved.items() if k != "action_token_ids"})
        with pytest.raises(RuntimeError, match="mapping"):
            restored.load_checkpoint_state_dict({**saved, "action_token_ids": saved["action_token_ids"] + 1})


def test_emu_cache_compatibility_with_current_transformers():
    assert _get_usable_cache_length(DynamicCache(), new_seq_length=1) == 0
