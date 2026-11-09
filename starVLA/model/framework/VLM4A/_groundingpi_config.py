"""Shared configuration defaults for GroundingPI action frameworks."""

from omegaconf import OmegaConf

from starVLA.model.framework.share_tools import apply_config_compat


def grounding_config(config, name):
    defaults = OmegaConf.create({
        "framework": {
            "name": name,
            "qwenvl": {
                "backbone": "groundingpi",
                "base_vlm": "playground/Pretrained_models/GroundingPI",
                "attn_implementation": "sdpa",
                "enable_gradient_checkpointing": True,
                "action_vocab_size": 2048,
            },
            "action_model": {"action_horizon": 16},
        },
        "datasets": {"vla_data": {}},
    })
    supplied = getattr(config, "_cfg", config) if config is not None else {}
    supplied = apply_config_compat(OmegaConf.create(OmegaConf.to_container(supplied) if OmegaConf.is_config(supplied) else supplied))
    cfg = OmegaConf.merge(defaults, supplied)
    cfg.framework.name = name
    cfg.framework.qwenvl.backbone = "groundingpi"
    cfg = apply_config_compat(cfg)
    if hasattr(config, "_cfg"):
        config.framework = cfg.framework
        if "datasets" not in config:
            config.datasets = cfg.datasets
        return config
    return cfg
