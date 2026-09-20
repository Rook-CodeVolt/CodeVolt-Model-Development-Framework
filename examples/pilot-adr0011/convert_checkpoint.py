#!/usr/bin/env python3
"""Convert MiniMind raw state-dict checkpoint to a standard HF checkpoint dir.

Bug fix vs prior attempt: instead of loading MiniMind's own
MiniMindForCausalLM custom-code class (which requires copying
model_minimind.py into the output dir via transformers'
custom_object_save/register_for_auto_class path -- and crashes with
PermissionError when the *source* checkout the custom code is read from
is chmod a-w, as our pinned MiniMind checkout deliberately is), this
script observes that MiniMind's minimind2-small architecture (RMSNorm,
SiLU MLP, q_norm/k_norm, RoPE theta=1e6, tied embeddings) is
parameter-for-parameter identical in shape and naming to HF's *standard*
Qwen3 architecture (transformers.Qwen3ForCausalLM/Qwen3Config), which
ships in transformers itself -- no custom code, no
register_for_auto_class, no copying any .py file anywhere. This avoids
the custom-code-copy path entirely, which is simpler and sufficient for
HFLocalCausalLMEvaluatorAdapter (it only needs a standard
AutoModelForCausalLM-loadable directory: config.json + weights +
tokenizer, loaded without trust_remote_code).

Verified: raw MiniMind state_dict key names already exactly match
Qwen3ForCausalLM's own state_dict key names (model.embed_tokens.weight,
model.layers.N.self_attn.{q,k,v,o}_proj.weight,
model.layers.N.self_attn.{q,k}_norm.weight,
model.layers.N.mlp.{gate,up,down}_proj.weight,
model.layers.N.{input,post_attention}_layernorm.weight, model.norm.weight,
lm_head.weight) -- so this is a direct load_state_dict, not a manual
key-renaming exercise.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAW_CHECKPOINT = Path(
    "./local-evidence/pilot-runs/adr0011-minimind-20260920/scratch/trainer_work/"
    "adr0011-pilot-20260920/final/full_sft_512.pth"
)
MINIMIND_CHECKOUT = Path("./local-evidence/pilot-runs/adr0011-minimind-20260920/minimind")
TOKENIZER_DIR = MINIMIND_CHECKOUT / "model"


def convert(output_dir: Path) -> None:
    import torch
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    output_dir.mkdir(parents=True, exist_ok=True)

    # minimind2-small reference config, matching ADR-0011's locked
    # MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)
    # defaults (see model_minimind.py's MiniMindConfig.__init__ defaults).
    config = Qwen3Config(
        vocab_size=6400,
        hidden_size=512,
        intermediate_size=1664,  # actual raw ckpt mlp.gate_proj shape, not the
        # math.ceil(512*pi/64)*64=1664 formula default -- verified identical, both 1664
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,  # corrected: raw k/v_proj.weight shape is
        # (256, 512) = 4 * 64, not (128, 512) = 2 * 64 as previously assumed
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        attention_bias=False,
        attention_dropout=0.0,
        torch_dtype="float32",
    )

    raw_state_dict = torch.load(RAW_CHECKPOINT, map_location="cpu")
    raw_state_dict = {k: v.to(torch.float32) for k, v in raw_state_dict.items()}

    model = Qwen3ForCausalLM(config)
    # tied embeddings: lm_head.weight is present in the raw checkpoint (as
    # MiniMind always writes it) but strict-loading a tied model that
    # aliases lm_head.weight to model.embed_tokens.weight is fine because
    # both are the same tensor content (tied at training time too).
    missing, unexpected = model.load_state_dict(raw_state_dict, strict=False)
    allowed_missing = {"lm_head.weight"}  # tied, gets bound to embed_tokens post-load
    real_missing = [m for m in missing if m not in allowed_missing]
    if real_missing or unexpected:
        raise SystemExit(
            f"FATAL: state_dict load mismatch. missing={real_missing} "
            f"unexpected={unexpected}"
        )
    model.tie_weights()
    model.eval()

    # Sanity check: does the model actually reproduce the raw checkpoint's
    # own lm_head weight (tied embedding correctness), not just silently
    # accept the mismatch above.
    assert torch.allclose(
        model.lm_head.weight, raw_state_dict["lm_head.weight"], atol=1e-6
    ), "tied lm_head.weight does not match raw checkpoint's own lm_head.weight"

    model.save_pretrained(str(output_dir), safe_serialization=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Converted checkpoint written to {output_dir}")
    print(sorted(p.name for p in output_dir.iterdir()))


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./converted_model")
    convert(out)
