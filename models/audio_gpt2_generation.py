import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from models.fusion.fusion_block import AudioLLMFusionBlock


_FUSION_INDICES = {2, 5, 8, 11}


class _LoRALayer(nn.Module):
    def __init__(self, base, rank, alpha=16):
        super().__init__()
        self.base = base
        in_dim = base.weight.shape[0]  # GPT-2 Conv1D stores weight as (in, out)
        out_dim = base.weight.shape[1]
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + ((x @ self.A.T) @ self.B.T) * self.scale


def _inject_lora(gpt2_transformer, rank, alpha=16):
    for block in gpt2_transformer.h:
        block.attn.c_attn = _LoRALayer(block.attn.c_attn, rank, alpha)
        block.attn.c_proj = _LoRALayer(block.attn.c_proj, rank, alpha)


class AudioGPT2Generation(nn.Module):
    def __init__(
        self,
        audio_dim=768,
        adapter_dim=64,
        dropout=0.3,
        lora_rank=0,
    ):
        super().__init__()

        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

        for param in self.gpt2.parameters():
            param.requires_grad = False

        if lora_rank > 0:
            _inject_lora(self.gpt2.transformer, lora_rank)

        self.fusion_indices = _FUSION_INDICES

        self.fusion_blocks = nn.ModuleList([
            AudioLLMFusionBlock(
                text_dim=768,
                audio_dim=audio_dim,
                adapter_dim=adapter_dim,
                dropout=dropout,
            )
            for _ in range(len(_FUSION_INDICES))
        ])

        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("AudioGPT2Generation parameter summary:")
        print(f"  Frozen:    {frozen:,}")
        print(f"  Trainable: {trainable:,}")

    def forward(self, input_ids, audio_hidden):
        position_ids = torch.arange(
            input_ids.size(1),
            device=input_ids.device,
        ).unsqueeze(0)

        hidden = (
            self.gpt2.transformer.wte(input_ids)
            + self.gpt2.transformer.wpe(position_ids)
        )

        fusion_iter = iter(self.fusion_blocks)

        for i, gpt2_block in enumerate(self.gpt2.transformer.h):
            block_out = gpt2_block(hidden)
            hidden = block_out if isinstance(block_out, torch.Tensor) else block_out[0]

            if i in self.fusion_indices:
                hidden, _ = next(fusion_iter)(hidden, audio_hidden)

        hidden = self.gpt2.transformer.ln_f(hidden)
        logits = self.gpt2.lm_head(hidden)

        return logits
