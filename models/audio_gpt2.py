import torch
import torch.nn as nn
from transformers import GPT2Model

from models.fusion.fusion_block import AudioLLMFusionBlock

# Fusion is applied at every 3rd GPT-2 layer (indices 2, 5, 8, 11) to reduce trainable
# parameter count while still conditioning all depth levels on audio.
_FUSION_INDICES = {2, 5, 8, 11}


class AudioGPT2(nn.Module):
    def __init__(self, num_classes=7, adapter_dim=64, dropout=0.3):
        super().__init__()

        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        for param in self.gpt2.parameters():
            param.requires_grad = False

        self.fusion_indices = _FUSION_INDICES
        self.fusion_blocks = nn.ModuleList([
            AudioLLMFusionBlock(
                text_dim=768,
                audio_dim=768,
                adapter_dim=adapter_dim,
                dropout=dropout,
            )
            for _ in range(len(_FUSION_INDICES))
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        frozen = sum(p.numel() for p in self.gpt2.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("AudioGPT2 parameter summary:")
        print(f"  Frozen   (GPT-2):              {frozen:,}")
        print(f"  Trainable (fusion + classifier): {trainable:,}")

    def forward(self, input_ids, audio_hidden):
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        hidden = self.gpt2.wte(input_ids) + self.gpt2.wpe(position_ids)

        fusion_iter = iter(self.fusion_blocks)
        for i, gpt2_block in enumerate(self.gpt2.h):
            block_out = gpt2_block(hidden)
            hidden = block_out if isinstance(block_out, torch.Tensor) else block_out[0]
            if i in self.fusion_indices:
                hidden, _ = next(fusion_iter)(hidden, audio_hidden)

        pooled = hidden[:, -1, :]
        return self.classifier(pooled)
