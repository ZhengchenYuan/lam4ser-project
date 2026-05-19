import torch
import torch.nn as nn


class CrossAttentionAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        num_heads=8,
        adapter_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_hidden, audio_hidden, audio_mask=None):

        attn_out, attn_weights = self.cross_attn(
            query=text_hidden,
            key=audio_hidden,
            value=audio_hidden,
            key_padding_mask=audio_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        fused_hidden = self.norm(
            text_hidden + self.adapter(attn_out)
        )

        return fused_hidden, attn_weights


if __name__ == "__main__":

    B = 2
    T_text = 32
    T_audio = 50
    d = 768

    text_hidden = torch.randn(B, T_text, d)
    audio_hidden = torch.randn(B, T_audio, d)

    module = CrossAttentionAdapter(
        hidden_dim=d,
        num_heads=8,
    )

    fused_hidden, attn_weights = module(
        text_hidden,
        audio_hidden,
    )

    print("text_hidden:", text_hidden.shape)
    print("audio_hidden:", audio_hidden.shape)
    print("fused_hidden:", fused_hidden.shape)
    print("attn_weights:", attn_weights.shape)
