import torch

from models.fusion.fusion_block import AudioLLMFusionBlock


def test_fusion_block_same_dim():
    B, T_text, T_audio, d = 2, 32, 50, 768

    text_hidden = torch.randn(B, T_text, d)
    audio_hidden = torch.randn(B, T_audio, d)

    block = AudioLLMFusionBlock(
        text_dim=d,
        audio_dim=d,
        num_heads=8,
    )

    fused_hidden, attn_weights = block(text_hidden, audio_hidden)

    assert fused_hidden.shape == (B, T_text, d)
    assert attn_weights.shape == (B, T_text, T_audio)


def test_fusion_block_with_projection():
    B, T_text, T_audio = 2, 32, 50
    text_dim = 768
    audio_dim = 1024

    text_hidden = torch.randn(B, T_text, text_dim)
    audio_hidden = torch.randn(B, T_audio, audio_dim)

    block = AudioLLMFusionBlock(
        text_dim=text_dim,
        audio_dim=audio_dim,
        num_heads=8,
    )

    fused_hidden, attn_weights = block(text_hidden, audio_hidden)

    assert fused_hidden.shape == (B, T_text, text_dim)
    assert attn_weights.shape == (B, T_text, T_audio)
