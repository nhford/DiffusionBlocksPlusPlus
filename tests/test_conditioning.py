import pytest
import torch

from diffusion_blocks_plus_plus.llama.model import (
    AdaRMS,
    LlamaDiTBlockForCausalLM,
    LlamaDiTConfig,
)


@pytest.mark.parametrize("bias", [True, False])
def test_adarms_uniform_matches_previous_behavior(bias):
    norm = AdaRMS(4, 8, eps=1e-6, bias=bias)
    embeddings = torch.randn(2, 6, 8)
    conditioning = torch.randn(2, 4)
    params = norm.silu(norm.linear(conditioning))
    scale, shift = params.chunk(2, -1) if bias else (params, torch.zeros_like(params))
    rms = (embeddings.square().mean(-1, keepdim=True) + norm.eps).sqrt()
    expected = embeddings / rms * (scale[:, None, :] + 1) + shift[:, None, :]
    torch.testing.assert_close(norm(embeddings, conditioning[:, None, :]), expected)
    torch.testing.assert_close(
        norm(embeddings, conditioning[:, None, :].expand(-1, 6, -1)), expected
    )


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4), (1, 6, 4)])
def test_adarms_rejects_invalid_conditioning_shape(shape):
    with pytest.raises(ValueError, match="Conditioning must have shape"):
        AdaRMS(4, 8, eps=1e-6)(torch.randn(2, 6, 8), torch.randn(*shape))


@pytest.mark.parametrize("sequence_length", [1, 6])
def test_conditioning_reaches_all_normalizations(sequence_length):
    config = LlamaDiTConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_blocks=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        cond_hidden_size=4,
        use_cache=False,
        attn_implementation="eager",
    )
    model = LlamaDiTBlockForCausalLM(config)
    timesteps = torch.randn(2, sequence_length)
    expected = model.model.time_embedder(timesteps)
    seen = []

    def check_conditioning(module, args):
        torch.testing.assert_close(args[1], expected)
        seen.append(module)

    norms = [module for module in model.modules() if isinstance(module, AdaRMS)]
    handles = [norm.register_forward_pre_hook(check_conditioning) for norm in norms]
    try:
        output = model(inputs_embeds=torch.randn(2, 6, 16), timestep=timesteps)
        assert output.logits.shape == (2, 6, 16)
        assert len(norms) == 2 + 2 * len(model.model.layers)
        assert len(seen) == len(norms)
        assert set(seen) == set(norms)
        output.logits.square().mean().backward()
        for norm in norms:
            assert norm.linear.weight.grad is not None
            assert torch.isfinite(norm.linear.weight.grad).all()
    finally:
        for handle in handles:
            handle.remove()
