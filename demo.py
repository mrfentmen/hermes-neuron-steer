"""Offline demo for NeuronSteer — the full CNA flow with zero downloads.

Builds a tiny randomly-initialised Llama in memory and runs find ->
generate -> scan against it. The model is random, so its text is
gibberish and no behavior is being steered; what the demo proves is the
mechanics, end to end, in seconds: a circuit is found by contrast,
ablating it changes the output, amplifying it changes the output
differently, and every output gets the paper's quality score.

This exists so anyone can watch the plugin work the moment it is
installed, before pulling a multi-GB model. Behavior on a real model is
the smoke test's job (scripts/smoke_test.py) and is labelled as such.
"""

from __future__ import annotations

try:
    from . import engine
except ImportError:  # loaded as a flat module directory
    import engine

# Two readable, token-distinct sets. On a random model the content is
# arbitrary; the contrast is what exercises the machinery.
POSITIVE = [
    "the quick brown fox",
    "foxes are quick animals",
    "a quick fox jumps high",
]
NEGATIVE = [
    "stock prices fell today",
    "the market closed lower",
    "bonds and interest rates",
]
DEMO_PROMPT = "the quick brown fox"


class CharTokenizer:
    """Char-level tokenizer: all the engine needs is __call__ + decode."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __call__(self, text, return_tensors=None):
        torch = engine._torch()
        ids = [max(3, ord(c) % 256) for c in text] or [3]

        class Enc:
            pass

        enc = Enc()
        enc.input_ids = torch.tensor([ids], dtype=torch.long)
        return enc

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids if int(i) > 2)


def build_tiny_model(seed: int = 1234):
    """The same tiny Llama shape the unit tests use (4 layers x 128 MLP)."""
    torch = engine._torch()
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _gen(model, tok, circuit, multiplier, max_new_tokens):
    text = engine.generate(
        model, tok, DEMO_PROMPT,
        max_new_tokens=max_new_tokens,
        circuit=circuit, multiplier=multiplier,
        device="cpu", chat_template=False,
    )
    return {"text": text, "quality": round(engine.repetition_quality(text), 4)}


def run_demo(top_k: int = 16, max_new_tokens: int = 16) -> dict:
    """Run find -> generate -> scan on the tiny model; return a transcript."""
    model = build_tiny_model()
    tok = CharTokenizer()

    circuit = engine.find_circuit(
        model, tok, POSITIVE, NEGATIVE,
        model_id="neuron-steer-demo (tiny random Llama)",
        top_k=top_k, device="cpu", chat_template=False,
    )
    hist = circuit.layer_histogram()
    top_layers = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)[:5]

    baseline = _gen(model, tok, None, None, max_new_tokens)
    ablated = _gen(model, tok, circuit, 0.0, max_new_tokens)
    amplified = _gen(model, tok, circuit, 2.0, max_new_tokens)

    scan_rows = engine.scan(
        model, tok, circuit, POSITIVE,
        multipliers=[1.0, 0.0, 2.0],
        max_new_tokens=max_new_tokens,
        device="cpu", chat_template=False,
    )

    return {
        "demo": True,
        "about": (
            "Full CNA flow on a tiny randomly initialised Llama — no "
            "downloads, no behavior being steered (the weights are random, "
            "so the text is gibberish). It proves the mechanics end to "
            "end: contrast finds a circuit, ablating it changes the "
            "output, amplifying it changes the output differently, and "
            "every output carries the paper's quality score. For a real "
            "model, use neuron_steer_find."
        ),
        "model": {
            "architecture": "LlamaForCausalLM, random weights (seed 1234)",
            "total_mlp_neurons": engine.total_mlp_neurons(model),
        },
        "find": {
            "positive_prompts": POSITIVE,
            "negative_prompts": NEGATIVE,
            "circuit_size": len(circuit.neurons),
            "top_neurons": [n.to_dict() for n in circuit.neurons[:10]],
            "layers_with_most_circuit_neurons": [
                {"layer": layer, "count": count} for layer, count in top_layers
            ],
        },
        "generate": {
            "prompt": DEMO_PROMPT,
            "baseline": baseline,
            "ablated_x0": ablated,
            "amplified_x2": amplified,
        },
        "scan": [
            {
                "multiplier": row["multiplier"],
                "mean_quality": round(row["mean_quality"], 4),
            }
            for row in scan_rows
        ],
    }
