"""NeuronSteer engine — Contrastive Neuron Attribution (CNA) for Hermes Agent.

Implements the method from Nous Research, "Targeted Neuron Modulation via
Contrastive Pair Search" (arXiv:2605.12290): identify the sparse set of MLP
neurons whose activations most distinguish two prompt sets, then modulate
those neurons at inference time to steer behavior. Forward passes only —
no gradients, no auxiliary training, no weight modification.

Method, exactly as published:
  1. For every MLP layer, record the down-projection *input* activations
     (the post-activation hidden units) at the last token position, for
     every prompt in a positive set and a negative set.
  2. delta_j = mean(activation on positive) - mean(activation on negative).
  3. The circuit is the top-k neurons by |delta| across all layers
     (the paper uses k = 0.1% of total MLP neurons).
  4. Steering = multiplying each circuit neuron's activation by a scalar
     multiplier m at inference (m = 0 ablates, m = 1 is baseline,
     m > 1 amplifies). Causality in the paper is verified this way.

Output quality is scored with the paper's own metric: 1 minus the fraction
of repeated n-grams.

The engine is model-agnostic: any Hugging Face causal LM whose blocks
expose ``mlp.down_proj`` works (Llama, Qwen, Granite, Mistral families).
torch and transformers are imported lazily so the Hermes plugin shell can
load (and explain the missing dependency) without them installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_MODEL_CACHE: dict = {}


# --------------------------------------------------------------------------
# Model loading / architecture resolution
# --------------------------------------------------------------------------

def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only w/o torch
        raise RuntimeError(
            "NeuronSteer needs torch and transformers. Install with: "
            "pip install torch transformers"
        ) from exc
    return torch


def pick_device(requested: str | None = None) -> str:
    torch = _torch()
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, device: str | None = None):
    """Load (and cache) a causal LM + tokenizer by Hugging Face id."""
    torch = _torch()
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = pick_device(device)
    dtype = torch.float32 if dev == "cpu" else torch.float16
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:  # transformers < 5 used the torch_dtype kwarg
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(dev)
    model.eval()
    _MODEL_CACHE[model_id] = (model, tok)
    return model, tok


def mlp_modules(model) -> list:
    """Return the per-layer MLP modules of a causal LM.

    Resolves the common layouts (Llama/Qwen/Granite/Mistral all expose
    ``model.model.layers[i].mlp``) and verifies ``down_proj`` exists.
    """
    candidates = []
    base = getattr(model, "model", None)
    if base is not None:
        candidates.append(base)
    candidates.append(model)
    for root in candidates:
        layers = getattr(root, "layers", None)
        if layers is None:
            layers = getattr(root, "decoder", None)
            layers = getattr(layers, "layers", None) if layers is not None else None
        if layers is None:
            continue
        mlps = []
        ok = True
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "down_proj"):
                ok = False
                break
            mlps.append(mlp)
        if ok and mlps:
            return mlps
    raise ValueError(
        "Unsupported architecture: could not find per-layer mlp.down_proj "
        "modules. NeuronSteer supports Llama/Qwen/Granite/Mistral-style "
        "decoder stacks."
    )


def total_mlp_neurons(model) -> int:
    mlps = mlp_modules(model)
    return sum(mlp.down_proj.in_features for mlp in mlps)


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

def encode_prompts(tok, prompts: list[str], device: str, chat_template: bool = True):
    """Encode each prompt to a [1, T] LongTensor on device.

    When the tokenizer ships a chat template (instruct models), each prompt
    is wrapped as a single user turn with a generation prompt appended, so
    the last token is the position where the model is about to answer —
    the position whose activations decide the behavior being contrasted.
    Circuit finding and generation use the same encoding, so they agree.
    """
    torch = _torch()
    out = []
    use_chat = chat_template and getattr(tok, "chat_template", None)
    for prompt in prompts:
        if use_chat:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(ids, "input_ids"):  # BatchEncoding (transformers v5)
                ids = ids.input_ids
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            tensor = torch.tensor([ids], dtype=torch.long)
        else:
            tensor = tok(prompt, return_tensors="pt").input_ids
        out.append(tensor.to(device))
    return out


# --------------------------------------------------------------------------
# CNA circuit finding
# --------------------------------------------------------------------------

@dataclass
class NeuronRef:
    layer: int
    neuron: int
    delta: float

    def to_dict(self) -> dict:
        return {"layer": self.layer, "neuron": self.neuron, "delta": self.delta}

    @staticmethod
    def from_dict(d: dict) -> "NeuronRef":
        return NeuronRef(layer=int(d["layer"]), neuron=int(d["neuron"]), delta=float(d["delta"]))


@dataclass
class Circuit:
    model_id: str
    neurons: list[NeuronRef]
    positive_count: int = 0
    negative_count: int = 0

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "neurons": [n.to_dict() for n in self.neurons],
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }

    @staticmethod
    def from_dict(d: dict) -> "Circuit":
        return Circuit(
            model_id=d["model_id"],
            neurons=[NeuronRef.from_dict(n) for n in d["neurons"]],
            positive_count=int(d.get("positive_count", 0)),
            negative_count=int(d.get("negative_count", 0)),
        )

    def by_layer(self) -> dict[int, list[NeuronRef]]:
        grouped: dict[int, list[NeuronRef]] = {}
        for n in self.neurons:
            grouped.setdefault(n.layer, []).append(n)
        return grouped

    def layer_histogram(self) -> dict[int, int]:
        return {layer: len(ns) for layer, ns in sorted(self.by_layer().items())}


def _last_token_means(model, tok, prompts: list[str], device: str, chat_template: bool):
    """Mean last-token down-proj input activations per layer: [L, I] tensor."""
    torch = _torch()
    mlps = mlp_modules(model)
    captured: list = [None] * len(mlps)
    handles = []

    def make_hook(idx):
        def hook(module, inputs, output):
            captured[idx] = inputs[0][:, -1, :].detach().float().cpu()
        return hook

    for i, mlp in enumerate(mlps):
        handles.append(mlp.down_proj.register_forward_hook(make_hook(i)))
    try:
        sums = None
        for ids in encode_prompts(tok, prompts, device, chat_template):
            with torch.no_grad():
                model(input_ids=ids)
            vec = torch.cat(captured, dim=0)  # [L, I] (I uniform across layers in supported archs)
            sums = vec if sums is None else sums + vec
    finally:
        for h in handles:
            h.remove()
    return sums / max(1, len(prompts))


def find_circuit(
    model,
    tok,
    positive_prompts: list[str],
    negative_prompts: list[str],
    model_id: str = "",
    top_fraction: float = 0.001,
    top_k: int | None = None,
    device: str | None = None,
    chat_template: bool = True,
) -> Circuit:
    """Run CNA: contrast two prompt sets, return the top-|delta| circuit.

    top_fraction defaults to the paper's 0.1% of total MLP neurons.
    """
    torch = _torch()
    if not positive_prompts or not negative_prompts:
        raise ValueError("Both a positive and a negative prompt set are required.")
    dev = device or next(model.parameters()).device.type
    pos = _last_token_means(model, tok, positive_prompts, dev, chat_template)
    neg = _last_token_means(model, tok, negative_prompts, dev, chat_template)
    delta = pos - neg  # [L, I]
    layers, inter = delta.shape
    total = layers * inter
    k = int(top_k) if top_k else max(1, int(round(total * float(top_fraction))))
    k = min(k, total)
    flat = delta.abs().flatten()
    _, idx = torch.topk(flat, k)
    neurons = []
    for i in idx.tolist():
        layer, neuron = divmod(i, inter)
        neurons.append(NeuronRef(layer=layer, neuron=neuron, delta=float(delta[layer, neuron])))
    neurons.sort(key=lambda n: abs(n.delta), reverse=True)
    return Circuit(
        model_id=model_id,
        neurons=neurons,
        positive_count=len(positive_prompts),
        negative_count=len(negative_prompts),
    )


# --------------------------------------------------------------------------
# Steered generation
# --------------------------------------------------------------------------

class _Modulator:
    """Forward pre-hooks that scale selected down-proj input activations."""

    def __init__(self, model, circuit: Circuit, multiplier: float):
        torch = _torch()
        self.handles = []
        mlps = mlp_modules(model)
        for layer, refs in circuit.by_layer().items():
            if layer >= len(mlps):
                continue
            idx = torch.tensor([r.neuron for r in refs], dtype=torch.long)
            m = float(multiplier)

            def make_pre_hook(indices, mult):
                def pre_hook(module, inputs):
                    x = inputs[0]
                    idx_dev = indices.to(x.device)
                    new_x = x.clone()
                    new_x[..., idx_dev] = new_x[..., idx_dev] * mult
                    return (new_x,) + tuple(inputs[1:])
                return pre_hook

            self.handles.append(
                mlps[layer].down_proj.register_forward_pre_hook(make_pre_hook(idx, m))
            )

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def generate(
    model,
    tok,
    prompt: str,
    max_new_tokens: int = 64,
    circuit: Circuit | None = None,
    multiplier: float | None = None,
    device: str | None = None,
    chat_template: bool = True,
    seed: int | None = None,
) -> str:
    """Greedy generation, optionally with a circuit modulated in.

    Greedy by default so results are reproducible and a multiplier of 1.0
    is a true no-op against the baseline.
    """
    torch = _torch()
    dev = device or next(model.parameters()).device.type
    if seed is not None:
        torch.manual_seed(seed)
    ids = encode_prompts(tok, [prompt], dev, chat_template)[0]
    mod = None
    if circuit is not None and multiplier is not None and float(multiplier) != 1.0:
        mod = _Modulator(model, circuit, float(multiplier))
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
    finally:
        if mod is not None:
            mod.remove()
    new_tokens = out[0][ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------
# Metrics + dose scan
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def repetition_quality(text: str, n: int = 3) -> float:
    """The CNA paper's output-quality metric: 1 - repeated n-gram fraction.

    Returns 1.0 for text with no repeated n-grams (and for text too short
    to contain two n-grams), lower as repetition grows.
    """
    words = _WORD_RE.findall(text.lower())
    if len(words) <= n:
        return 1.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if len(grams) <= 1:
        return 1.0
    repeated = len(grams) - len(set(grams))
    return 1.0 - (repeated / len(grams))


def scan(
    model,
    tok,
    circuit: Circuit,
    prompts: list[str],
    multipliers: list[float],
    max_new_tokens: int = 64,
    marker: str | None = None,
    device: str | None = None,
    chat_template: bool = True,
) -> list[dict]:
    """Dose scan: generate at each multiplier, score quality + marker rate.

    This is the dose window made explicit: where the behavior turns on,
    and where output quality collapses. ``marker`` is an optional regex;
    marker_rate is the fraction of outputs matching it.
    """
    marker_re = re.compile(marker, re.IGNORECASE) if marker else None
    rows = []
    for m in multipliers:
        outputs = []
        qualities = []
        hits = 0
        for prompt in prompts:
            text = generate(
                model, tok, prompt,
                max_new_tokens=max_new_tokens,
                circuit=circuit, multiplier=float(m),
                device=device, chat_template=chat_template,
            )
            outputs.append(text)
            qualities.append(repetition_quality(text))
            if marker_re is not None and marker_re.search(text):
                hits += 1
        rows.append({
            "multiplier": float(m),
            "mean_quality": sum(qualities) / max(1, len(qualities)),
            "marker_rate": (hits / len(prompts)) if marker_re is not None and prompts else None,
            "outputs": outputs,
        })
    return rows
