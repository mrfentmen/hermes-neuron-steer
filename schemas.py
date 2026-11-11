"""Tool schemas for NeuronSteer — what the model sees."""

FIND = {
    "name": "neuron_steer_find",
    "description": (
        "Find a neuron steering circuit with Contrastive Neuron Attribution "
        "(CNA, Nous Research arXiv:2605.12290). Runs a local Hugging Face "
        "model over two prompt sets (positive = the behavior you want more "
        "of, negative = the behavior you want less of), records MLP neuron "
        "activations at the last token, and ranks neurons by the mean "
        "activation difference. The top neurons (default: top 0.1%, the "
        "paper's setting) are saved as a named circuit you can steer with. "
        "Forward passes only: no training, no weight changes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name to save the circuit under, e.g. 'refusal' or 'formal'.",
            },
            "positive_prompts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Prompts that express the target behavior (8-32 short prompts work well).",
            },
            "negative_prompts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Matched prompts that do not express the behavior.",
            },
            "model": {
                "type": "string",
                "description": "Hugging Face model id (Llama/Qwen/Granite/Mistral family). Optional; defaults to the NEURON_STEER_MODEL env var or Qwen/Qwen2.5-1.5B-Instruct.",
            },
            "top_fraction": {
                "type": "number",
                "description": "Fraction of all MLP neurons to keep (default 0.001 = 0.1%, the CNA paper's setting). Ignored if top_k is set.",
            },
            "top_k": {
                "type": "integer",
                "description": "Exact number of neurons to keep. Overrides top_fraction.",
            },
        },
        "required": ["name", "positive_prompts", "negative_prompts"],
    },
}

GENERATE = {
    "name": "neuron_steer_generate",
    "description": (
        "Generate from the local model with a saved neuron circuit "
        "modulated in. Each circuit neuron's activation is multiplied by "
        "'multiplier' at inference: 0 ablates the circuit, 1 is baseline, "
        ">1 amplifies it. Returns the baseline and steered outputs side by "
        "side, each scored with the CNA quality metric (1 = no repeated "
        "n-grams)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The prompt to generate from."},
            "circuit": {"type": "string", "description": "Name of a saved circuit (from neuron_steer_find)."},
            "multiplier": {
                "type": "number",
                "description": "Activation multiplier for the circuit neurons (default 0 = ablate).",
            },
            "model": {
                "type": "string",
                "description": "Hugging Face model id. Optional; defaults to the circuit's model.",
            },
            "max_new_tokens": {"type": "integer", "description": "Max tokens to generate (default 64)."},
            "include_baseline": {
                "type": "boolean",
                "description": "Also generate the unsteered baseline for comparison (default true).",
            },
            "min_quality": {
                "type": "number",
                "description": "Optional quality floor (0-1). If the steered output's CNA quality falls below it, the result carries a warning and a fallback generation at a safer dose, halfway back toward baseline.",
            },
        },
        "required": ["prompt", "circuit"],
    },
}

SCAN = {
    "name": "neuron_steer_scan",
    "description": (
        "Dose scan: generate the same prompts at several circuit "
        "multipliers and report, per multiplier, the mean CNA output "
        "quality (1 - repeated n-gram fraction) and optionally the rate at "
        "which outputs match a behavior marker regex. Use it to find the "
        "dose window: where the behavior turns on before quality breaks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "circuit": {"type": "string", "description": "Name of a saved circuit."},
            "prompts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Prompts to test at every multiplier.",
            },
            "multipliers": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Multipliers to sweep, e.g. [1, 0.5, 0, -1, 2]. Include 1 for the baseline.",
            },
            "marker": {
                "type": "string",
                "description": "Optional regex; marker_rate is the fraction of outputs matching it (case-insensitive).",
            },
            "model": {
                "type": "string",
                "description": "Hugging Face model id. Optional; defaults to the circuit's model.",
            },
            "max_new_tokens": {"type": "integer", "description": "Max tokens per generation (default 64)."},
        },
        "required": ["circuit", "prompts", "multipliers"],
    },
}

LIST = {
    "name": "neuron_steer_list",
    "description": "List the neuron steering circuits saved on this machine, with their model and size.",
    "parameters": {"type": "object", "properties": {}},
}

SHOW = {
    "name": "neuron_steer_show",
    "description": (
        "Inspect a saved neuron circuit: how many neurons it has, which "
        "layers they sit in (the full layer histogram plus the strongest "
        "neuron in each layer), and the strongest neurons overall with "
        "their CNA deltas. Reads the saved circuit only; no model is "
        "loaded."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "circuit": {"type": "string", "description": "Name of a saved circuit."},
        },
        "required": ["circuit"],
    },
}

DEMO = {
    "name": "neuron_steer_demo",
    "description": (
        "Run the whole NeuronSteer flow offline, in seconds, with no "
        "model download: builds a tiny randomly initialised Llama in "
        "memory, finds a circuit by contrast, generates baseline vs "
        "ablated vs amplified, and dose-scans it. The weights are random "
        "so the text is gibberish; the demo proves the mechanics end to "
        "end. Use neuron_steer_find for a real model."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "top_k": {
                "type": "integer",
                "description": "Circuit size for the demo (default 16, max 512).",
            },
            "max_new_tokens": {
                "type": "integer",
                "description": "Tokens per demo generation (default 16, max 64).",
            },
        },
        "required": [],
    },
}

ALL = [FIND, GENERATE, SCAN, LIST, SHOW, DEMO]
