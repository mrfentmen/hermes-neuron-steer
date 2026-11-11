"""Tool handlers for NeuronSteer.

Handlers follow the Hermes plugin convention: take a params dict, return a
JSON string. The engine (torch/transformers) is imported lazily inside each
handler so the plugin loads instantly and reports a clean error when the
dependencies are not installed yet, instead of failing to load.
"""

from __future__ import annotations

import json
import os

try:
    from . import store
except ImportError:  # loaded as a flat module directory
    import store

DEFAULT_MODEL = os.environ.get("NEURON_STEER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_OUTPUT_CHARS = 1200


def _ok(payload: dict) -> str:
    return json.dumps({"success": True, **payload})


def _err(exc: Exception) -> str:
    return json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"})


def _clip(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT_CHARS else text[:MAX_OUTPUT_CHARS] + " […]"


def handle_find(params, **kwargs):
    del kwargs
    try:
        from . import engine
    except ImportError:
        import engine
    try:
        # Validate BEFORE loading any model: a bad call must fail cheap and
        # must never trigger a model download.
        name = params.get("name")
        positive = params.get("positive_prompts")
        negative = params.get("negative_prompts")
        if not name or not isinstance(positive, list) or not positive \
                or not isinstance(negative, list) or not negative:
            raise ValueError(
                "neuron_steer_find requires 'name', a non-empty "
                "'positive_prompts' list and a non-empty 'negative_prompts' list."
            )
        model_id = params.get("model") or DEFAULT_MODEL
        model, tok = engine.load_model(model_id)
        circuit = engine.find_circuit(
            model,
            tok,
            positive_prompts=list(params["positive_prompts"]),
            negative_prompts=list(params["negative_prompts"]),
            model_id=model_id,
            top_fraction=float(params.get("top_fraction", 0.001)),
            top_k=params.get("top_k"),
        )
        store.save_circuit(params["name"], circuit)
        hist = circuit.layer_histogram()
        top_layers = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)[:5]
        return _ok({
            "circuit": params["name"],
            "model": model_id,
            "neurons_selected": len(circuit.neurons),
            "total_mlp_neurons": engine.total_mlp_neurons(model),
            "top_neurons": [n.to_dict() for n in circuit.neurons[:10]],
            "layers_with_most_circuit_neurons": [
                {"layer": layer, "count": count} for layer, count in top_layers
            ],
            "note": (
                "Circuit saved. Steer it with neuron_steer_generate "
                "(multiplier 0 ablates it) or map the dose window with "
                "neuron_steer_scan."
            ),
        })
    except Exception as exc:  # surfaced to the model/user as a tool error
        return _err(exc)


def handle_generate(params, **kwargs):
    del kwargs
    try:
        from . import engine
    except ImportError:
        import engine
    try:
        if not params.get("prompt") or not params.get("circuit"):
            raise ValueError("neuron_steer_generate requires 'prompt' and 'circuit'.")
        circuit = store.load_circuit(params["circuit"])
        model_id = params.get("model") or circuit.model_id or DEFAULT_MODEL
        model, tok = engine.load_model(model_id)
        multiplier = float(params.get("multiplier", 0.0))
        max_new = int(params.get("max_new_tokens", 64))
        steered = engine.generate(
            model, tok, params["prompt"],
            max_new_tokens=max_new, circuit=circuit, multiplier=multiplier,
        )
        steered_quality = engine.repetition_quality(steered)
        result = {
            "circuit": params["circuit"],
            "model": model_id,
            "multiplier": multiplier,
            "steered": _clip(steered),
            "steered_quality": round(steered_quality, 4),
        }
        # Quality guardrail: when the caller sets a floor and the steered
        # output falls below it, say so plainly and offer one safer dose —
        # halfway back toward baseline — instead of a wrecked output.
        if params.get("min_quality") is not None:
            floor = float(params["min_quality"])
            result["min_quality"] = floor
            if steered_quality < floor:
                fallback_multiplier = (multiplier + 1.0) / 2.0
                fallback = engine.generate(
                    model, tok, params["prompt"],
                    max_new_tokens=max_new, circuit=circuit,
                    multiplier=fallback_multiplier,
                )
                fallback_quality = engine.repetition_quality(fallback)
                result["warning"] = (
                    f"Steered quality {steered_quality:.3f} fell below the "
                    f"floor {floor:.3f} at multiplier {multiplier}. A safer "
                    f"dose was also run at multiplier {fallback_multiplier} "
                    f"(quality {fallback_quality:.3f})."
                )
                result["fallback_multiplier"] = fallback_multiplier
                result["fallback"] = _clip(fallback)
                result["fallback_quality"] = round(fallback_quality, 4)
        if params.get("include_baseline", True):
            baseline = engine.generate(model, tok, params["prompt"], max_new_tokens=max_new)
            result["baseline"] = _clip(baseline)
            result["baseline_quality"] = round(engine.repetition_quality(baseline), 4)
        return _ok(result)
    except Exception as exc:
        return _err(exc)


def handle_scan(params, **kwargs):
    del kwargs
    try:
        from . import engine
    except ImportError:
        import engine
    try:
        if not params.get("circuit") or not params.get("prompts") \
                or not params.get("multipliers"):
            raise ValueError(
                "neuron_steer_scan requires 'circuit', 'prompts' and 'multipliers'."
            )
        circuit = store.load_circuit(params["circuit"])
        model_id = params.get("model") or circuit.model_id or DEFAULT_MODEL
        model, tok = engine.load_model(model_id)
        rows = engine.scan(
            model, tok, circuit,
            prompts=list(params["prompts"]),
            multipliers=[float(m) for m in params["multipliers"]],
            max_new_tokens=int(params.get("max_new_tokens", 64)),
            marker=params.get("marker"),
        )
        for row in rows:
            row["mean_quality"] = round(row["mean_quality"], 4)
            if row["marker_rate"] is not None:
                row["marker_rate"] = round(row["marker_rate"], 4)
            row["outputs"] = [_clip(o) for o in row["outputs"]]
        return _ok({"circuit": params["circuit"], "model": model_id, "scan": rows})
    except Exception as exc:
        return _err(exc)


def handle_list(params, **kwargs):
    del params, kwargs
    try:
        return _ok({"circuits": store.list_circuits()})
    except Exception as exc:
        return _err(exc)


def handle_show(params, **kwargs):
    del kwargs
    try:
        if not params.get("circuit"):
            raise ValueError("neuron_steer_show requires 'circuit'.")
        circuit = store.load_circuit(params["circuit"])
        doc = store.load_circuit_doc(params["circuit"])
        hist = circuit.layer_histogram()
        deltas = [abs(n.delta) for n in circuit.neurons]
        per_layer_top = []
        for layer, refs in sorted(circuit.by_layer().items()):
            best = max(refs, key=lambda r: abs(r.delta))
            per_layer_top.append({
                "layer": layer,
                "count": len(refs),
                "top_neuron": best.neuron,
                "top_delta": round(best.delta, 4),
            })
        return _ok({
            "circuit": params["circuit"],
            "saved_at": doc.get("saved_at", ""),
            "model": circuit.model_id,
            "neurons": len(circuit.neurons),
            "positive_count": circuit.positive_count,
            "negative_count": circuit.negative_count,
            "layers_touched": len(hist),
            "layer_histogram": [
                {"layer": layer, "count": count}
                for layer, count in sorted(hist.items())
            ],
            "per_layer_top": per_layer_top,
            "top_neurons": [n.to_dict() for n in circuit.neurons[:15]],
            "mean_abs_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
            "max_abs_delta": round(max(deltas), 4) if deltas else 0.0,
        })
    except Exception as exc:
        return _err(exc)


def handle_demo(params, **kwargs):
    del kwargs
    try:
        from . import demo
    except ImportError:
        import demo
    try:
        # No model is ever loaded from disk or the network here: the demo
        # builds a tiny random model in memory (see demo.py).
        top_k = int(params.get("top_k", 16))
        max_new = int(params.get("max_new_tokens", 16))
        if not 1 <= top_k <= 512:
            raise ValueError("top_k must be between 1 and 512.")
        if not 1 <= max_new <= 64:
            raise ValueError("max_new_tokens must be between 1 and 64.")
        return _ok(demo.run_demo(top_k=top_k, max_new_tokens=max_new))
    except Exception as exc:
        return _err(exc)


HANDLERS = {
    "neuron_steer_find": handle_find,
    "neuron_steer_generate": handle_generate,
    "neuron_steer_scan": handle_scan,
    "neuron_steer_list": handle_list,
    "neuron_steer_show": handle_show,
    "neuron_steer_demo": handle_demo,
}
