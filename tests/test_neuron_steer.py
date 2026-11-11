"""NeuronSteer tests — run against a tiny randomly-initialised Llama.

The tiny model proves the mechanics exactly (CNA ranking, modulation
hooks, metric math, plugin registration) without any downloads. Behavior
on a real model is covered by the smoke test in the README.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

import engine  # noqa: E402
import store  # noqa: E402
from engine import Circuit, NeuronRef  # noqa: E402


class FakeTokenizer:
    """Char-level tokenizer: engine only needs __call__ + decode."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __call__(self, text, return_tensors=None):
        ids = [max(3, ord(c) % 256) for c in text] or [3]

        class Enc:
            pass

        enc = Enc()
        enc.input_ids = torch.tensor([ids], dtype=torch.long)
        return enc

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids if int(i) > 2)


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(1234)
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


POSITIVE = ["aaaaaaaaaa", "AAAAAAAAAA", "aaaa AAAA aaaa"]
NEGATIVE = ["zzzzzzzzzz", "0000000000", "zzzz 0000 zzzz"]


class FindCircuitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.tok = FakeTokenizer()

    def test_matches_manual_recompute(self):
        pos = engine._last_token_means(self.model, self.tok, POSITIVE, "cpu", False)
        neg = engine._last_token_means(self.model, self.tok, NEGATIVE, "cpu", False)
        delta = pos - neg
        k = 8
        _, idx = torch.topk(delta.abs().flatten(), k)
        expected = {(int(i) // 128, int(i) % 128) for i in idx.tolist()}

        circuit = engine.find_circuit(
            self.model, self.tok, POSITIVE, NEGATIVE,
            model_id="tiny", top_k=k, device="cpu", chat_template=False,
        )
        got = {(n.layer, n.neuron) for n in circuit.neurons}
        self.assertEqual(got, expected)
        self.assertEqual(len(circuit.neurons), k)
        mags = [abs(n.delta) for n in circuit.neurons]
        self.assertEqual(mags, sorted(mags, reverse=True))
        for n in circuit.neurons:
            self.assertAlmostEqual(n.delta, float(delta[n.layer, n.neuron]), places=5)

    def test_fraction_default_size(self):
        circuit = engine.find_circuit(
            self.model, self.tok, POSITIVE, NEGATIVE,
            model_id="tiny", top_fraction=0.01, device="cpu", chat_template=False,
        )
        total = engine.total_mlp_neurons(self.model)
        self.assertEqual(total, 4 * 128)
        self.assertEqual(len(circuit.neurons), round(total * 0.01))

    def test_requires_both_sets(self):
        with self.assertRaises(ValueError):
            engine.find_circuit(
                self.model, self.tok, [], NEGATIVE,
                device="cpu", chat_template=False,
            )


class ModulatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.tok = FakeTokenizer()
        cls.ids = cls.tok("hello world").input_ids

    def _capture(self, layer, circuit=None, multiplier=None):
        """Run one forward pass; return the down-proj inputs of `layer`."""
        mlps = engine.mlp_modules(self.model)
        seen = {}

        def probe(module, inputs, output):
            seen["x"] = inputs[0].detach().clone()

        handle = mlps[layer].down_proj.register_forward_hook(probe)
        mod = None
        if circuit is not None:
            mod = engine._Modulator(self.model, circuit, multiplier)
        try:
            with torch.no_grad():
                self.model(input_ids=self.ids)
        finally:
            handle.remove()
            if mod is not None:
                mod.remove()
        return seen["x"]

    def test_zero_ablates_only_the_target(self):
        circuit = Circuit("tiny", [NeuronRef(1, 5, 1.0)])
        base = self._capture(1)
        ablated = self._capture(1, circuit, 0.0)
        self.assertTrue(torch.all(ablated[..., 5] == 0))
        others = [i for i in range(128) if i != 5]
        self.assertTrue(torch.allclose(ablated[..., others], base[..., others]))

    def test_double_doubles_and_identity_is_identity(self):
        circuit = Circuit("tiny", [NeuronRef(2, 7, 1.0)])
        base = self._capture(2)
        doubled = self._capture(2, circuit, 2.0)
        self.assertTrue(torch.allclose(doubled[..., 7], base[..., 7] * 2))
        identity = self._capture(2, circuit, 1.0)
        self.assertTrue(torch.allclose(identity, base))

    def test_full_ablation_moves_the_logits(self):
        mlps = engine.mlp_modules(self.model)
        all_neurons = [
            NeuronRef(layer, n, 1.0)
            for layer in range(len(mlps))
            for n in range(128)
        ]
        circuit = Circuit("tiny", all_neurons)
        with torch.no_grad():
            base_logits = self.model(input_ids=self.ids).logits
        mod = engine._Modulator(self.model, circuit, 0.0)
        try:
            with torch.no_grad():
                steered_logits = self.model(input_ids=self.ids).logits
        finally:
            mod.remove()
        self.assertFalse(torch.allclose(base_logits, steered_logits))


class GenerateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.tok = FakeTokenizer()

    def test_greedy_is_deterministic(self):
        a = engine.generate(self.model, self.tok, "hello", max_new_tokens=8,
                            device="cpu", chat_template=False)
        b = engine.generate(self.model, self.tok, "hello", max_new_tokens=8,
                            device="cpu", chat_template=False)
        self.assertIsInstance(a, str)
        self.assertEqual(a, b)

    def test_steered_generate_runs(self):
        circuit = Circuit("tiny", [NeuronRef(0, 3, 1.0), NeuronRef(3, 9, -1.0)])
        out = engine.generate(self.model, self.tok, "hello", max_new_tokens=8,
                              circuit=circuit, multiplier=0.0,
                              device="cpu", chat_template=False)
        self.assertIsInstance(out, str)


class MetricTest(unittest.TestCase):
    def test_repetition_quality_values(self):
        self.assertEqual(engine.repetition_quality("one two three four five"), 1.0)
        self.assertEqual(engine.repetition_quality(""), 1.0)
        self.assertEqual(engine.repetition_quality("x y"), 1.0)
        # 6 identical words -> 4 trigrams, all the same -> 3 of 4 repeated
        self.assertAlmostEqual(engine.repetition_quality("a a a a a a"), 0.25)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("NEURON_STEER_HOME")
        os.environ["NEURON_STEER_HOME"] = self.tmp.name

    def tearDown(self):
        if self._old is None:
            os.environ.pop("NEURON_STEER_HOME", None)
        else:
            os.environ["NEURON_STEER_HOME"] = self._old
        self.tmp.cleanup()

    def test_roundtrip(self):
        circuit = Circuit("tiny", [NeuronRef(1, 2, 0.5)], positive_count=3, negative_count=3)
        store.save_circuit("demo-circuit", circuit)
        loaded = store.load_circuit("demo-circuit")
        self.assertEqual(loaded.model_id, "tiny")
        self.assertEqual(loaded.neurons[0].neuron, 2)
        listing = store.list_circuits()
        self.assertEqual([c["name"] for c in listing], ["demo-circuit"])
        self.assertTrue(store.delete_circuit("demo-circuit"))

    def test_bad_name_rejected(self):
        with self.assertRaises(ValueError):
            store.save_circuit("../escape", Circuit("tiny", []))


class FakeCtx:
    def __init__(self):
        self.tools = {}
        self.commands = {}
        self.cli_commands = {}

    def register_tool(self, name, toolset, schema, handler):
        self.tools[name] = (toolset, schema, handler)

    def register_command(self, name, handler, description):
        self.commands[name] = (handler, description)

    def register_cli_command(self, name, help, setup_fn, handler_fn):
        self.cli_commands[name] = (help, setup_fn, handler_fn)


class RegistrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["NEURON_STEER_HOME"] = self.tmp.name

    def tearDown(self):
        os.environ.pop("NEURON_STEER_HOME", None)
        self.tmp.cleanup()

    def _load_plugin(self):
        spec = importlib.util.spec_from_file_location(
            "neuron_steer_plugin",
            PLUGIN_DIR / "__init__.py",
            submodule_search_locations=[str(PLUGIN_DIR)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["neuron_steer_plugin"] = module
        spec.loader.exec_module(module)
        return module

    def test_register_wires_everything(self):
        plugin = self._load_plugin()
        ctx = FakeCtx()
        plugin.register(ctx)
        self.assertEqual(
            set(ctx.tools),
            {"neuron_steer_find", "neuron_steer_generate",
             "neuron_steer_scan", "neuron_steer_list"},
        )
        for name, (toolset, schema, handler) in ctx.tools.items():
            self.assertEqual(toolset, "neuron_steer")
            self.assertTrue(schema["description"])
            self.assertTrue(callable(handler))
        self.assertIn("steer", ctx.commands)
        self.assertIn("steer", ctx.cli_commands)

    def test_handlers_return_json_and_fail_cleanly(self):
        plugin = self._load_plugin()
        ctx = FakeCtx()
        plugin.register(ctx)
        out = json.loads(ctx.tools["neuron_steer_list"][2]({}))
        self.assertTrue(out["success"])
        out = json.loads(ctx.tools["neuron_steer_find"][2]({"name": "x"}))
        self.assertFalse(out["success"])  # missing prompt sets -> clean error
        slash = ctx.commands["steer"][0]()
        self.assertIn("NeuronSteer", slash)


if __name__ == "__main__":
    unittest.main()
