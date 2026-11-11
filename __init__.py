"""NeuronSteer — CNA neuron steering for Hermes Agent.

Registers six tools (find / generate / scan / list / show / demo), a
/steer slash command, and a `hermes steer ...` CLI. The heavy lifting
lives in engine.py (Contrastive Neuron Attribution, arXiv:2605.12290);
tool handlers import it lazily so this module loads fast and clean.
"""

from __future__ import annotations

try:
    from . import cli, schemas, store, tools
except ImportError:  # Hermes may load the plugin dir as flat modules
    import cli
    import schemas
    import store
    import tools

TOOLSET = "neuron_steer"


def _steer_slash(raw_args: str = "", **kwargs) -> str:
    """Slash command handler: return text, never print (gateway sessions)."""
    del kwargs
    saved = store.list_circuits()
    lines = [
        "NeuronSteer — Contrastive Neuron Attribution (Nous Research, arXiv:2605.12290).",
        "Tools: neuron_steer_find (build a circuit from two prompt sets),",
        "neuron_steer_generate (steer with it), neuron_steer_scan (dose window),",
        "neuron_steer_show (inspect a circuit), neuron_steer_list, and",
        "neuron_steer_demo (the whole flow offline, no download).",
        "CLI: hermes steer find|generate|scan|list|show|demo.",
    ]
    if saved:
        lines.append("Saved circuits:")
        for c in saved:
            lines.append(f"  - {c['name']}: {c['neurons']} neurons on {c['model_id']}")
    else:
        lines.append("No circuits saved yet — run neuron_steer_find to build one.")
    return "\n".join(lines)


def register(ctx):
    for schema in schemas.ALL:
        ctx.register_tool(
            name=schema["name"],
            toolset=TOOLSET,
            schema=schema,
            handler=tools.HANDLERS[schema["name"]],
        )
    ctx.register_command(
        "steer",
        _steer_slash,
        "NeuronSteer: CNA neuron steering — status and saved circuits.",
    )
    if hasattr(ctx, "register_cli_command"):
        ctx.register_cli_command(
            "steer",
            "NeuronSteer: find and apply CNA neuron steering circuits.",
            cli.setup,
            cli.handler,
        )
