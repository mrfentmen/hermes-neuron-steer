"""CLI surface for NeuronSteer: `hermes neuron-steer <subcommand>`.

Mirrors the tools so a circuit can be found, tested and dose-scanned from
the terminal — which is also how you demo it. Prompt sets are text files,
one prompt per line (blank lines and lines starting with # are skipped).
"""

from __future__ import annotations

import json


def _read_prompts(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [
            line.strip()
            for line in fh
            if line.strip() and not line.strip().startswith("#")
        ]


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2))


def setup(parser) -> None:
    sub = parser.add_subparsers(dest="steer_command")

    p_find = sub.add_parser("find", help="Find a CNA circuit from two prompt sets.")
    p_find.add_argument("--name", required=True, help="Circuit name to save.")
    p_find.add_argument("--positive-file", required=True, help="Prompts WITH the behavior, one per line.")
    p_find.add_argument("--negative-file", required=True, help="Prompts WITHOUT the behavior, one per line.")
    p_find.add_argument("--model", default=None, help="Hugging Face model id.")
    p_find.add_argument("--top-fraction", type=float, default=0.001)
    p_find.add_argument("--top-k", type=int, default=None)
    p_find.add_argument("--allow-download", action="store_true",
                        help="Allow downloading the model if it is not cached (default: cached models only).")

    p_gen = sub.add_parser("generate", help="Generate with a saved circuit modulated in.")
    p_gen.add_argument("--circuit", required=True)
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--multiplier", type=float, default=0.0)
    p_gen.add_argument("--model", default=None)
    p_gen.add_argument("--max-new-tokens", type=int, default=64)
    p_gen.add_argument("--no-baseline", action="store_true")
    p_gen.add_argument("--min-quality", type=float, default=None,
                       help="Quality floor; warn + safer-dose fallback if steered output falls below it.")
    p_gen.add_argument("--allow-download", action="store_true",
                       help="Allow downloading the model if it is not cached (default: cached models only).")

    p_scan = sub.add_parser("scan", help="Dose scan: one prompt set across multipliers.")
    p_scan.add_argument("--circuit", required=True)
    p_scan.add_argument("--prompts-file", required=True)
    p_scan.add_argument("--multipliers", required=True, help="Comma list, e.g. 1,0.5,0,-1")
    p_scan.add_argument("--marker", default=None, help="Regex scored per output.")
    p_scan.add_argument("--model", default=None)
    p_scan.add_argument("--max-new-tokens", type=int, default=64)
    p_scan.add_argument("--allow-download", action="store_true",
                        help="Allow downloading the model if it is not cached (default: cached models only).")

    sub.add_parser("list", help="List saved circuits.")

    p_show = sub.add_parser("show", help="Inspect a saved circuit (layers, top neurons).")
    p_show.add_argument("--circuit", required=True)

    p_demo = sub.add_parser("demo", help="Run the full flow offline on a tiny random model.")
    p_demo.add_argument("--top-k", type=int, default=16)
    p_demo.add_argument("--max-new-tokens", type=int, default=16)


def handler(args) -> None:
    try:
        from . import tools
    except ImportError:
        import tools

    command = getattr(args, "steer_command", None)
    if command == "find":
        result = tools.handle_find({
            "name": args.name,
            "positive_prompts": _read_prompts(args.positive_file),
            "negative_prompts": _read_prompts(args.negative_file),
            "model": args.model,
            "top_fraction": args.top_fraction,
            "top_k": args.top_k,
            "allow_download": args.allow_download,
        })
    elif command == "generate":
        result = tools.handle_generate({
            "circuit": args.circuit,
            "prompt": args.prompt,
            "multiplier": args.multiplier,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "include_baseline": not args.no_baseline,
            "min_quality": args.min_quality,
            "allow_download": args.allow_download,
        })
    elif command == "scan":
        result = tools.handle_scan({
            "circuit": args.circuit,
            "prompts": _read_prompts(args.prompts_file),
            "multipliers": [float(x) for x in args.multipliers.split(",") if x.strip()],
            "marker": args.marker,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "allow_download": args.allow_download,
        })
    elif command == "list":
        result = tools.handle_list({})
    elif command == "show":
        result = tools.handle_show({"circuit": args.circuit})
    elif command == "demo":
        result = tools.handle_demo({
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        })
    else:
        print("Usage: hermes neuron-steer {find|generate|scan|list|show|demo} — see --help.")
        return
    _print_json(json.loads(result))
