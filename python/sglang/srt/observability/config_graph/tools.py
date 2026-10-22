"""Dependency-free graph validation and config lookup; no SGLang/GPU imports.

python path/to/config_graph/tools.py --check
python path/to/config_graph/tools.py --config chunked-prefill-size
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def load_graph():
    return json.loads((HERE / "graph.json").read_text())


def inventory(graph):
    audited = {n["id"] for n in graph["nodes"] if n["kind"] == "server_field"}
    fields = []
    for path in sorted((ROOT / "python/sglang/srt/arg_groups/fields").glob("*.py")):
        for cls in ast.walk(ast.parse(path.read_text())):
            if not isinstance(cls, ast.ClassDef):
                continue
            for field in cls.body:
                if not isinstance(field, ast.AnnAssign) or not isinstance(
                    field.target, ast.Name
                ):
                    continue
                annotation = ast.unparse(field.annotation)
                if not annotation.startswith("A["):
                    continue
                name = field.target.id
                fields.append(
                    dict(
                        id="config." + name,
                        name=name,
                        kind="server_field",
                        cli_candidate=(
                            None
                            if "no_cli=True" in annotation
                            else "--" + name.replace("_", "-")
                        ),
                        source=dict(
                            path=str(path.relative_to(ROOT)), anchor=name + ":"
                        ),
                        declared_default=(
                            ast.unparse(field.value) if field.value else None
                        ),
                        review_status=(
                            "causal_mapping"
                            if "config." + name in audited
                            else "not_causally_audited"
                        ),
                    )
                )
    return dict(
        schema_version=1,
        scope="Declarative A fields only; CLI names are candidates, aliases/default resolution and extensions require parser. Not an env/launcher inventory.",
        fields=fields,
    )


def check(graph):
    assert graph["schema_version"] == 1
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert len(nodes) == len(graph["nodes"]), "duplicate node IDs"
    for node in nodes.values():
        src = node["source"]
        path = (ROOT / src["path"]).resolve()
        assert path.is_relative_to(ROOT), src
        assert path.is_file() and src["anchor"] in path.read_text(), src
        if node["kind"] == "mechanism":
            for key in (
                "behavior",
                "conditions",
                "confounders",
                "hypothesis",
                "next_test",
                "remaining_gap",
                "evidence",
            ):
                assert node.get(key), (node["id"], key)
    for edge in graph["edges"]:
        assert edge["source"] in nodes and edge["target"] in nodes, edge
        assert edge["relation"] in {
            "controls_or_constrains",
            "observed_by",
            "indirectly_observed_by",
            "interacts_with",
        }
    current = inventory(graph)
    assert current == json.loads(
        (HERE / "inventory.json").read_text()
    ), "field inventory drift; run --refresh-inventory"
    # Validate observation contracts against actual production call sites, including
    # fields hidden in **values via focused runtime tests, not guessed metric names.
    spec = importlib.util.spec_from_file_location(
        "tuning_contract", HERE.parent / "tuning.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    used = set()
    for path in (ROOT / "python/sglang/srt").rglob("*.py"):
        if "tuning_recorder.observe(" not in path.read_text():
            continue
        for call in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "tuning_recorder"
                and call.func.attr == "observe"
                and call.args
                and isinstance(call.args[0], ast.Constant)
            ):
                event = call.args[0].value
                assert event in module.EVENTS, (path, event)
                assert {
                    k.arg for k in call.keywords if k.arg is not None
                } <= module.EVENTS[event], event
                used.add(event)
    for node in nodes.values():
        if node["kind"] == "decision_log":
            assert node["name"] in module.EVENTS
            assert node["numeric_fields"] == sorted(module.EVENTS[node["name"]])
    companion = json.loads((HERE / "explanation_model.json").read_text())
    assert companion["schema_version"] == 1
    rule_ids = set()
    for rule in companion["rules"]:
        assert rule["id"] not in rule_ids, rule["id"]
        rule_ids.add(rule["id"])
        for key in (
            "observation", "diagnosis", "alternatives", "check", "action",
            "expected", "reject_if", "status",
        ):
            assert rule.get(key), (rule["id"], key)
        assert rule["graph_nodes"] and all(
            n in nodes for n in rule["graph_nodes"]
        ), rule["id"]
        assert isinstance(rule["runtime_evidence"], list), rule["id"]
    return dict(
        explanation_rules=len(rule_ids),
        nodes=len(nodes),
        edges=len(graph["edges"]),
        inventoried_fields=len(current["fields"]),
        mapped_fields=sum(
            f["review_status"] == "causal_mapping" for f in current["fields"]
        ),
        literal_observation_hooks=len(used),
    )


def lookup(graph, name):
    name = name.removeprefix("--").replace("-", "_")
    target = "config." + name
    nodes = {n["id"]: n for n in graph["nodes"]}
    if target not in nodes:
        return dict(
            config=name,
            status="not_causally_audited",
            guidance="Check inventory.json and source; do not infer absence of an effect.",
        )
    mechanisms = {e["target"] for e in graph["edges"] if e["source"] == target}
    keep = {target} | mechanisms
    for edge in graph["edges"]:
        if edge["source"] in mechanisms or edge["target"] in mechanisms:
            keep.update((edge["source"], edge["target"]))
    return dict(
        nodes=[n for n in graph["nodes"] if n["id"] in keep],
        edges=[
            e for e in graph["edges"] if e["source"] in keep and e["target"] in keep
        ],
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--refresh-inventory", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args()
    graph = load_graph()
    if args.refresh_inventory:
        (HERE / "inventory.json").write_text(
            json.dumps(inventory(graph), indent=2) + "\n"
        )
    if args.check:
        print(json.dumps(check(graph), indent=2))
    if args.config:
        print(json.dumps(lookup(graph, args.config), indent=2))
    if not any((args.check, args.refresh_inventory, args.config)):
        parser.print_help()


if __name__ == "__main__":
    main()
