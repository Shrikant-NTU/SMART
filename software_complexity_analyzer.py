from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    from radon.metrics import h_visit
    from radon.visitors import ComplexityVisitor
except ImportError:  # pragma: no cover - handled when the analyzer is run.
    h_visit = None
    ComplexityVisitor = None


HALSTEAD_FIELDS = (
    "h1",
    "h2",
    "N1",
    "N2",
    "vocabulary",
    "length",
    "calculated_length",
    "volume",
    "difficulty",
    "effort",
    "time",
)

IGNORED_DIRECTORIES = {
    ".git",
    ".images",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
}

@dataclass
class ImportBindings:
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class SourceModule:
    name: str
    path: Path
    relative_path: str
    source: str
    tree: ast.Module
    is_package: bool
    imports: ImportBindings
    callable_aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class FunctionRecord:
    identifier: str
    module_name: str
    relative_file: str
    qualname: str
    name: str
    line: int
    end_line: int
    node: ast.FunctionDef | ast.AsyncFunctionDef
    parent_qualname: str | None
    class_qualname: str | None


@dataclass(frozen=True)
class CallSite:
    source: str
    requested_name: str
    canonical_name: str
    file: str
    line: int
    resolved_target: str | None
    classification: str


def terminal_print(message: Any, file: Any = sys.stdout) -> None:
    for line in str(message).splitlines() or [""]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {line}", file=file, flush=True)


def require_radon() -> None:
    if h_visit is None or ComplexityVisitor is None:
        raise RuntimeError(
            "radon is required for software-complexity analysis. "
            "Install it with: python -m pip install radon>=6.0.1"
        )


def module_name_for_path(root_path: Path, path: Path) -> tuple[str, bool]:
    relative = path.resolve().relative_to(root_path.resolve())
    parts = list(relative.with_suffix("").parts)
    is_package = bool(parts and parts[-1] == "__init__")
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def relative_import_module(
    current_module: str,
    is_package: bool,
    imported_module: str | None,
    level: int,
) -> str:
    if level == 0:
        return imported_module or ""
    package_parts = current_module.split(".") if is_package else current_module.split(".")[:-1]
    keep = max(0, len(package_parts) - (level - 1))
    base_parts = package_parts[:keep]
    if imported_module:
        base_parts.extend(imported_module.split("."))
    return ".".join(part for part in base_parts if part)


def collect_imports(
    tree: ast.Module,
    module_name: str,
    is_package: bool,
) -> ImportBindings:
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                local_name = item.asname or item.name.split(".")[0]
                aliases[local_name] = item.name if item.asname else item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            base = relative_import_module(
                module_name,
                is_package,
                node.module,
                node.level,
            )
            for item in node.names:
                if item.name == "*":
                    continue
                local_name = item.asname or item.name
                aliases[local_name] = ".".join(part for part in (base, item.name) if part)
    return ImportBindings(aliases=aliases)


def discover_python_files(root_path: Path) -> list[Path]:
    files: list[Path] = []
    for path in root_path.rglob("*.py"):
        relative = path.relative_to(root_path)
        if any(part in IGNORED_DIRECTORIES or part.startswith(".") for part in relative.parts[:-1]):
            continue
        files.append(path)
    return sorted(files)


def load_modules(root_path: Path) -> dict[str, SourceModule]:
    modules: dict[str, SourceModule] = {}
    for path in discover_python_files(root_path):
        source = path.read_text(encoding="utf-8", errors="replace")
        relative_path = path.relative_to(root_path).as_posix()
        tree = ast.parse(source, filename=str(path))
        module_name, is_package = module_name_for_path(root_path, path)
        if not module_name:
            continue
        modules[module_name] = SourceModule(
            name=module_name,
            path=path,
            relative_path=relative_path,
            source=source,
            tree=tree,
            is_package=is_package,
            imports=collect_imports(tree, module_name, is_package),
        )
    return modules


def collect_functions(modules: dict[str, SourceModule]) -> dict[str, FunctionRecord]:
    functions: dict[str, FunctionRecord] = {}

    def visit_body(
        module: SourceModule,
        body: list[ast.stmt],
        parent_qualname: str | None = None,
        class_qualname: str | None = None,
    ) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{parent_qualname}.{node.name}" if parent_qualname else node.name
                identifier = f"{module.name}.{qualname}"
                functions[identifier] = FunctionRecord(
                    identifier=identifier,
                    module_name=module.name,
                    relative_file=module.relative_path,
                    qualname=qualname,
                    name=node.name,
                    line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    node=node,
                    parent_qualname=parent_qualname,
                    class_qualname=class_qualname,
                )
                visit_body(
                    module,
                    node.body,
                    parent_qualname=qualname,
                    class_qualname=class_qualname,
                )
            elif isinstance(node, ast.ClassDef):
                qualname = f"{parent_qualname}.{node.name}" if parent_qualname else node.name
                visit_body(
                    module,
                    node.body,
                    parent_qualname=qualname,
                    class_qualname=qualname,
                )

    for module in modules.values():
        visit_body(module, module.tree.body)
    return functions


def dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def canonicalize_name(name: str, module: SourceModule) -> str:
    parts = name.split(".")
    replacement = module.imports.aliases.get(parts[0])
    if replacement:
        return ".".join([replacement, *parts[1:]])
    return name


def resolve_relative_function_name(
    name: str,
    module: SourceModule,
    current: FunctionRecord | None,
    functions: dict[str, FunctionRecord],
    callable_aliases: dict[str, str] | None = None,
) -> str | None:
    callable_aliases = callable_aliases or {}
    canonical = canonicalize_name(name, module)
    candidates: list[str] = []

    if canonical != name:
        candidates.append(canonical)
    elif current is not None and current.class_qualname and canonical.startswith(("self.", "cls.")):
        _, _, method_name = canonical.partition(".")
        candidates.append(f"{module.name}.{current.class_qualname}.{method_name}")
    elif "." in canonical:
        candidates.extend((canonical, f"{module.name}.{canonical}"))
    else:
        if current is not None:
            parent = current.qualname
            while parent:
                candidates.append(f"{module.name}.{parent}.{canonical}")
                parent = parent.rpartition(".")[0]
            if current.class_qualname:
                candidates.append(f"{module.name}.{current.class_qualname}.{canonical}")
        candidates.append(f"{module.name}.{canonical}")

    for candidate in candidates:
        if candidate in functions:
            return candidate
        alias_target = callable_aliases.get(candidate) or module.callable_aliases.get(candidate)
        visited: set[str] = set()
        while alias_target and alias_target not in visited:
            if alias_target in functions:
                return alias_target
            visited.add(alias_target)
            alias_target = callable_aliases.get(alias_target)
    return None


def find_project_function_reference(
    node: ast.AST,
    module: SourceModule,
    functions: dict[str, FunctionRecord],
) -> str | None:
    name = dotted_name(node)
    if name:
        return resolve_relative_function_name(name, module, None, functions)
    if isinstance(node, ast.Call):
        for argument in [*node.args, *[keyword.value for keyword in node.keywords]]:
            resolved = find_project_function_reference(argument, module, functions)
            if resolved:
                return resolved
    return None


def collect_callable_aliases(
    modules: dict[str, SourceModule],
    functions: dict[str, FunctionRecord],
) -> None:
    for module in modules.values():
        for node in module.tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            target_names = [target.id for target in targets if isinstance(target, ast.Name)]
            if not target_names:
                continue
            resolved = find_project_function_reference(value, module, functions)
            if resolved:
                for target_name in target_names:
                    module.callable_aliases[f"{module.name}.{target_name}"] = resolved


def combined_callable_aliases(modules: dict[str, SourceModule]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for module in modules.values():
        aliases.update(module.callable_aliases)
    return aliases


class ScopedCallCollector(ast.NodeVisitor):
    def __init__(self, root: ast.AST) -> None:
        self.root = root
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.root:
            for statement in node.body:
                self.visit(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node is self.root:
            for statement in node.body:
                self.visit(statement)


def calls_for_scope(scope: ast.AST) -> list[ast.Call]:
    collector = ScopedCallCollector(scope)
    if isinstance(scope, ast.Module):
        for statement in scope.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                collector.visit(statement)
    else:
        collector.visit(scope)
    return collector.calls


def classify_call(
    call: ast.Call,
    source_identifier: str,
    module: SourceModule,
    current: FunctionRecord | None,
    modules: dict[str, SourceModule],
    functions: dict[str, FunctionRecord],
    callable_aliases: dict[str, str],
) -> CallSite | None:
    requested_name = dotted_name(call.func)
    if requested_name is None:
        return None
    canonical_name = canonicalize_name(requested_name, module)
    resolved = resolve_relative_function_name(
        requested_name,
        module,
        current,
        functions,
        callable_aliases,
    )
    if resolved:
        classification = "project_function"
    elif any(
        canonical_name == module_name or canonical_name.startswith(f"{module_name}.")
        for module_name in modules
    ):
        classification = "unresolved_project_call"
    elif requested_name.split(".")[0] in module.imports.aliases:
        classification = "external_library"
    else:
        classification = "builtin_or_runtime_value"
    return CallSite(
        source=source_identifier,
        requested_name=requested_name,
        canonical_name=canonical_name,
        file=module.relative_path,
        line=call.lineno,
        resolved_target=resolved,
        classification=classification,
    )


def higher_order_project_function_sites(
    call: ast.Call,
    source_identifier: str,
    module: SourceModule,
    current: FunctionRecord | None,
    functions: dict[str, FunctionRecord],
    callable_aliases: dict[str, str],
) -> list[CallSite]:
    sites: dict[str, CallSite] = {}
    arguments = [*call.args, *[keyword.value for keyword in call.keywords]]
    for argument in arguments:
        for node in ast.walk(argument):
            if not isinstance(node, (ast.Name, ast.Attribute)):
                continue
            requested_name = dotted_name(node)
            if not requested_name:
                continue
            resolved = resolve_relative_function_name(
                requested_name,
                module,
                current,
                functions,
                callable_aliases,
            )
            if not resolved:
                continue
            sites[resolved] = CallSite(
                source=source_identifier,
                requested_name=requested_name,
                canonical_name=canonicalize_name(requested_name, module),
                file=module.relative_path,
                line=getattr(node, "lineno", call.lineno),
                resolved_target=resolved,
                classification="project_function_reference",
            )
    return list(sites.values())


def build_resolved_project_scope(
    root_module: SourceModule,
    modules: dict[str, SourceModule],
    functions: dict[str, FunctionRecord],
    callable_aliases: dict[str, str],
    root_function_identifier: str | None = None,
) -> tuple[set[str], list[CallSite]]:
    root_functions = (
        {root_function_identifier}
        if root_function_identifier is not None
        else {
            identifier
            for identifier, record in functions.items()
            if record.module_name == root_module.name
        }
    )
    reachable = set(root_functions)
    queue = sorted(root_functions)
    call_sites: list[CallSite] = []

    if root_function_identifier is None:
        for call in calls_for_scope(root_module.tree):
            site = classify_call(
                call,
                f"{root_module.name}.<module>",
                root_module,
                None,
                modules,
                functions,
                callable_aliases,
            )
            if site is None:
                continue
            call_sites.append(site)
            if site.resolved_target and site.resolved_target not in reachable:
                reachable.add(site.resolved_target)
                queue.append(site.resolved_target)
            for reference_site in higher_order_project_function_sites(
                call,
                f"{root_module.name}.<module>",
                root_module,
                None,
                functions,
                callable_aliases,
            ):
                call_sites.append(reference_site)
                if reference_site.resolved_target not in reachable:
                    reachable.add(reference_site.resolved_target)
                    queue.append(reference_site.resolved_target)

    processed: set[str] = set()
    while queue:
        identifier = queue.pop(0)
        if identifier in processed:
            continue
        processed.add(identifier)
        record = functions[identifier]
        module = modules[record.module_name]
        for call in calls_for_scope(record.node):
            site = classify_call(
                call,
                identifier,
                module,
                record,
                modules,
                functions,
                callable_aliases,
            )
            if site is None:
                continue
            call_sites.append(site)
            if site.resolved_target and site.resolved_target not in reachable:
                reachable.add(site.resolved_target)
                queue.append(site.resolved_target)
            for reference_site in higher_order_project_function_sites(
                call,
                identifier,
                module,
                record,
                functions,
                callable_aliases,
            ):
                call_sites.append(reference_site)
                if reference_site.resolved_target not in reachable:
                    reachable.add(reference_site.resolved_target)
                    queue.append(reference_site.resolved_target)
    return reachable, call_sites


class NestedDefinitionStripper(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return ast.copy_location(ast.Pass(), node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return ast.copy_location(ast.Pass(), node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return ast.copy_location(ast.Pass(), node)


def isolated_function_source(record: FunctionRecord) -> str:
    node = copy.deepcopy(record.node)
    stripper = NestedDefinitionStripper()
    cleaned_body: list[ast.stmt] = []
    for statement in node.body:
        transformed = stripper.visit(statement)
        if transformed is None:
            continue
        if isinstance(transformed, list):
            cleaned_body.extend(transformed)
        else:
            cleaned_body.append(transformed)
    node.body = cleaned_body or [ast.Pass()]
    node.decorator_list = []
    ast.fix_missing_locations(node)
    return ast.unparse(node) + "\n"


def function_standard_complexity(record: FunctionRecord) -> int:
    require_radon()
    visitor = ComplexityVisitor.from_code(isolated_function_source(record))
    if not visitor.functions:
        return 1
    return int(visitor.functions[0].complexity)


def function_halstead(record: FunctionRecord) -> dict[str, float]:
    require_radon()
    result = h_visit(isolated_function_source(record))
    report = result.functions[0][1] if result.functions else result.total
    return halstead_to_dict(report)


def halstead_to_dict(report: Any) -> dict[str, float]:
    return {field: round(float(getattr(report, field)), 6) for field in HALSTEAD_FIELDS}


def synthetic_resolved_project_source(
    root_module: SourceModule,
    reachable_external: Iterable[FunctionRecord],
) -> str:
    parts = [root_module.source.rstrip(), ""]
    for record in sorted(reachable_external, key=lambda item: item.identifier):
        parts.append(isolated_function_source(record).rstrip())
        parts.append("")
    return "\n".join(parts)


def synthetic_function_scope_source(records: Iterable[FunctionRecord]) -> str:
    parts: list[str] = []
    for record in sorted(records, key=lambda item: item.identifier):
        parts.append(isolated_function_source(record).rstrip())
        parts.append("")
    return "\n".join(parts)


def analyze_software_complexity(
    root_path: Path,
    start_path: str = "jsfm/sfm.py",
    start_function: str | None = None,
) -> dict[str, Any]:
    require_radon()
    root_path = root_path.resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Root path does not exist: {root_path}")
    start_file_path = (root_path / start_path).resolve()
    try:
        start_file_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"Start path escapes root path: {start_path}") from exc
    if not start_file_path.is_file():
        raise FileNotFoundError(f"Start Python file does not exist: {start_path}")

    modules = load_modules(root_path)
    root_module_name, _ = module_name_for_path(root_path, start_file_path)
    if root_module_name not in modules:
        raise RuntimeError(f"Unable to index start module: {start_path}")
    root_module = modules[root_module_name]
    functions = collect_functions(modules)
    collect_callable_aliases(modules, functions)
    callable_aliases = combined_callable_aliases(modules)
    root_function_identifier = None
    if start_function:
        root_function_identifier = f"{root_module.name}.{start_function}"
        if root_function_identifier not in functions:
            raise ValueError(
                f"Start function does not exist in {start_path}: {start_function}"
            )
    reachable, call_sites = build_resolved_project_scope(
        root_module,
        modules,
        functions,
        callable_aliases,
        root_function_identifier,
    )

    root_function_ids = (
        {root_function_identifier}
        if root_function_identifier is not None
        else {
            identifier
            for identifier, record in functions.items()
            if record.module_name == root_module.name
        }
    )
    helper_function_ids = reachable - root_function_ids
    external_function_ids = {
        identifier
        for identifier in helper_function_ids
        if functions[identifier].module_name != root_module.name
    }
    external_records = [functions[identifier] for identifier in sorted(external_function_ids)]

    function_rows: list[dict[str, Any]] = []
    for identifier in sorted(reachable):
        record = functions[identifier]
        standard_cc = function_standard_complexity(record)
        function_rows.append(
            {
                "identifier": identifier,
                "scope": "start_function" if identifier in root_function_ids and start_function else (
                    "start_path" if identifier in root_function_ids else "reachable_helper"
                ),
                "file": record.relative_file,
                "line": record.line,
                "end_line": record.end_line,
                "qualname": record.qualname,
                "standard_cyclomatic_complexity": standard_cc,
                "halstead": function_halstead(record),
            }
        )

    if root_function_identifier is not None:
        root_records = [functions[root_function_identifier]]
        direct_source = synthetic_function_scope_source(root_records)
        resolved_project_source = synthetic_function_scope_source(
            functions[identifier] for identifier in reachable
        )
        module_standard = 0
    else:
        direct_source = root_module.source
        resolved_project_source = synthetic_resolved_project_source(root_module, external_records)
        module_standard = int(ComplexityVisitor.from_code(root_module.source).complexity)
    direct_halstead = h_visit(direct_source)
    resolved_project_halstead = h_visit(resolved_project_source)
    files_in_resolved_project_scope = sorted(
        {root_module.relative_path, *[record.relative_file for record in external_records]}
    )
    unresolved_project_calls = [
        asdict(site) for site in call_sites if site.classification == "unresolved_project_call"
    ]
    resolved_edges = sorted(
        {
            (site.source, site.resolved_target, site.file, site.line)
            for site in call_sites
            if site.resolved_target
        }
    )

    root_function_standard = sum(
        row["standard_cyclomatic_complexity"]
        for row in function_rows
        if row["scope"] in {"start_path", "start_function"}
    )
    resolved_project_function_standard = sum(
        row["standard_cyclomatic_complexity"] for row in function_rows
    )
    direct_standard = module_standard + root_function_standard
    resolved_project_standard = module_standard + resolved_project_function_standard
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root_path": str(root_path),
        "start_path": root_module.relative_path,
        "start_function": start_function,
        "scope": (
            "start function plus statically resolved project-defined function calls"
            if start_function
            else "start path plus statically resolved project-defined function calls"
        ),
        "methodology": {
            "standard_cyclomatic": (
                "Sum of Radon complexity for each uniquely included function; "
                "module-level complexity is included only in file-root mode"
            ),
            "halstead": "Radon metrics.h_visit",
            "call_graph": (
                "Static AST resolution of same-file functions, imported project functions, "
                "class methods, nested functions, and common jit/vmap-style callable aliases"
            ),
            "deduplication": "Each reachable project-defined function is included once.",
            "external_libraries": "Library internals are not analyzed.",
        },
        "files_indexed": len(modules),
        "files_in_resolved_project_scope": files_in_resolved_project_scope,
        "start_path_only": {
            "function_count": len(root_function_ids),
            "module_level_standard_cyclomatic_complexity": module_standard,
            "function_standard_cyclomatic_complexity_sum": root_function_standard,
            "standard_cyclomatic_complexity": direct_standard,
            "halstead": halstead_to_dict(direct_halstead.total),
        },
        "resolved_project_scope": {
            "function_count": len(reachable),
            "reachable_external_function_count": len(external_function_ids),
            "reachable_helper_function_count": len(helper_function_ids),
            "module_level_standard_cyclomatic_complexity": module_standard,
            "function_standard_cyclomatic_complexity_sum": resolved_project_function_standard,
            "standard_cyclomatic_complexity": resolved_project_standard,
            "halstead": halstead_to_dict(resolved_project_halstead.total),
        },
        "functions": function_rows,
        "call_edges": [
            {
                "source": source,
                "target": target,
                "file": file,
                "line": line,
            }
            for source, target, file, line in resolved_edges
        ],
        "unresolved_project_calls": unresolved_project_calls,
    }


def render_text_report(result: dict[str, Any]) -> str:
    direct = result["start_path_only"]
    resolved_scope = result["resolved_project_scope"]
    lines = [
        "Software Complexity Report",
        "",
        f"Start path: {result['start_path']}",
        f"Start function: {result.get('start_function') or 'all functions in start path'}",
        f"Root path: {result['root_path']}",
        "",
        "Start function only" if result.get("start_function") else "Start path only",
        f"  Cyclomatic complexity: {direct['standard_cyclomatic_complexity']}",
        f"  Halstead effort: {direct['halstead']['effort']:.3f}",
        "",
        "Total",
        f"  Cyclomatic complexity: {resolved_scope['standard_cyclomatic_complexity']}",
        f"  Halstead effort: {resolved_scope['halstead']['effort']:.3f}",
        f"  Reachable helper functions: {resolved_scope['reachable_helper_function_count']}",
        f"  Files used: {', '.join(result['files_in_resolved_project_scope'])}",
        "",
        "Functions counted",
    ]
    for row in result["functions"]:
        lines.append(
            f"  {row['identifier']} | CC={row['standard_cyclomatic_complexity']} | "
            f"Halstead effort={row['halstead']['effort']:.3f}"
        )
    lines.extend(
        [
            "",
            f"Resolved project call edges: {len(result['call_edges'])}",
            f"Unresolved project calls: {len(result['unresolved_project_calls'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "software_complexity_report.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (output_dir / "software_complexity_report.txt").write_text(
        render_text_report(result),
        encoding="utf-8",
    )
    csv_path = output_dir / "software_complexity_functions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "identifier",
            "scope",
            "file",
            "line",
            "end_line",
            "standard_cyclomatic_complexity",
            *[f"halstead_{field}" for field in HALSTEAD_FIELDS],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["functions"]:
            flat = {key: row[key] for key in fieldnames if key in row}
            flat.update(
                {
                    f"halstead_{field}": row["halstead"][field]
                    for field in HALSTEAD_FIELDS
                }
            )
            writer.writerow(flat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Radon Halstead and cyclomatic complexity from a start path "
            "and its transitive project-defined function calls inside a root path."
        )
    )
    parser.add_argument(
        "--root-path",
        required=True,
        dest="root_path",
        type=Path,
        help="Root directory of the generated model or project.",
    )
    parser.add_argument(
        "--start-path",
        default="jsfm/sfm.py",
        dest="start_path",
        help="Start Python file relative to --root-path (default: jsfm/sfm.py).",
    )
    parser.add_argument(
        "--start-function",
        dest="start_function",
        help="Optional function name inside --start-path used as the analysis root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for JSON, text, and CSV reports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = analyze_software_complexity(
            args.root_path,
            args.start_path,
            args.start_function,
        )
    except (FileNotFoundError, RuntimeError, SyntaxError, ValueError) as exc:
        terminal_print(str(exc), file=sys.stderr)
        return 2

    terminal_print(render_text_report(result).rstrip())
    if args.output_dir:
        write_outputs(result, args.output_dir)
        terminal_print(f"Reports saved to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
