"""Validate a Python flow: the report ``createPythonFlow`` stores.

Two passes. The host parses the source with ``ast`` (never executing it) to check its shape
and read the entrypoint's signature into ports. Then Monty type-checks it against the stubs
of the manifest's actions -- which is also what rejects a call to anything the manifest does
not offer, as an unresolved name.

Feeding the source defines its functions, so its module body may only import, define, and
bind literals. No action can be called before the entrypoint (the host binds them only then),
but feeding does run what a definition evaluates -- class bodies, decorators, defaults -- in
the sandbox and within its limits.
"""

import ast
import re
from collections.abc import Iterable, Sequence

import pydantic_monty

from fluss_monty.manifest import (
    ManifestAction,
    ManifestError,
    build_stubs,
    check_manifest,
    manifest_entries,
)
from fluss_monty.models import Diagnostic, Port, ValidationReport
from fluss_monty.ports import AnnotationError, signature_to_ports

#: The version of the helpers a flow may use beside its manifest (none yet but the stdlib Monty ships).
HELPERS_VERSION = "1"
#: What a validation is valid for: the interpreter and the helpers the source was checked against.
RUNTIME = f"monty-{pydantic_monty.__version__}/helpers-{HELPERS_VERSION}"

SCRIPT_NAME = "flow.py"

#: What checking a source may use. Feeding it defines its functions, and so runs what a
#: definition evaluates (class bodies, decorators, defaults): on a server, that is code
#: someone else wrote. Defining functions takes milliseconds; a second is a generous bound on
#: how long one check may hold a (synchronous) server thread.
CHECK_LIMITS: pydantic_monty.ResourceLimits = {
    "max_suspensions": 0,
    "max_memory": 64 * 1024 * 1024,
    "max_feed_duration_secs": 1.0,
    "max_turn_duration_secs": 1.0,
    "max_recursion_depth": 200,
}


class SourceError(ValueError):
    """The source's shape is wrong (before any type checking)."""

    def __init__(self, message: str, line: int | None = None) -> None:
        """``line`` is where in the source, when known."""
        super().__init__(message)
        self.line = line


def _is_literal(node: ast.expr | None) -> bool:
    if node is None:
        return True
    try:
        ast.literal_eval(node)
    except ValueError:
        return False
    return True


def check_module(module: ast.Module) -> None:
    """Only imports, definitions, a docstring and literal bindings at module level."""
    for index, node in enumerate(module.body):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Expr) and index == 0 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets) and _is_literal(node.value):
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and _is_literal(node.value):
            continue
        raise SourceError(
            f"Only imports, functions, classes and literal constants may appear at module level, not `{ast.unparse(node).splitlines()[0]}`",
            node.lineno,
        )


def find_entrypoint(module: ast.Module, entrypoint: str) -> ast.AsyncFunctionDef:
    """The ``async def`` the run calls."""
    for node in module.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == entrypoint:
            return node
        if isinstance(node, ast.FunctionDef) and node.name == entrypoint:
            raise SourceError(f"{entrypoint}() must be `async def`, so it can await the actions it calls", node.lineno)
    raise SourceError(f"The source defines no `async def {entrypoint}(...)`")


def analyse(source: str, entrypoint: str, structures: dict[str, str]) -> tuple[list[Port], list[Port]]:
    """Shape-check the source and read its entrypoint's ports (host side, no execution)."""
    try:
        module = ast.parse(source, filename=SCRIPT_NAME)
    except SyntaxError as error:
        raise SourceError(f"Syntax error: {error.msg}", error.lineno) from error
    check_module(module)
    function = find_entrypoint(module, entrypoint)
    try:
        return signature_to_ports(function, structures)
    except AnnotationError as error:
        raise SourceError(str(error), error.line) from error


#: One concise type-check diagnostic: ``flow.py:LINE:COLUMN: message``.
_CONCISE = re.compile(rf"^{re.escape(SCRIPT_NAME)}:(\d+):\d+: (.+)$")


def _diagnostics(error: pydantic_monty.MontyError) -> list[Diagnostic]:
    """A check's failure as one diagnostic per problem, each on its line where the checker says."""
    if isinstance(error, pydantic_monty.MontyRuntimeError):
        frames = error.traceback()  # a session names its feeds `<python-input-N>`: the source is the only one
        return [Diagnostic(message=error.display("type-msg"), line=frames[-1].line if frames else None)]
    display = getattr(error, "display", None)
    text = str(display() if callable(display) else error)
    found = [match for match in (_CONCISE.match(line) for line in text.splitlines()) if match]
    if not found:
        return [Diagnostic(message=text.strip())]
    return [Diagnostic(message=match.group(2), line=int(match.group(1))) for match in found]


def _prepare(
    source: str, actions: Sequence[ManifestAction], entrypoint: str, structures: Iterable[str]
) -> tuple[ValidationReport, tuple[list[Port], list[Port]] | None]:
    """The report so far, and the ports when the source's shape is right (no type check yet)."""
    report = ValidationReport(ok=False, entrypoint=entrypoint, runtime=RUNTIME)
    try:
        check_manifest(actions, reserved=[entrypoint])
        report.stubs, classes = build_stubs(actions, structures)
        return report, analyse(source, entrypoint, classes)
    except (ManifestError, AnnotationError) as error:
        report.diagnostics.append(Diagnostic(message=str(error)))
    except SourceError as error:
        report.diagnostics.append(Diagnostic(message=str(error), line=error.line))
    return report, None


def _finish(report: ValidationReport, ports: tuple[list[Port], list[Port]], actions: Sequence[ManifestAction], diagnostics: list[Diagnostic]) -> ValidationReport:
    if diagnostics:
        report.diagnostics.extend(diagnostics)
        return report
    report.ok = True
    report.args, report.returns = ports
    report.manifest = manifest_entries(actions)
    return report


async def validate_python_flow(
    source: str,
    actions: Sequence[ManifestAction] = (),
    *,
    entrypoint: str = "main",
    structures: Iterable[str] = (),
    pool: pydantic_monty.AsyncMonty | None = None,
    limits: pydantic_monty.ResourceLimits = CHECK_LIMITS,
) -> ValidationReport:
    """Check ``source`` against the ``actions`` it may call; derive its ports and manifest.

    ``structures`` are identifiers the entrypoint's signature uses that no action mentions.
    Pass a running ``pool`` to reuse its workers; otherwise one is started for this check.
    """
    report, ports = _prepare(source, actions, entrypoint, structures)
    if ports is None:
        return report
    return _finish(report, ports, actions, await type_check(source, report.stubs, pool=pool, limits=limits))


def validate_python_flow_sync(
    source: str,
    actions: Sequence[ManifestAction] = (),
    *,
    entrypoint: str = "main",
    structures: Iterable[str] = (),
    pool: pydantic_monty.Monty | None = None,
    limits: pydantic_monty.ResourceLimits = CHECK_LIMITS,
) -> ValidationReport:
    """:func:`validate_python_flow` for synchronous callers (a Django mutation)."""
    report, ports = _prepare(source, actions, entrypoint, structures)
    if ports is None:
        return report
    return _finish(report, ports, actions, type_check_sync(source, report.stubs, pool=pool, limits=limits))


_CHECK_ERRORS = (pydantic_monty.MontyTypingError, pydantic_monty.MontySyntaxError, pydantic_monty.MontyRuntimeError)


def _checkout_options(stubs: str, limits: pydantic_monty.ResourceLimits) -> dict:
    return {"script_name": SCRIPT_NAME, "limits": limits, "type_check": True, "type_check_stubs": stubs, "type_check_format": "concise"}


async def type_check(
    source: str, stubs: str, *, pool: pydantic_monty.AsyncMonty | None = None, limits: pydantic_monty.ResourceLimits = CHECK_LIMITS
) -> list[Diagnostic]:
    """Type-check (and define) the source in a fresh session; the problems, if any."""
    if pool is None:
        async with pydantic_monty.AsyncMonty(min_processes=1, max_processes=1) as own_pool:
            return await type_check(source, stubs, pool=own_pool, limits=limits)
    async with pool.checkout(**_checkout_options(stubs, limits)) as session:
        try:
            await session.feed_run(source)
        except _CHECK_ERRORS as error:
            return _diagnostics(error)
    return []


def type_check_sync(
    source: str, stubs: str, *, pool: pydantic_monty.Monty | None = None, limits: pydantic_monty.ResourceLimits = CHECK_LIMITS
) -> list[Diagnostic]:
    """:func:`type_check` for synchronous callers."""
    if pool is None:
        with pydantic_monty.Monty(min_processes=1, max_processes=1) as own_pool:
            return type_check_sync(source, stubs, pool=own_pool, limits=limits)
    with pool.checkout(**_checkout_options(stubs, limits)) as session:
        try:
            session.feed_run(source)
        except _CHECK_ERRORS as error:
            return _diagnostics(error)
    return []
