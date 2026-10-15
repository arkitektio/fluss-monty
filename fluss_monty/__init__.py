"""Flows written as Python source, checked and run in a Monty sandbox.

A flow is a source whose ``async def main(...)`` calls the actions of its manifest by alias.
:func:`validate_python_flow` derives what a flow is stored with (its ports from the
entrypoint's signature, its manifest, the runtime it was checked against) and type-checks it
against stubs of exactly those actions; :func:`arun_python_flow` runs it, each call going
out through a caller you inject.

Nothing here talks to a server: the fluss client runs flows as rekuest tasks with it, and
the fluss server validates what it stores with it.
"""

from fluss_monty.manifest import ManifestAction, ManifestError, build_stubs
from fluss_monty.models import (
    ActionLike,
    Diagnostic,
    ManifestEntry,
    Port,
    PortKind,
    PortLike,
    ValidationReport,
)
from fluss_monty.runner import (
    ActionCallError,
    Caller,
    CallKeys,
    PythonFlowError,
    PythonFlowLimits,
    arun_python_flow,
)
from fluss_monty.validate import RUNTIME, validate_python_flow, validate_python_flow_sync

__all__ = [
    "RUNTIME",
    "ActionCallError",
    "ActionLike",
    "CallKeys",
    "Caller",
    "Diagnostic",
    "ManifestAction",
    "ManifestEntry",
    "ManifestError",
    "Port",
    "PortKind",
    "PortLike",
    "PythonFlowError",
    "PythonFlowLimits",
    "ValidationReport",
    "arun_python_flow",
    "build_stubs",
    "validate_python_flow",
    "validate_python_flow_sync",
]
