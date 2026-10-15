"""The ports, manifest entries and kinds a Python flow is described by.

These are plain models of the wire shapes: whoever stores or registers a flow (the fluss
client, the fluss server) converts them to its own port types with ``model_dump``. Anything
port-shaped is read through :class:`PortLike`, so those types are accepted back as they are.
"""

from collections.abc import Sequence
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field


class PortKind(str, Enum):
    """The port kinds a flow's own signature can have (an action's ports may have others)."""

    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    BOOL = "BOOL"
    LIST = "LIST"
    DICT = "DICT"
    STRUCTURE = "STRUCTURE"
    MEMORY_STRUCTURE = "MEMORY_STRUCTURE"
    INTERFACE = "INTERFACE"


#: Kinds that carry a structure ``identifier`` (``@package/key``).
STRUCTURE_KINDS = frozenset({PortKind.STRUCTURE.value, PortKind.MEMORY_STRUCTURE.value, PortKind.INTERFACE.value})


class HasKind(Protocol):
    """A port or an action: anything with a ``kind``."""

    @property
    def kind(self) -> Any: ...  # noqa: ANN401, D102 -- an enum or its string


def kind_of(item: HasKind) -> str:
    """A port's (or action's) kind as its wire string, whatever enum (or string) carries it."""
    return str(getattr(item.kind, "value", item.kind))


class Port(BaseModel):
    """One argument or return of a flow, as its entrypoint's signature declares it."""

    key: str
    kind: PortKind
    nullable: bool = False
    identifier: str | None = None
    default: Any = None
    """Only on arguments: the literal the parameter defaults to."""
    children: tuple["Port", ...] = ()
    """The one item port of a LIST or DICT, keyed ``...``."""


class ManifestEntry(BaseModel):
    """One action a flow may call, as it is stored: by the hash of the action."""

    alias: str
    action_hash: str
    effect: str = "UNKNOWN"
    """rekuest's ``Effects``: what calling it again would do to the world."""
    app: str | None = None
    key: str | None = None
    version: str | None = None


class Diagnostic(BaseModel):
    """One problem with the source."""

    message: str
    line: int | None = None


class ValidationReport(BaseModel):
    """What validation derived from a source; store it when ``ok``."""

    ok: bool
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    entrypoint: str = "main"
    args: list[Port] = Field(default_factory=list)
    returns: list[Port] = Field(default_factory=list)
    manifest: list[ManifestEntry] = Field(default_factory=list)
    runtime: str = ""
    stubs: str = ""


class PortLike(Protocol):
    """What is read of a port: these models, rekuest's ``ArgPort``/``ReturnPort`` and their inputs alike.

    Read-only, so any port model satisfies it; ``kind`` is compared by value (:func:`kind_of`).
    """

    @property
    def key(self) -> str: ...  # noqa: D102
    @property
    def kind(self) -> Any: ...  # noqa: ANN401, D102 -- an enum or its string
    @property
    def nullable(self) -> bool | None: ...  # noqa: D102
    @property
    def identifier(self) -> str | None: ...  # noqa: D102
    @property
    def children(self) -> Sequence[Any] | None: ...  # noqa: D102 -- nested ports (fragments stop nesting at some depth)


class ArgPortLike(PortLike, Protocol):
    """An arg port: a port that may carry a default."""

    @property
    def default(self) -> Any: ...  # noqa: ANN401, D102 -- any JSON value


class ActionLike(Protocol):
    """What a flow needs of a rekuest ``Action``: its hash, kind and ports."""

    @property
    def hash(self) -> str: ...  # noqa: D102
    @property
    def kind(self) -> Any: ...  # noqa: ANN401, D102 -- ActionKind or its string
    @property
    def args(self) -> Sequence[ArgPortLike]: ...  # noqa: D102
    @property
    def returns(self) -> Sequence[PortLike]: ...  # noqa: D102
