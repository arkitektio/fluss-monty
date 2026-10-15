"""The actions a Python flow may call: their stubs, their manifest entries, their call convention.

Each action is injected under an ``alias`` as an ``async`` function. The stub the source is
type-checked against and the host function the runner binds are built from the same ports,
so they agree on one convention:

* arg ports in order are the parameters; required ones may be passed positionally until the
  first optional one, after which every parameter is keyword-only;
* 0 return ports is ``None``, 1 is the value, several are a tuple in port order.

The stubs declare nothing but these functions and the structure classes, so a name outside
the manifest is an unresolved reference to the type checker: the stubs are also the
permission boundary the runner enforces by only binding what they declare.
"""

import dataclasses
from collections.abc import Iterable, Sequence
from typing import Any

from fluss_monty.models import ActionLike, ArgPortLike, ManifestEntry, kind_of
from fluss_monty.ports import (
    is_parameter_name,
    port_identifiers,
    port_to_annotation,
    returns_to_annotation,
    structure_classes,
)


@dataclasses.dataclass(frozen=True)
class ManifestAction:
    """One action a flow may call, under the name the source calls it by.

    ``effect`` (rekuest's ``Effects``) is declared by whoever builds the manifest: a rekuest
    ``Action`` does not carry its implementations' effects, so a flow's ``physical`` is only as
    honest as this.
    """

    alias: str
    action: ActionLike
    effect: str = "UNKNOWN"
    app: str | None = None
    key: str | None = None
    version: str | None = None


class ManifestError(ValueError):
    """An action that cannot be offered to a Python flow."""


def _has_default(port: ArgPortLike) -> bool:
    return bool(getattr(port, "nullable", None)) or getattr(port, "default", None) is not None


def check_action(entry: ManifestAction) -> None:
    """Raise if the action cannot be called from a flow under this alias."""
    if not is_parameter_name(entry.alias):
        raise ManifestError(f"The alias {entry.alias!r} is not a Python identifier")
    if kind_of(entry.action) != "FUNCTION":
        raise ManifestError(f"{entry.alias} is a {entry.action.kind} action; a Python flow can only call FUNCTION actions")
    for port in entry.action.args:
        if not is_parameter_name(port.key):
            raise ManifestError(f"{entry.alias} has an argument {port.key!r} that is not a Python identifier")


#: Names an alias would shadow: the stubs' own, and what flows import.
RESERVED_ALIASES = frozenset({"Any", "asyncio", "typing", "json", "math", "re"})


def check_manifest(entries: Sequence[ManifestAction], reserved: Iterable[str] = ()) -> None:
    """Raise unless every alias is unique, not ``reserved``, and every action callable from a flow."""
    aliases = [entry.alias for entry in entries]
    duplicates = sorted({alias for alias in aliases if aliases.count(alias) > 1})
    if duplicates:
        raise ManifestError(f"Aliases must be unique; duplicated: {', '.join(duplicates)}")
    shadowing = sorted(set(aliases) & (RESERVED_ALIASES | set(reserved) | set(structure_classes(manifest_identifiers(entries)))))
    if shadowing:
        raise ManifestError(f"Aliases would shadow names the flow needs: {', '.join(shadowing)}")
    for entry in entries:
        check_action(entry)


def positional_keys(action: ActionLike) -> list[str]:
    """The arg keys that may be passed positionally: the leading required ones."""
    keys = []
    for port in action.args:
        if _has_default(port):
            break
        keys.append(port.key)
    return keys


def action_stub(entry: ManifestAction) -> str:
    """``async def alias(image: MikroImage, *, sigma: float | None = ...) -> MikroImage: ...``"""
    positional = set(positional_keys(entry.action))
    parameters: list[str] = []
    keyword_only = False
    for port in entry.action.args:
        if port.key not in positional and not keyword_only:
            parameters.append("*")
            keyword_only = True
        default = " = ..." if _has_default(port) else ""
        parameters.append(f"{port.key}: {port_to_annotation(port)}{default}")
    returns = returns_to_annotation(entry.action.returns)
    return f"async def {entry.alias}({', '.join(parameters)}) -> {returns}: ..."


def manifest_identifiers(entries: Iterable[ManifestAction]) -> set[str]:
    """Every structure identifier the actions' ports use."""
    found: set[str] = set()
    for entry in entries:
        found |= port_identifiers(entry.action.args) | port_identifiers(entry.action.returns)
    return found


def build_stubs(entries: Sequence[ManifestAction], identifiers: Iterable[str] = ()) -> tuple[str, dict[str, str]]:
    """The stub source for these actions, and the structure class name -> identifier map.

    ``identifiers`` adds structures the entrypoint uses that no action mentions.
    """
    classes = structure_classes(manifest_identifiers(entries) | set(identifiers))
    lines = ["from typing import Any", ""]
    lines += [f"class {name}(str):\n    \"\"\"A {identifier} (its ID).\"\"\"\n" for name, identifier in classes.items()]
    lines += [action_stub(entry) for entry in entries]
    return "\n".join(lines) + "\n", classes


def manifest_entries(entries: Sequence[ManifestAction]) -> list[ManifestEntry]:
    """The manifest a flow is stored with."""
    return [
        ManifestEntry(alias=entry.alias, action_hash=str(entry.action.hash), effect=str(getattr(entry.effect, "value", entry.effect)), app=entry.app, key=entry.key, version=entry.version)
        for entry in entries
    ]


def bind_arguments(entry: ManifestAction, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """A call's positional and keyword arguments as the action's kwargs (the stub's convention)."""
    positional = positional_keys(entry.action)
    if len(args) > len(positional):
        raise TypeError(f"{entry.alias}() takes {len(positional)} positional arguments but {len(args)} were given")
    bound = dict(zip(positional, args))
    known = {port.key for port in entry.action.args}
    for key, value in kwargs.items():
        if key not in known:
            raise TypeError(f"{entry.alias}() got an unexpected keyword argument {key!r}")
        if key in bound:
            raise TypeError(f"{entry.alias}() got multiple values for argument {key!r}")
        bound[key] = value
    return bound


def unwrap_returns(entry: ManifestAction, returns: dict[str, Any] | None) -> Any:  # noqa: ANN401 -- whatever the action returns
    """An action's returns dict as the stub's return value."""
    ports = entry.action.returns
    values = [(returns or {}).get(port.key) for port in ports]
    if not ports:
        return None
    if len(ports) == 1:
        return values[0]
    return tuple(values)
