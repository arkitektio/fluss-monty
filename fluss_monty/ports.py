"""Python annotations and rekuest ports, both ways.

A Python flow sees raw wire values (its runner bypasses expand/shrink), so a structure is
its ID string. Each structure identifier gets a ``str`` subclass in the stubs --
``@mikro/image`` is ``MikroImage`` -- which only the type checker ever sees: the sandbox
never evaluates annotations.

``annotation_to_port`` reads the entrypoint's signature (host ``ast``, never executed);
``port_to_annotation`` writes the stubs the source is type-checked against. They are
inverses on the kinds both support: INT, FLOAT, STRING, BOOL, LIST, DICT and the
identifier-carrying kinds, each optionally nullable. Any other kind of an action's port is
stubbed as ``Any``; an entrypoint annotation outside that set is a validation error.
"""

import ast
import keyword
import re
from collections.abc import Iterable, Sequence
from typing import Any

from fluss_monty.models import STRUCTURE_KINDS, Port, PortKind, PortLike, kind_of

PRIMITIVES: dict[str, PortKind] = {
    "int": PortKind.INT,
    "float": PortKind.FLOAT,
    "str": PortKind.STRING,
    "bool": PortKind.BOOL,
}
PRIMITIVE_NAMES: dict[str, str] = {kind.value: name for name, kind in PRIMITIVES.items()}

#: The key rekuest gives the item port of a LIST or DICT.
ITEM_KEY = "..."


class AnnotationError(ValueError):
    """An annotation that has no port (or a port that has no annotation)."""

    def __init__(self, message: str, node: ast.AST | None = None) -> None:
        """Keep the line of ``node``, when there is one."""
        super().__init__(message)
        self.line: int | None = getattr(node, "lineno", None)


def structure_class_name(identifier: str) -> str:
    """``@mikro/image`` -> ``MikroImage``: the stub class a structure is typed as."""
    words = re.split(r"[^0-9A-Za-z]+", identifier)
    name = "".join(word[:1].upper() + word[1:] for word in words if word)
    if not name or name[0].isdigit():
        name = f"S{name}"
    return name


def structure_classes(identifiers: Iterable[str]) -> dict[str, str]:
    """Class name -> identifier, for every identifier; two identifiers on one name is an error."""
    classes: dict[str, str] = {}
    for identifier in sorted(set(identifiers)):
        name = structure_class_name(identifier)
        if name in classes and classes[name] != identifier:
            raise AnnotationError(f"Structures {classes[name]} and {identifier} both map to the type {name}")
        classes[name] = identifier
    return classes


def _field(port: Any, name: str) -> Any:  # noqa: ANN401 -- a port field, if the fragment fetched it
    """A port's field, or ``None`` where a GraphQL fragment stopped nesting before it."""
    return getattr(port, name, None)


def port_identifiers(ports: Iterable[PortLike]) -> set[str]:
    """Every structure identifier in ``ports``, nested ones included (as deep as they were fetched)."""
    found: set[str] = set()
    for port in ports:
        identifier = _field(port, "identifier")
        if kind_of(port) in STRUCTURE_KINDS and identifier:
            found.add(identifier)
        found |= port_identifiers(_field(port, "children") or ())
    return found


# --- annotation -> port ------------------------------------------------------------------


def _strip_optional(node: ast.expr) -> tuple[ast.expr, bool]:
    """``T | None`` / ``Optional[T]`` -> (T, True); anything else -> (node, False)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        if _is_none(node.right):
            return node.left, True
        if _is_none(node.left):
            return node.right, True
    if isinstance(node, ast.Subscript) and _name(node.value) == "Optional":
        return node.slice, True
    return node, False


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _name(node: ast.expr) -> str | None:
    """``int`` -> "int", ``typing.Optional`` -> "Optional"."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _parse_string_annotation(node: ast.expr) -> ast.expr:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return ast.parse(node.value, mode="eval").body
        except SyntaxError as error:
            raise AnnotationError(f"Unparseable annotation {node.value!r}", node) from error
    return node


def annotation_to_port(node: ast.expr, key: str, structures: dict[str, str], default: Any = None) -> Port:  # noqa: ANN401 -- a literal
    """The port an annotation describes; ``default`` only for an argument."""
    node, nullable = _strip_optional(_parse_string_annotation(node))
    name = _name(node)

    if name in PRIMITIVES:
        return Port(key=key, kind=PRIMITIVES[name], nullable=nullable, default=default)
    if name in structures:
        return Port(key=key, kind=PortKind.STRUCTURE, identifier=structures[name], nullable=nullable, default=default)
    if isinstance(node, ast.Subscript):
        container = _name(node.value)
        if container in ("list", "List"):
            child = annotation_to_port(node.slice, ITEM_KEY, structures)
            return Port(key=key, kind=PortKind.LIST, nullable=nullable, children=(child,), default=default)
        if container in ("dict", "Dict"):
            if not isinstance(node.slice, ast.Tuple) or len(node.slice.elts) != 2 or _name(node.slice.elts[0]) != "str":
                raise AnnotationError(f"A dict port is dict[str, T]; {key!r} is {ast.unparse(node)}", node)
            child = annotation_to_port(node.slice.elts[1], ITEM_KEY, structures)
            return Port(key=key, kind=PortKind.DICT, nullable=nullable, children=(child,), default=default)
    known = ", ".join([*PRIMITIVES, *sorted(structures)])
    raise AnnotationError(f"{key!r} is annotated {ast.unparse(node)}, which has no port; use {known}, list[T], dict[str, T] or T | None", node)


def signature_to_ports(function: ast.AsyncFunctionDef | ast.FunctionDef, structures: dict[str, str]) -> tuple[list[Port], list[Port]]:
    """The entrypoint's parameters as arg ports and its return annotation as return ports."""
    arguments = function.args
    if arguments.vararg or arguments.kwarg or arguments.posonlyargs:
        raise AnnotationError(f"{function.name}() takes named parameters only (no *args, **kwargs or /)", function)

    positional = arguments.args
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(arguments.defaults)) + list(arguments.defaults)
    parameters = list(zip(positional, defaults)) + list(zip(arguments.kwonlyargs, arguments.kw_defaults))

    args = []
    for parameter, default_node in parameters:
        if parameter.annotation is None:
            raise AnnotationError(f"Parameter {parameter.arg!r} of {function.name}() needs a type annotation", parameter)
        default = None
        if default_node is not None:
            try:
                default = ast.literal_eval(default_node)
            except ValueError as error:
                raise AnnotationError(f"The default of {parameter.arg!r} must be a literal", default_node) from error
        args.append(annotation_to_port(parameter.annotation, parameter.arg, structures, default))

    return args, return_ports(function, structures)


def return_ports(function: ast.AsyncFunctionDef | ast.FunctionDef, structures: dict[str, str]) -> list[Port]:
    """``-> None`` is no port, ``-> tuple[A, B]`` two (``return0``, ``return1``), anything else one."""
    node = function.returns
    if node is None:
        raise AnnotationError(f"{function.name}() needs a return annotation (-> None if it returns nothing)", function)
    node = _parse_string_annotation(node)
    if _is_none(node):
        return []
    if isinstance(node, ast.Subscript) and _name(node.value) in ("tuple", "Tuple"):
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return [annotation_to_port(item, f"return{i}", structures) for i, item in enumerate(items)]
    return [annotation_to_port(node, "return0", structures)]


# --- port -> annotation ------------------------------------------------------------------


def port_to_annotation(port: PortLike) -> str:
    """The stub type of a port's value; ``Any`` for kinds with no Python counterpart here.

    Ports read from GraphQL are only as deep as their fragment: a LIST or DICT whose item was
    not fetched is ``list[Any]``/``dict[str, Any]``'s looser cousin, ``Any``.
    """
    kind = kind_of(port)
    children = list(_field(port, "children") or ())
    identifier = _field(port, "identifier")
    if kind in PRIMITIVE_NAMES:
        annotation = PRIMITIVE_NAMES[kind]
    elif kind in STRUCTURE_KINDS and identifier:
        annotation = structure_class_name(identifier)
    elif kind == PortKind.LIST.value and len(children) == 1:
        annotation = f"list[{port_to_annotation(children[0])}]"
    elif kind == PortKind.DICT.value and len(children) == 1:
        annotation = f"dict[str, {port_to_annotation(children[0])}]"
    else:
        return "Any"
    return f"{annotation} | None" if _field(port, "nullable") else annotation


def returns_to_annotation(ports: Sequence[PortLike]) -> str:
    """The stub return type: ``None``, the one value, or a tuple in port order."""
    if not ports:
        return "None"
    if len(ports) == 1:
        return port_to_annotation(ports[0])
    return f"tuple[{', '.join(port_to_annotation(port) for port in ports)}]"


def is_parameter_name(key: str) -> bool:
    """Whether ``key`` can be a Python parameter."""
    return key.isidentifier() and not keyword.iskeyword(key)
