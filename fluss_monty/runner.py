"""Run a Python flow in Monty.

Every run type-checks the source again against the stubs of its manifest before calling the
entrypoint, in the same session: what a run may call is decided by the manifest it is given
now, not by what a validation once reported. Only the manifest's actions are bound, as
``async`` host functions that go out through ``call`` -- in a rekuest task, each one is a
child task of the run.

A run is a workflow: when its agent dies it is run again from the start, and each call it
already made is found again by its call key. Gathered calls reach the host in whatever order
their siblings' results allow, so a key cannot count calls in arrival order -- two calls of
one action could trade keys, and each would be answered with the other's result. A key is
the action, its arguments, and how often that exact call was made before: interleaving can
only swap identical calls, and a call whose arguments changed is a new one.
"""

import asyncio
import collections
import dataclasses
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import pydantic_monty

from fluss_monty.manifest import (
    ManifestAction,
    bind_arguments,
    build_stubs,
    check_manifest,
    unwrap_returns,
)
from fluss_monty.models import PortLike, kind_of
from fluss_monty.ports import port_identifiers
from fluss_monty.validate import SCRIPT_NAME, SourceError, analyse

logger = logging.getLogger(__name__)

#: ``call(action, kwargs, call_key)`` -> the action's returns dict.
Caller = Callable[[ManifestAction, dict[str, Any], str], Awaitable[Mapping[str, Any] | None]]
PrintCallback = Callable[[str, str], None]

_KWARGS = "__flow_kwargs__"


@dataclasses.dataclass(frozen=True)
class PythonFlowLimits:
    """What one run may use. The durations count sandbox time only, never time spent in actions."""

    max_suspensions: int = 10_000
    """Action calls (and other host round trips) per run."""
    max_memory: int | None = 512 * 1024 * 1024
    max_feed_duration_secs: float | None = 600.0
    max_turn_duration_secs: float | None = 60.0
    max_recursion_depth: int | None = None

    def resource_limits(self) -> pydantic_monty.ResourceLimits:
        """These limits as Monty takes them."""
        limits: pydantic_monty.ResourceLimits = {"max_suspensions": self.max_suspensions}
        for field in ("max_memory", "max_feed_duration_secs", "max_turn_duration_secs", "max_recursion_depth"):
            value = getattr(self, field)
            if value is not None:
                limits[field] = value  # type: ignore[literal-required]
        return limits


class PythonFlowError(Exception):
    """The flow did not produce its returns: it failed its checks, raised, or returned the wrong shape."""

    def __init__(self, message: str, *, traceback: str | None = None) -> None:
        """``traceback`` is the sandbox's, when the flow raised."""
        super().__init__(message)
        self.traceback = traceback


class ActionCallError(RuntimeError):
    """An action the flow called failed; raised inside the sandbox, where the flow may catch it."""


class CallKeys:
    """The call keys of one run: ``alias:digest:n``, the ``n``-th call of that alias with those arguments."""

    def __init__(self) -> None:
        """No call made yet."""
        self._made: collections.Counter[tuple[str, str]] = collections.Counter()

    def next(self, alias: str, kwargs: Mapping[str, Any]) -> str:
        """The key of this call (wire values, so JSON is their canonical form)."""
        canonical = json.dumps(kwargs, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        self._made[(alias, digest)] += 1
        return f"{alias}:{digest}:{self._made[(alias, digest)]}"


def _host_function(entry: ManifestAction, call: Caller, keys: CallKeys, inflight: set["asyncio.Task[Any]"]) -> Callable[..., Awaitable[Any]]:
    async def host(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- the stub types them
        bound = bind_arguments(entry, args, kwargs)
        call_key = keys.next(entry.alias, bound)
        current = asyncio.current_task()
        if current is not None:
            inflight.add(current)
        try:
            returns = await call(entry, bound, call_key)
        except Exception as error:
            raise ActionCallError(f"{entry.alias} failed: {error}") from error
        finally:
            inflight.discard(current)  # type: ignore[arg-type]
        return unwrap_returns(entry, dict(returns) if returns is not None else None)

    host.__name__ = entry.alias
    return host


def _signature(ports: Sequence[PortLike]) -> list[tuple[str, str, str | None, bool]]:
    """What must agree between a stored and a derived port list (top level)."""
    return [(port.key, kind_of(port), getattr(port, "identifier", None), bool(getattr(port, "nullable", False))) for port in ports]


def pack_returns(value: Any, ports: Sequence[PortLike]) -> dict[str, Any]:  # noqa: ANN401 -- what the entrypoint returned
    """The entrypoint's return value as the returns dict (the inverse of the stub convention)."""
    if not ports:
        return {}
    if len(ports) == 1:
        return {ports[0].key: value}
    if not isinstance(value, (tuple, list)) or len(value) != len(ports):
        raise PythonFlowError(f"The entrypoint must return {len(ports)} values, got {value!r}")
    return {port.key: item for port, item in zip(ports, value)}


async def _acancel_inflight(inflight: set["asyncio.Task[Any]"]) -> None:
    """Cancel the calls still in flight and wait until each has ended."""
    pending = list(inflight)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.shield(asyncio.gather(*pending, return_exceptions=True))


async def arun_python_flow(
    source: str,
    kwargs: Mapping[str, Any],
    actions: Sequence[ManifestAction],
    call: Caller,
    *,
    entrypoint: str = "main",
    args: Sequence[PortLike] = (),
    returns: Sequence[PortLike] = (),
    pool: pydantic_monty.AsyncMonty | None = None,
    limits: PythonFlowLimits | None = None,
    print_callback: PrintCallback | None = None,
    os_policy: pydantic_monty.OSPolicy | None = None,
) -> dict[str, Any]:
    """Run ``entrypoint(**kwargs)`` of ``source`` with ``actions`` bound; its returns by port key.

    ``args``/``returns`` are the flow's stored ports: the structures they name are typed in the
    stubs, and ``returns`` names the keys of the result. ``os_policy`` is the sandbox's clock and
    ``random`` policy: a fixed seed makes a run made again make the same calls (and find them).
    """
    if pool is None:
        async with pydantic_monty.AsyncMonty(min_processes=1, max_processes=1) as own_pool:
            return await arun_python_flow(
                source,
                kwargs,
                actions,
                call,
                entrypoint=entrypoint,
                args=args,
                returns=returns,
                pool=own_pool,
                limits=limits,
                print_callback=print_callback,
                os_policy=os_policy,
            )

    check_manifest(actions, reserved=[entrypoint])
    stubs, classes = build_stubs(actions, port_identifiers(args) | port_identifiers(returns))
    # The run's ports are the source's, derived again: stored ones only have to agree with them.
    try:
        derived_args, derived_returns = analyse(source, entrypoint, classes)
    except SourceError as error:
        raise PythonFlowError(f"The flow's source does not validate: {error}") from error
    for stored, derived, side in ((args, derived_args, "arguments"), (returns, derived_returns, "returns")):
        if stored and _signature(stored) != _signature(derived):
            raise PythonFlowError(f"The flow's stored {side} {_signature(stored)} do not match its source's {_signature(derived)}")
    keys = CallKeys()
    # Monty leaves the calls in flight when a feed ends early (cancelled, or the flow raised
    # past them): the run cancels them itself, so no child task outlives its run.
    inflight: set[asyncio.Task[Any]] = set()
    external_lookup = {entry.alias: _host_function(entry, call, keys, inflight) for entry in actions}
    printer = print_callback or (lambda stream, text: logger.info("[%s] %s", stream, text.rstrip("\n")))

    async with pool.checkout(
        script_name=SCRIPT_NAME,
        limits=(limits or PythonFlowLimits()).resource_limits(),
        type_check=True,
        type_check_stubs=stubs,
        type_check_format="concise",
        os_policy=os_policy,
    ) as session:
        try:
            await session.feed_run(source, print_callback=printer)
        except (pydantic_monty.MontyTypingError, pydantic_monty.MontySyntaxError) as error:
            raise PythonFlowError(f"The flow does not check against its manifest:\n{error.display()}") from error
        except pydantic_monty.MontyRuntimeError as error:
            raise PythonFlowError(str(error), traceback=error.display()) from error

        try:
            value = await session.feed_run(
                f"await {entrypoint}(**{_KWARGS})",
                inputs={_KWARGS: dict(kwargs)},
                external_lookup=external_lookup,
                print_callback=printer,
                skip_type_check=True,
            )
        except pydantic_monty.MontyRuntimeError as error:
            raise PythonFlowError(str(error), traceback=error.display()) from error
        finally:
            await _acancel_inflight(inflight)

    return pack_returns(value, derived_returns)
