"""Run stunnel with a test configuration, see if it works."""

from __future__ import annotations

import argparse
import asyncio
import asyncio.base_events
import contextlib
import dataclasses
import os
import pathlib
import random
import re
import shlex
import ssl
import subprocess
import sys
import tempfile

from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Union,
    Type,
    TypeVar,
)


PathStr = Union[str, "os.PathLike[Any]"]
PathList = List[PathStr]
_TYPING_USED = (Any,)

VERSION = "0.1.1"

DEFAULT_PROG = pathlib.Path("/usr/bin/stunnel4")

RE_VERSION = re.compile(
    r""" ^
    stunnel \s+
    (?P<version> (?: [5-9] | [1-9][0-9]* ) \. \S+ )
    (?: \s .* )?
    $ """,
    re.X,
)

RE_LINE_IDX = re.compile(r" ^ Hello \s+ (?P<idx> 0 | [1-9][0-9]* ) $ ", re.X)


@dataclasses.dataclass(frozen=True)
class Event:
    """The base class for an event."""

    etype: str


@dataclasses.dataclass(frozen=True)
class FatalEvent(Event):
    """Something really, really bad happened."""

    reason: str


@dataclasses.dataclass(frozen=True)
class ListenerStartedEvent(Event):
    """The listener task succeeded in setting up its socket."""

    hostname: str
    port: int


@dataclasses.dataclass(frozen=True)
class ListenerClientEvent(Event):
    """The listener task handled a connected client event."""

    peer: str


@dataclasses.dataclass(frozen=True)
class ClientConnectedEvent(ListenerClientEvent):
    """The listener task accepted a connection from a client."""


@dataclasses.dataclass(frozen=True)
class ClientDoneEvent(ListenerClientEvent):
    """The listener task closed a connection to a client."""


@dataclasses.dataclass(frozen=True)
class ClientDataEvent(ListenerClientEvent):
    """Some data was transferred to or from the client."""

    line: bytes


@dataclasses.dataclass(frozen=True)
class ClientReceivedDataEvent(ClientDataEvent):
    """A client sent us some data."""

    idx: int


@dataclasses.dataclass(frozen=True)
class ClientSentDataEvent(ClientDataEvent):
    """We sent some data to the client."""


@dataclasses.dataclass(frozen=True)
class ChildReadyEvent(Event):
    """The stunnel process is ready to accept connections."""


@dataclasses.dataclass(frozen=True)
class ConnectionDoneEvent(Event):
    """A test connection was completed."""

    idx: int


@dataclasses.dataclass
class TestConnection:
    """A single connection to the listener, via stunnel or not."""

    idx: int
    encrypted: bool
    hostname: str
    port: int
    peer: Optional[str]
    msgq: asyncio.Queue[ListenerClientEvent]  # pylint: disable=unsubscriptable-object


class TestConnections(NamedTuple):
    """The various states of the test connections."""

    by_id: Dict[int, TestConnection]
    by_peer: Dict[str, TestConnection]
    pending: Dict[str, List[ListenerClientEvent]]
    unresolved: Dict[int, TestConnection]


class Config(NamedTuple):
    """Runtime configuration for the stunnel test."""

    certdir: pathlib.Path
    children: Dict[int, asyncio.subprocess.Process]  # pylint: disable=no-member
    mainq: asyncio.Queue[Event]  # pylint: disable=unsubscriptable-object
    program: pathlib.Path
    tasks: Dict[str, asyncio.Task[None]]  # pylint: disable=unsubscriptable-object
    tempd: pathlib.Path
    utf8_env: Dict[str, str]


@contextlib.contextmanager
def parse_args() -> Iterator[Config]:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--certdir",
        type=pathlib.Path,
        required=True,
        help="the path to the test certificate directory",
    )
    parser.add_argument(
        "--program",
        type=pathlib.Path,
        default=DEFAULT_PROG,
        help=f"the path to the stunnel executable to use (default: {DEFAULT_PROG}",
    )

    args = parser.parse_args()

    # Generally we'd use the utf8_locale Python library here, but, well,
    # this is Debian, right?
    utf8_env = dict(os.environ)
    utf8_env.update({"LC_ALL": "C.UTF-8", "LANGUAGE": ""})

    with tempfile.TemporaryDirectory(prefix="struntime.") as tempd_name:
        print(f"Using {tempd_name} as a temporary directory")
        yield Config(
            certdir=args.certdir,
            children={},
            program=args.program,
            mainq=asyncio.Queue(),
            tasks={},
            tempd=pathlib.Path(tempd_name),
            utf8_env=utf8_env,
        )


async def get_stunnel_version(cfg: Config) -> str:
    """Obtain the version of stunnel."""
    print(f"Trying to obtain the version of {cfg.program}")
    cmd = [str(cfg.program), "-version"]
    cmd_str = " ".join(shlex.quote(word) for word in cmd)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=cfg.utf8_env,
        )
    except (IOError, subprocess.CalledProcessError) as err:
        sys.exit(f"Could not start `{cmd_str}`: {err}")

    print(f"Started `{cmd_str}` as process {proc.pid}")
    b_out, b_err = await proc.communicate()
    assert b_out is not None and b_err is not None
    p_out, p_err = b_out.decode("UTF-8"), b_err.decode("UTF-8")

    rcode = await proc.wait()
    if rcode != 0:
        print(b_out.decode("UTF-8"))
        print(b_err.decode("UTF-8"), file=sys.stderr)
        sys.exit(f"`{cmd_str}` exited with code {rcode}")

    if p_out:
        sys.exit(f"`{cmd_str}` produced output on its stdout stream:\n{p_out}")

    lines = p_err.splitlines()
    if not lines:
        sys.exit(f"Expected at least one line of output from `{cmd_str}`")
    for line in lines:
        match = RE_VERSION.match(line)
        if match:
            return match.group("version")
    sys.exit("Could not find the version line in the `{cmd_str}` output:\n" + "\n".join(lines))


async def find_listening_port(
    port_first: int,
    port_last: int,
    callback: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Coroutine[Any, Any, None]],
    *,
    hostname: str = "localhost",
) -> asyncio.base_events.Server:
    """Find a suitable listening port."""
    print("[find_listening_port] Looking for a port to listen on")
    for port in range(port_first, port_last):
        print(f"[find_listening_port] Trying {hostname} port {port}")
        try:
            return await asyncio.start_server(
                callback,
                host=hostname,
                port=port,
                reuse_address=True,
            )
        except IOError as err:
            print(f"[listener] - {port} failed: {err}")

    raise Exception("Could not find a port to listen on")


async def start_listener(cfg: Config) -> None:
    """Find a port to listen on, return the listener task."""

    async def client_connected(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle... something."""
        try:
            print("[listener] A client connected from somewhere")
            peer_addr, peer_port = writer.get_extra_info("peername")[:2]
            peer = f"[{peer_addr}]:{peer_port}"
        except Exception as err:  # pylint: disable=broad-except
            print(f"[listener] Complaining about the client: {err}")
            await cfg.mainq.put(
                FatalEvent(
                    etype="listener",
                    reason=f"Handling a new connection: {err}",
                )
            )
            return

        try:
            print(f"[{peer}] New connection")
            print(f"[{peer}] Telling the main thread")
            await cfg.mainq.put(ClientConnectedEvent(etype="listener", peer=peer))

            print(f"[{peer}] Let them tell us something...")
            line = await reader.readline()
            match = RE_LINE_IDX.match(line.decode("UTF-8"))
            if not match:
                sys.exit(f"Unexpected message from {peer!r}: {line!r}")
            print(f"[{peer}] Telling the main thread about {line!r}")
            await cfg.mainq.put(
                ClientReceivedDataEvent(
                    etype="listener",
                    peer=peer,
                    line=line,
                    idx=int(match.group("idx")),
                )
            )

            print(f"[{peer}] Writing something back...")
            line = "There!\n".encode("UTF-8")
            writer.writelines([line])
            await writer.drain()
            print(f"[{peer}] Telling the main thread about {line!r}")
            await cfg.mainq.put(ClientSentDataEvent(etype="listener", peer=peer, line=line))

            print(f"[{peer}] Closing down the writer socket")
            writer.close()
            await writer.wait_closed()
            print(f"[{peer}] Telling the main thread done")
            await cfg.mainq.put(ClientDoneEvent(etype="listener", peer=peer))
        except Exception as err:  # pylint: disable=broad-except
            print(f"[{peer}] Complaining about the client: {err}")
            await cfg.mainq.put(FatalEvent(etype="listener", reason=f"Handling {peer}: {err}"))

    try:
        srv = await find_listening_port(6502, 6502 + 1000, client_connected)
        if not srv.sockets:
            raise Exception(f"[listener] Expected a listening socket, got {srv.sockets!r}")
        hostname, port = srv.sockets[0].getsockname()[:2]
        print(f"[listener] Telling the main thread about [{hostname}]:{port}")
        await cfg.mainq.put(ListenerStartedEvent(etype="listener", hostname=hostname, port=port))
        print("[listener] Awaiting client connections...")
        await srv.serve_forever()
        print("[listener] Done.")
    except Exception as err:  # pylint: disable=broad-except
        print(f"[listener] Complaining to the main thread: {err}")
        await cfg.mainq.put(FatalEvent(etype="listener", reason=f"Listener thread: {err}"))


async def cleanup_tasks(cfg: Config) -> None:
    """Cancel any remaining tasks."""
    print(f"About to cancel {len(cfg.tasks)} remaining task(s)")
    for name, task in cfg.tasks.items():
        print(f"- {name}")
        task.cancel()

    print("Waiting for the tasks to hopefully finish")
    await asyncio.gather(*cfg.tasks.values(), return_exceptions=True)


async def cleanup_children(cfg: Config) -> None:
    """Kill any remaining child processes."""
    print(f"About to kill and wait for {len(cfg.children)} child process(es)")
    waiters = [asyncio.create_task(proc.wait()) for proc in cfg.children.values()]

    for pid, proc in cfg.children.items():
        print(f"- pid {pid}")
        try:
            proc.kill()
        except ProcessLookupError:
            print("  - already finished, it seems")
        except Exception as err:  # pylint: disable=broad-except
            print(f"  - {err!r}")

    print("Waiting for the processes to exit...")
    wait_res = await asyncio.gather(*waiters)
    print(f"Got processes' exit status: {wait_res!r}")


TEvent = TypeVar("TEvent", bound=Event)


async def expect_event(
    msgq: asyncio.Queue[Event],  # pylint: disable=unsubscriptable-object
    tag: str,
    evtype: Type[TEvent],
) -> TEvent:
    """Make sure the next event in the main queue is of that type."""
    evt = await msgq.get()
    if not isinstance(evt, evtype):
        sys.exit(f"[{tag}] Expected {evtype.__name__}, got {evt!r}")
    return evt


TListenerClientEvent = TypeVar("TListenerClientEvent", bound=ListenerClientEvent)


async def expect_client_event(
    msgq: asyncio.Queue[ListenerClientEvent],  # pylint: disable=unsubscriptable-object
    tag: str,
    evtype: Type[TListenerClientEvent],
) -> TListenerClientEvent:
    """Make sure the next event in the main queue is of that type."""
    evt = await msgq.get()
    if not isinstance(evt, evtype):
        sys.exit(f"[{tag}] Expected {evtype.__name__}, got {evt!r}")
    return evt


async def test_connect(cfg: Config, conn: TestConnection) -> None:
    """Whee!"""
    # pylint: disable=too-many-statements

    async def wait() -> None:
        """Wait for a little while."""
        await asyncio.sleep(random.randint(3, 10) / 10)

    tag = f"test_connect [{conn.hostname}]:{conn.port} {conn.idx}"
    try:
        print(f"[{tag}] Trying port {conn.port} encrypted {conn.encrypted}")
        await wait()
        try:
            if conn.encrypted:
                print(f"[{tag}] Creating an SSL context")
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                print(f"[{tag}] - cert required")
                ctx.verify_mode = ssl.CERT_REQUIRED
                print(f"[{tag}] - load_verify_locations()")
                ctx.load_verify_locations(cafile=str(cfg.certdir / "certificate.pem"))
                print(f"[{tag}] Opening an SSL connection")
                reader, writer = await asyncio.open_connection(
                    conn.hostname,
                    conn.port,
                    ssl=ctx,
                    server_hostname="localhost",
                )
            else:
                print(f"[{tag}] Opening an unencrypted connection")
                reader, writer = await asyncio.open_connection(
                    conn.hostname,
                    conn.port,
                    ssl=False,
                )
        except IOError as err:
            sys.exit(f"Failed to connect to {conn.hostname}:{conn.port}: {err}")

        sock_addr, sock_port = writer.get_extra_info("sockname")[:2]
        local = f"[{sock_addr}]:{sock_port}"
        print(f"[{tag}] Connected to the server: {local}")
        await wait()
        print(f"[{tag}] Sending something...")
        line = f"Hello {conn.idx}\n".encode("UTF-8")
        writer.writelines([line])
        await writer.drain()

        print(f"[{tag}] Waiting for the main thread to figure it out")
        evt_conn = await expect_client_event(conn.msgq, "test_connect", ClientConnectedEvent)
        print(f"[{tag}] The listener acknowledged {evt_conn.peer}")
        if conn.encrypted:
            if local == evt_conn.peer:
                sys.exit(f"[{tag}] expected something other than {local!r}")
        else:
            if local != evt_conn.peer:
                sys.exit(f"[{tag}] expected {local!r}, got {evt_conn.peer!r}")

        print(f"[{tag}] Waiting for the server to receive it...")
        evt_recv = await expect_client_event(conn.msgq, "test_connect", ClientReceivedDataEvent)
        if evt_recv.line != line:
            sys.exit(f"[{tag}] Send: expected {line!r}, got {evt_recv.line!r}")

        print(f"[{tag}] Waiting for the server to send something")
        evt_sent = await expect_client_event(conn.msgq, "test_connect", ClientSentDataEvent)
        print(f"[{tag}] Trying to receive the actual data")
        line = await reader.readline()
        if line != evt_sent.line:
            sys.exit(f"[{tag}] Receive: expected {evt_sent.line!r}, got {line!r}")

        print(f"[{tag}] Waiting for the server to close the connection")
        await expect_client_event(conn.msgq, "test_connect", ClientDoneEvent)
        print(f"[{tag}] Waiting for an EOF on the reader socket")
        line = await reader.read(1)
        if line:
            sys.exit(f"[{tag}] Did not expect to read {line!r}")

        print(f"[{tag}] Closing our writer socket, too")
        writer.close()
        await writer.wait_closed()

        print(f"[{tag}] Letting the main thread know we're done")
        await cfg.mainq.put(ConnectionDoneEvent(etype=tag, idx=conn.idx))
        print(f"[{tag}] Done")
    except Exception as err:  # pylint: disable=broad-except
        print(f"[{tag}] Telling the main thread about {err}")
        await cfg.mainq.put(FatalEvent(etype=tag, reason="Something went wrong: {err}"))


async def find_stunnel_port(cfg: Config, hostname: str, port: int) -> int:
    """Find a suitable listening port for the stunnel server."""

    async def callback(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """This should never be called... right?"""
        print("[find_stunnel_port] the callback was invoked?!")
        await cfg.mainq.put(
            FatalEvent(
                etype="find_stunnel_port",
                reason=f"callback invoked: reader {reader!r} writer {writer!r}",
            )
        )

    srv = await find_listening_port(port + 1, port + 1001, callback, hostname=hostname)
    if not srv.sockets:
        sys.exit(f"Expected a listening socket, got {srv.sockets!r}")
    lport = srv.sockets[0].getsockname()[1]
    assert isinstance(lport, int)
    print(f"[find_listening_port] got {lport}, shutting down the listener")
    srv.close()
    return lport


async def prepare_config_file(
    cfg: Config, hostname: str, port: int, stunnel_port: int
) -> pathlib.Path:
    """Create a configuration file for stunnel."""
    proc = await asyncio.create_subprocess_exec(
        "install",
        "-m",
        "400",
        "--",
        str(cfg.certdir / "key.pem"),
        str(cfg.tempd / "key.pem"),
    )
    rcode = await proc.wait()
    if rcode != 0:
        sys.exit(f"Could not copy the key file, install exit code {rcode}")

    contents = f"""
pid = {cfg.tempd}/stunnel.pid
foreground = yes

cert = {cfg.certdir}/certificate.pem
key = {cfg.tempd}/key.pem

[test]
accept = {hostname}:{stunnel_port}
connect = {hostname}:{port}
"""

    cfgfile = cfg.tempd / "stunnel.conf"
    cfgfile.write_text(contents, encoding="UTF-8")
    return cfgfile


async def launch_stunnel(
    cfg: Config, cfgfile: pathlib.Path
) -> asyncio.subprocess.Process:  # pylint: disable=no-member
    """Launch the stunnel server with the specified config file."""

    return await asyncio.create_subprocess_exec(
        str(cfg.program),
        str(cfgfile),
        stdin=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
        env=cfg.utf8_env,
    )


async def stunnel_output(cfg: Config, p_out: asyncio.StreamReader) -> None:
    """Pipe the stunnel output thing."""
    try:
        while True:
            data = await p_out.readline()
            if not data:
                raise Exception("The stunnel child process ended unexpectedly")

            line = data.decode("UTF-8").rstrip("\r\n")
            print(f"[stunnel_output] Read {line!r}")
            if "Configuration successful" in line:
                print("[stunnel_output] The main thread ought to know")
                await cfg.mainq.put(ChildReadyEvent(etype="stunnel_output"))
    except Exception as err:  # pylint: disable=broad-except
        print(f"[stunnel_output] Complaining to the main thread about {err}")
        await cfg.mainq.put(
            FatalEvent(etype="stunnel_output", reason=f"Something went wrong: {err}")
        )


def validate_conns(conns: TestConnections) -> None:
    """Ensure the connections state is consistent."""

    def validate_by_id_unresolved(idx: int, conn: TestConnection) -> None:
        """Validate a still-unresolved connection."""
        weird_p = [
            (peer, pconn)
            for peer, pconn in conns.by_peer.items()
            if pconn is conn or pconn.idx == idx
        ]
        if weird_p:
            sys.exit(f"by_id conn {conn!r} in by_peer " f"weird_p {weird_p!r}")

        unres = conns.unresolved.get(idx)
        if unres is None:
            sys.exit(f"conn {conn!r} not in unresolved")
        if unres is not conn:
            sys.exit(f"by_id conn {conn!r} not the same as unresolved {unres!r}")
        weird_i = [
            (uidx, uconn)
            for uidx, uconn in conns.unresolved.items()
            if (uconn is conn) != (uidx == idx)
        ]
        if weird_i:
            sys.exit(f"by_id conn {conn!r} doubly unresolved weird_i {weird_i!r}")

    def validate_by_id_resolved(idx: int, conn: TestConnection) -> None:
        """Validate an already-resolved connection."""
        assert conn.peer is not None
        res = conns.by_peer.get(conn.peer)
        if res is None:
            sys.exit(f"by_id conn {conn!r} not in by_peer")
        if res is not conn:
            sys.exit(f"by_id conn {conn!r} not the same as by_peer {res!r}")

        weird_p = [
            (pidx, pconn)
            for pidx, pconn in conns.by_peer.items()
            if (pidx == conn.peer) != (pconn is conn)
        ]
        if weird_p:
            sys.exit(f"by_id conn {conn!r} doubly in by_peer weird_p {weird_p!r}")

        weird_i = [
            (iidx, iconn)
            for iidx, iconn in conns.unresolved.items()
            if iidx == idx or iconn is conn
        ]
        if weird_i:
            sys.exit(f"by_id conn {conn!r} in unresolved weird_i {weird_i!r}")

        if conn.peer in conns.pending:
            sys.exit(f"by_id conn {conn!r} in pending {conns.pending[conn.peer]!r}")

    def validate_by_id(idx: int, conn: TestConnection) -> None:
        """Validate a connection in the by_id mapping."""
        if conn.idx != idx:
            sys.exit(f"by_id conn {conn!r} should have idx {idx!r}")
        weird_i = [
            (iidx, iconn) for iidx, iconn in conns.by_id.items() if (iidx == idx) != (iconn is conn)
        ]
        if weird_i:
            sys.exit(f"by_id conn {conn!r} double weird_i {weird_i!r}")

        if conn.peer is None:
            validate_by_id_unresolved(idx, conn)
        else:
            validate_by_id_resolved(idx, conn)

    for idx, conn in sorted(conns.by_id.items()):
        validate_by_id(idx, conn)

    for peer, conn in sorted(conns.by_peer.items()):
        if conn.peer != peer:
            sys.exit(f"by_peer conn {conn!r} should have peer {peer!r}")
        if conn.idx not in conns.by_id:
            sys.exit(f"by_peer conn {conn!r} not in by_id")

    for idx, conn in conns.unresolved.items():
        if conn.idx != idx:
            sys.exit(f"unresolved conn {conn!r} should have idx {idx!r}")
        if idx not in conns.by_id:
            sys.exit(f"unresolved conn {conn!r} not in by_id")

    for peer, events in sorted(conns.pending.items()):
        others = [
            evt for evt in events if not isinstance(evt, ListenerClientEvent) or evt.peer != peer
        ]
        if others:
            sys.exit(f"pending peer {peer!r} weird events {others!r}")

        if peer in conns.by_peer:
            sys.exit(f"pending peer {peer!r} events {events!r} in by_peer {conns.by_peer[peer]!r}")


def start_connections(
    cfg: Config,
    conns: TestConnections,
    hostname: str,
    port: int,
    *,
    encrypted: bool,
    prefix: str,
) -> None:
    """Start a group of similar connections."""
    for idx in range(10):
        conn = TestConnection(
            idx=idx,
            encrypted=encrypted,
            hostname=hostname,
            port=port,
            peer=None,
            msgq=asyncio.Queue(),
        )
        conns.by_id[idx] = conn
        conns.unresolved[idx] = conn
        validate_conns(conns)
        cfg.tasks[f"{prefix}{idx}"] = asyncio.create_task(test_connect(cfg, conn))


async def process_connections(cfg: Config, conns: TestConnections, *, prefix: str) -> None:
    """Wait for all the connections to complete."""

    async def process_listener_event(evt: ListenerClientEvent) -> None:
        """Shuffle things around the conns structure."""
        peer = evt.peer
        if peer in conns.by_peer:
            await conns.by_peer[peer].msgq.put(evt)
            return

        if peer in conns.pending:
            conns.pending[peer].append(evt)
            if isinstance(evt, ClientReceivedDataEvent):
                conn = conns.by_id.get(evt.idx)
                if conn is None:
                    sys.exit(f"Listener reported unknown connection {evt!r}")
                if conn.peer is not None:
                    sys.exit(f"Listener reported bad conn {conn!r} {evt!r}")
                conn.peer = peer
                conns.by_peer[peer] = conns.unresolved.pop(evt.idx)

                for pevt in conns.pending.pop(peer):
                    await conn.msgq.put(pevt)
            return

        if not isinstance(evt, ClientConnectedEvent):
            sys.exit(f"Expected 'client connected' first, " f"got {evt!r}")
        conns.pending[peer] = [evt]

    async def process_connection_done(evt: ConnectionDoneEvent) -> None:
        """Remove a connection from the structure."""
        conn = conns.by_id.get(evt.idx)
        if conn is None:
            sys.exit(f"No connection for {evt!r}")
        if conn.peer is None:
            sys.exit(f"Connection done too early: evt {evt!r} conn {conn!r}")

        del conns.by_id[evt.idx]
        del conns.by_peer[conn.peer]

        task_name = f"{prefix}{evt.idx}"
        print(f"[process_connections] Fetching task {task_name}")
        task = cfg.tasks.pop(task_name)
        print(f"[process_connections] Waiting for task {task_name}")
        await asyncio.gather(task)
        print(f"[process_connections] Done with task {task_name}")

    while conns.by_id:
        evt = await cfg.mainq.get()
        validate_conns(conns)

        if isinstance(evt, ListenerClientEvent):
            await process_listener_event(evt)
        elif isinstance(evt, ConnectionDoneEvent):
            await process_connection_done(evt)
        else:
            sys.exit(f"Did not expect {evt!r}")

        validate_conns(conns)

    validate_conns(conns)


async def main() -> None:
    """Main program: parse arguments, prepare an environment, run tests."""
    with parse_args() as cfg:
        stunnel_version = await get_stunnel_version(cfg)
        print(f"Got stunnel version {stunnel_version}")

        try:
            print("[main] Starting to do things")
            cfg.tasks["listen"] = asyncio.create_task(start_listener(cfg))

            print("[main] Awaiting the 'listener started' event")
            evt_listener = await expect_event(cfg.mainq, "main", ListenerStartedEvent)
            hostname = evt_listener.hostname
            port = evt_listener.port
            print(f"[main] Apparently we are listening on [{hostname}]:{port}")

            conns = TestConnections(by_id={}, by_peer={}, pending={}, unresolved={})

            print("[main] Testing cleartext connections")
            validate_conns(conns)
            start_connections(cfg, conns, hostname, port, encrypted=False, prefix="plain-")
            await process_connections(cfg, conns, prefix="plain-")

            print("[main] Picking a listen address:port for stunnel")
            stunnel_port = await find_stunnel_port(cfg, hostname, port)
            print(f"[main] Will put stunnel at {hostname}:{stunnel_port}")

            print("[main] Preparing the stunnel config file")
            cfgfile = await prepare_config_file(cfg, hostname, port, stunnel_port)
            print(f"[main] Using config file {cfgfile}")
            for line in cfgfile.read_text(encoding="UTF-8").splitlines():
                print(f"[main] {line}")
            print("[main] End of the config file")

            print("[main] Launching the stunnel server")
            proc = await launch_stunnel(cfg, cfgfile)
            print(f"[main] - got pid {proc.pid}")
            cfg.children[proc.pid] = proc

            print("[main] Launching the stunnel output pipe thread")
            assert proc.stderr is not None
            cfg.tasks["output"] = asyncio.create_task(stunnel_output(cfg, proc.stderr))
            print("[main] Waiting for stunnel to start up")
            await expect_event(cfg.mainq, "main", ChildReadyEvent)

            print("[main] Testing the encrypted connections")
            validate_conns(conns)
            start_connections(
                cfg,
                conns,
                hostname,
                stunnel_port,
                encrypted=True,
                prefix="encr-",
            )
            await process_connections(cfg, conns, prefix="encr-")

            print("[main] Everything seems to be all right!")
        finally:
            try:
                await cleanup_tasks(cfg)
            finally:
                await cleanup_children(cfg)


if __name__ == "__main__":
    asyncio.run(main())
