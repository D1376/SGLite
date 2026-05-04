"""ZMQ queue wrappers for local process communication."""

from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")


class ZmqPushQueue(Generic[T]):
    """Queue wrapper for ZMQ push."""
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        """Initialize the ZMQ push queue."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T):
        """Enqueue one item onto the transport."""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        """Close the transport and release its resources."""
        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    """Queue wrapper for ZMQ async push."""
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        """Initialize the ZMQ async push queue."""
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        """Enqueue one item onto the transport."""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        """Close the transport and release its resources."""
        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    """Queue wrapper for ZMQ pull."""
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        """Initialize the ZMQ pull queue."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        """Receive one item from the transport."""
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        """Receive one raw payload without decoding it."""
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        """Decode one raw payload into a structured message."""
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        """Return whether the transport currently has no available items."""
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        """Close the transport and release its resources."""
        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    """Queue wrapper for ZMQ async pull."""
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        """Initialize the ZMQ async pull queue."""
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get(self) -> T:
        """Receive one item from the transport."""
        event = await self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        """Close the transport and release its resources."""
        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    """Queue wrapper for ZMQ pub."""
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        """Initialize the ZMQ pub queue."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        """Enqueue one raw payload onto the transport."""
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        """Enqueue one item onto the transport."""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        """Close the transport and release its resources."""
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    """Queue wrapper for ZMQ sub."""
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        """Initialize the ZMQ sub queue."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get(self) -> T:
        """Receive one item from the transport."""
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        """Return whether the transport currently has no available items."""
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        """Close the transport and release its resources."""
        self.socket.close()
        self.context.term()
