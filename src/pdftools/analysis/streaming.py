
import atexit
import json
import threading
import traceback
from pathlib import Path

import bluesky.plans as bp
from bluesky.run_engine import RunEngine
from bluesky_tiled_plugins import BlueskyEventStream, BlueskyRun, TiledWriter
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider, init_devices
from ophyd_async.sim import PatternGenerator, SimBlobDetector, SimPointDetector
from tiled.client import from_uri
from tiled.client.container import Any, Container
from tiled.server import SimpleTiledServer


def tprint(msg):
    "Thread-aware print"
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] {msg}")


class StreamingClient:
    """Manages tiled streaming subscriptions and their background threads."""

    def __init__(self, client: Container):
        self._client = client
        self._subscriptions = []
        self._current_run_md: dict[str, Any] = {}

    def start(self):
        """Start watching the catalog for new children."""
        catalog_sub = self._client.subscribe()
        catalog_sub.child_created.add_callback(self._on_new_child)
        catalog_sub.start_in_thread(1)
        self._subscriptions.append(catalog_sub)

    def stop(self):
        """Disconnect all subscriptions and stop their threads."""
        for sub in self._subscriptions:
            try:
                sub.disconnect()
            except Exception:
                pass
        self._subscriptions.clear()

    def _on_new_child(self, update):
        "A new child node has been created in a container."
        try:
            child = update.child()

            # If this is a new run, print the start document
            if isinstance(child, BlueskyRun):
                tprint(f"Start document: {child.start}")
                self._current_run_md = child.start  #type: ignore

            # Filter: only subscribe to 'primary' streams
            if isinstance(child, BlueskyEventStream) and child.item.get("id") != "primary":
                tprint(f"Skipping stream '{child.item.get('id')}' (only subscribing to 'primary')")
                return

            sub = child.subscribe()

            # Is the child also a container?
            if child.structure_family == "container":
                # Recursively subscribe to the children of this new container.
                sub.child_created.add_callback(self._on_new_child)
            else:
                # Subscribe to data updated (maybe appended).
                sub.new_data.add_callback(self._on_new_data)  #type: ignore

            sub.start_in_thread(1)
            self._subscriptions.append(sub)
        except Exception:
            traceback.print_exc()

    def _on_new_data(self, update):
        "Data has been updated (maybe appended) to an array or table."
        try:
            data = update.data()
            print(data.shape)
            print(type(update))
            tprint(f"Data received:\n{data}")
        except Exception:
            traceback.print_exc()


s = SimpleTiledServer("/tmp", readable_storage="/tmp")
c: Container = from_uri(s.uri)
tw = TiledWriter(c)
RE = RunEngine({})
RE.subscribe(tw)

streaming_client = StreamingClient(c)
streaming_client.start()


def _cleanup():
    streaming_client.stop()
    s.stop()


atexit.register(_cleanup)

pp = StaticPathProvider(UUIDFilenameProvider(), Path("/tmp"))
with init_devices():
    sim_point = SimPointDetector(None)
    pattern_gen = PatternGenerator()
    sim_blob = SimBlobDetector(pp, pattern_generator=pattern_gen)

