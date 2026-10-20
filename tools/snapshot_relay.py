#!/usr/bin/env python3
"""Serve a vol_field snapshot over TCP, for a live plot of a headless run.

Runs next to vf_sim on a WATcloud node: reads the shared-memory snapshot
(seqlock-stable copies) and sends each new one to every connected client as
[u32 big-endian length][header + current records]. Then, on the laptop:

  ssh -L 7700:<node>:7700 wato-login
  python3 tools/plot_surface.py --tcp localhost:7700

  tools/snapshot_relay.py --shm vol_field --port 7700
"""

import argparse
import socket
import socketserver
import struct
import sys
import threading
import time

import vfsnap


class Latest:
    def __init__(self):
        self.cond = threading.Condition()
        self.frame = None
        self.id = 0


def closed(sock):
    """True once the client hung up (it never sends, so any readable byte is EOF)."""
    try:
        return sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
    except BlockingIOError:
        return False
    except OSError:
        return True


def reader(src, latest, hz, stop):
    period = 1.0 / hz
    while not stop.is_set():
        snap = src.read()
        if snap is not None and snap.id != latest.id:
            body = snap.compact()
            with latest.cond:
                latest.frame = struct.pack(">I", len(body)) + body
                latest.id = snap.id
                latest.cond.notify_all()
        time.sleep(period)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shm", default="vol_field")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7700)
    ap.add_argument("--hz", type=float, default=10.0, help="how often to look for a new snapshot")
    args = ap.parse_args()

    src = vfsnap.ShmSource(args.shm)
    latest = Latest()
    stop = threading.Event()
    threading.Thread(target=reader, args=(src, latest, args.hz, stop), daemon=True).start()

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            peer = "%s:%d" % self.client_address
            print(f"[relay] {peer} connected", flush=True)
            sent = 0
            try:
                while not stop.is_set():
                    with latest.cond:
                        latest.cond.wait_for(lambda: latest.id > sent or stop.is_set(), timeout=1.0)
                        frame, sid = latest.frame, latest.id
                    if frame is not None and sid > sent:
                        self.request.sendall(frame)
                        sent = sid
                    elif closed(self.request):
                        break
            except OSError:
                pass
            print(f"[relay] {peer} gone", flush=True)

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer.daemon_threads = True
    with socketserver.ThreadingTCPServer((args.bind, args.port), Handler) as server:
        host = socket.gethostname().split(".")[0]
        print(f"[relay] serving {src.path} on {args.bind}:{args.port}\n"
              f"[relay] laptop: ssh -L {args.port}:{host}:{args.port} wato-login, then\n"
              f"[relay]         python3 tools/plot_surface.py --tcp localhost:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
