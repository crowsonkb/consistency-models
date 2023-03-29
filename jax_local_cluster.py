#!/usr/bin/env python3

"""A simple JAX process launcher for multiple devices on a single host.

You must import jax_local_cluster somewhere inside the script you are launching.
"""

import argparse
from functools import partial
import os
import signal
import socketserver
from subprocess import Popen, TimeoutExpired
import sys

import jax
import jax._src as _src

error = partial(print, file=sys.stderr)


class LocalCluster(_src.clusters.ClusterEnv):
    @classmethod
    def is_env_present(cls):
        return "JAX_COORDINATOR_ADDRESS" in os.environ

    @classmethod
    def get_coordinator_address(cls):
        return os.environ["JAX_COORDINATOR_ADDRESS"]

    @classmethod
    def get_process_count(cls):
        return int(os.environ["JAX_PROCESS_COUNT"])

    @classmethod
    def get_process_id(cls):
        return int(os.environ["JAX_PROCESS_ID"])

    @classmethod
    def get_local_process_id(cls):
        return int(os.environ["JAX_LOCAL_PROCESS_ID"])


def get_free_port():
    with socketserver.TCPServer(("127.0.0.1", 0), None) as s:
        return s.server_address[1]


def signal_and_wait(signum, procs, ctx, timeout=None):
    for proc in procs:
        proc.send_signal(signum)
    for i, proc in enumerate(procs):
        ctx["i"] = i
        proc.wait(timeout)
    ctx["i"] = None


def interactive_shutdown(procs):
    ctx = {"i": None}
    try:
        signal_and_wait(signal.SIGINT, procs, ctx)
    except KeyboardInterrupt:
        try:
            error(
                f"Process {ctx['i']} (pid {procs[ctx['i']].pid}) did not exit on SIGINT, trying SIGTERM"
            )
            signal_and_wait(signal.SIGTERM, procs, ctx, timeout=1)
        except (KeyboardInterrupt, TimeoutExpired):
            error(
                f"Process {ctx['i']} (pid {procs[ctx['i']].pid}) did not exit on SIGTERM, trying SIGKILL"
            )
            for proc in procs:
                proc.kill()


class TerminationHandler:
    def __init__(self, procs, verbose):
        self.procs = procs
        self.verbose = verbose
        self.was_called = False

    def __call__(self, signum, frame):
        self.was_called = True
        if self.verbose:
            error("SIGTERM received, shutting down")
        try:
            signal_and_wait(signal.SIGTERM, self.procs, {}, timeout=1)
        except TimeoutExpired:
            if self.verbose:
                error("SIGTERM timed out, sending SIGKILL")
            for proc in self.procs:
                proc.kill()
        raise KeyboardInterrupt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-n",
        type=int,
        default=0,
        help="Number of processes to launch (default: one per local device)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to use for the coordinator (default: a free port)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    p.add_argument("command", type=str, nargs=argparse.REMAINDER, help="Command to run")
    args = p.parse_args()

    if not args.command:
        p.print_help()
        sys.exit(1)

    n = args.n if args.n else jax.local_device_count()
    if args.verbose:
        error(f"Launching {n} processes")
    port = args.port if args.port else get_free_port()
    if args.verbose:
        error(f"Using port {port} for coordinator")

    procs = []
    sigterm_handler = TerminationHandler(procs, args.verbose)
    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        for i in range(n):
            env = os.environ.copy()
            env.pop("OMPI_MCA_orte_hnp_uri", None)
            env.pop("SLURM_JOB_ID", None)
            env["JAX_COORDINATOR_ADDRESS"] = f"127.0.0.1:{port}"
            env["JAX_PROCESS_COUNT"] = str(n)
            env["JAX_PROCESS_ID"] = str(i)
            env["JAX_LOCAL_PROCESS_ID"] = str(i)
            proc = Popen(args.command, env=env)
            if args.verbose:
                error(f"Launched process {i} (pid {proc.pid})")
            procs.append(proc)
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if not sigterm_handler.was_called:
            interactive_shutdown(procs)
        if args.verbose:
            error("All processes terminated")


if __name__ == "__main__":
    main()
