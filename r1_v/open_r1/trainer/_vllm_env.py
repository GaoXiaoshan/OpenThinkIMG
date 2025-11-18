import os
import socket
from contextlib import closing
from typing import Tuple


def _find_free_port() -> int:
    """Return an available TCP port on the current host."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def build_isolated_vllm_env() -> Tuple[dict[str, str], bool]:
    """
    Build a sanitized distributed environment for vLLM so that it does not
    inherit torchrun's rendezvous settings. Returns the environment variables
    to patch and whether the rendezvous port was provided by the user.

    Users can override the defaults with:
      - VLLM_MASTER_ADDR
      - VLLM_MASTER_PORT
      - VLLM_RANK / VLLM_LOCAL_RANK
      - VLLM_WORLD_SIZE / VLLM_LOCAL_WORLD_SIZE
    """

    env: dict[str, str] = {}
    env["MASTER_ADDR"] = os.getenv("VLLM_MASTER_ADDR", "127.0.0.1")

    master_port = os.getenv("VLLM_MASTER_PORT")
    user_port = master_port is not None
    if master_port is None:
        master_port = str(_find_free_port())
    env["MASTER_PORT"] = master_port

    env["RANK"] = os.getenv("VLLM_RANK", "0")
    env["LOCAL_RANK"] = os.getenv("VLLM_LOCAL_RANK", env["RANK"])
    env["WORLD_SIZE"] = os.getenv("VLLM_WORLD_SIZE", "1")
    env["LOCAL_WORLD_SIZE"] = os.getenv("VLLM_LOCAL_WORLD_SIZE", env["WORLD_SIZE"])
    env["GROUP_RANK"] = os.getenv("VLLM_GROUP_RANK", "0")
    env["ROLE_RANK"] = os.getenv("VLLM_ROLE_RANK", "0")
    env["ROLE_WORLD_SIZE"] = os.getenv("VLLM_ROLE_WORLD_SIZE", "1")

    return env, user_port
