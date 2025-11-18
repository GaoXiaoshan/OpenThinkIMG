import atexit
import base64
import io
import os
import secrets
import subprocess
import sys
import time
from multiprocessing.connection import Client
from typing import List, Optional

from PIL import Image


DDP_ENV_VARS = [
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
]


def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def _sanitize_env() -> dict:
    env = os.environ.copy()
    for key in DDP_ENV_VARS:
        env.pop(key, None)
    return env


def _encode_image(image: Optional[Image.Image]) -> Optional[str]:
    if image is None:
        return None
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


class VLLMServiceClient:
    def __init__(
        self,
        *,
        model_path: str,
        vllm_device: str,
        gpu_memory_utilization: float,
        max_pixels: int,
        min_pixels: int,
        max_model_len: Optional[int],
        max_tokens: int,
        temperature: float,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        startup_timeout: float = 120.0,
    ):
        self.host = host
        self.port = port or _find_free_port()
        self.auth_key = secrets.token_hex(16)
        self.process: Optional[subprocess.Popen] = None
        self.model_path = model_path
        self.vllm_device = vllm_device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.startup_timeout = startup_timeout
        self._start_process()
        atexit.register(self.shutdown)

    def _start_process(self):
        cmd = [
            sys.executable,
            "-m",
            "r1_v.open_r1.trainer.vllm_service_worker",
            "--model-path",
            self.model_path,
            "--vllm-device",
            self.vllm_device,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-tokens",
            str(self.max_tokens),
            "--temperature",
            str(self.temperature),
            "--max-pixels",
            str(self.max_pixels),
            "--min-pixels",
            str(self.min_pixels),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--auth-key",
            self.auth_key,
        ]
        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        env = _sanitize_env()
        env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
        self.process = subprocess.Popen(cmd, env=env)
        self._wait_until_ready()

    def _wait_until_ready(self):
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError("vLLM service exited before becoming ready.")
            try:
                conn = Client(
                    (self.host, self.port),
                    authkey=bytes.fromhex(self.auth_key),
                )
                conn.send({"cmd": "ping"})
                resp = conn.recv()
                conn.close()
                if resp.get("status") == "ok":
                    return
            except (ConnectionRefusedError, FileNotFoundError, ConnectionResetError):
                time.sleep(0.5)
        raise TimeoutError("Timed out waiting for vLLM service to become ready.")

    def _request(self, payload: dict) -> dict:
        conn = Client((self.host, self.port), authkey=bytes.fromhex(self.auth_key))
        conn.send(payload)
        response = conn.recv()
        conn.close()
        return response

    def generate(
        self,
        *,
        prompts,
        images: List[Optional[Image.Image]],
        max_rounds: int,
        model_mode: str,
        controller_addr: str,
    ) -> List[dict]:
        serialized_images = [_encode_image(image) for image in images]
        response = self._request(
            {
                "cmd": "generate",
                "prompts": prompts,
                "images": serialized_images,
                "max_rounds": max_rounds,
                "model_mode": model_mode,
                "controller_addr": controller_addr,
            }
        )
        if response.get("status") != "ok":
            raise RuntimeError(f"vLLM service error: {response.get('error')}")
        return response["data"]

    def shutdown(self):
        if self.process is None:
            return
        try:
            self._request({"cmd": "shutdown"})
        except Exception:
            pass
        try:
            self.process.terminate()
        except Exception:
            pass
        self.process = None
