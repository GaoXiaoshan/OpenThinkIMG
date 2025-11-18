import argparse
import base64
import io
from multiprocessing.connection import Listener
from typing import List, Optional

from PIL import Image
import torch

from vllm import LLM, SamplingParams

from .tool_generation import vllm_generate_with_tool_calls


def _decode_image(data: Optional[str]) -> Optional[Image.Image]:
    if not data:
        return None
    image_bytes = base64.b64decode(data)
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
    return image


def _compact_outputs(outputs: List[dict]) -> List[dict]:
    compact = []
    for item in outputs:
        compact.append(
            {
                "model_outputs": item.get("model_outputs", []),
                "model_output_ids": item.get("model_output_ids", []),
                "tool_outputs": item.get("tool_outputs", []),
            }
        )
    return compact


def serve(args):
    llm_kwargs = {
        "model": args.model_path,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": torch.bfloat16,
        "enforce_eager": True,
    }
    if args.vllm_device != "auto":
        llm_kwargs["device"] = args.vllm_device
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    mm_kwargs = {"max_pixels": args.max_pixels, "min_pixels": args.min_pixels}
    llm_kwargs["mm_processor_kwargs"] = mm_kwargs
    llm_kwargs["limit_mm_per_prompt"] = {"image": 6}
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    listener = Listener(
        (args.host, args.port),
        authkey=bytes.fromhex(args.auth_key),
        backlog=8,
    )

    while True:
        conn = listener.accept()
        try:
            while True:
                message = conn.recv()
                cmd = message.get("cmd")
                if cmd == "ping":
                    conn.send({"status": "ok"})
                    continue
                if cmd == "shutdown":
                    conn.send({"status": "ok"})
                    conn.close()
                    listener.close()
                    return
                if cmd == "generate":
                    prompts = message["prompts"]
                    raw_images = message["images"]
                    images = [_decode_image(item) for item in raw_images]
                    outputs = vllm_generate_with_tool_calls(
                        llm,
                        prompts=prompts,
                        images=images,
                        sampling_params=sampling_params,
                        max_rounds=message["max_rounds"],
                        model_mode=message["model_mode"],
                        controller_addr=message["controller_addr"],
                    )
                    conn.send(
                        {
                            "status": "ok",
                            "data": _compact_outputs(outputs),
                        }
                    )
                else:
                    conn.send({"status": "error", "error": f"Unknown cmd {cmd}"})
        except EOFError:
            conn.close()
        except Exception as exc:  # pylint: disable=broad-except
            conn.send({"status": "error", "error": repr(exc)})
            conn.close()


def main():
    parser = argparse.ArgumentParser("vLLM sidecar worker")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vllm-device", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=12845056)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--auth-key", required=True)
    args = parser.parse_args()

    serve(args)


if __name__ == "__main__":
    main()
