import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)


def ensure_torch_triton_compat() -> None:
    """
    FlashAttention>=2.6 calls `torch.library.wrap_triton`, which was added in PyTorch 2.4.
    Older PyTorch releases (2.1/2.2/2.3) don't provide this helper, leading to
    AttributeError during import time when vLLM executes FlashAttention kernels.

    We only need the helper so FlashAttention can obtain a callable Triton kernel,
    so on older versions we emulate the PyTorch API by returning the original kernel.
    """

    try:
        import torch
    except Exception:  # pragma: no cover - torch is always expected to be available at runtime
        return

    torch_library: Any = getattr(torch, "library", None)
    if torch_library is None:
        return

    if hasattr(torch_library, "wrap_triton"):
        return

    def _wrap_triton_passthrough(kernel, *args, **kwargs):
        if args or kwargs:
            _LOGGER.debug(
                "torch.library.wrap_triton compatibility shim ignores meta-arguments: args=%s kwargs=%s",
                args,
                kwargs,
            )
        return kernel

    setattr(torch_library, "wrap_triton", _wrap_triton_passthrough)
    _LOGGER.warning(
        "torch.library.wrap_triton is unavailable in this torch build; "
        "registered a compatibility shim that falls back to invoking the raw Triton kernel. "
        "For optimal performance, please upgrade torch to >=2.4."
    )
