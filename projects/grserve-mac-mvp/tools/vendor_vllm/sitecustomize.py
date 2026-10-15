"""Opt-in bootstrap for local patches injected into the vendor vLLM image."""

from __future__ import annotations

import os

if os.getenv("GRSERVE_FANT_CONTIGUOUS_LINEAR") == "1":
    from fant_contiguous_linear import install_fant_contiguous_linear_patch

    install_fant_contiguous_linear_patch()

if os.getenv("GRSERVE_FANT_DUAL_LAYOUT_LINEAR") == "1":
    from fant_dual_layout_linear import install_fant_dual_layout_linear_patch

    install_fant_dual_layout_linear_patch(
        max_copies=int(os.getenv("GRSERVE_FANT_DUAL_LAYOUT_MAX_COPIES", "8")),
        defer_small_calls=int(
            os.getenv("GRSERVE_FANT_DUAL_LAYOUT_DEFER_SMALL_CALLS", "4")
        ),
    )

if os.getenv("GRSERVE_FANT_SLAB_LINEAR") == "1":
    from fant_slab_linear import install_fant_slab_linear_patch

    install_fant_slab_linear_patch(
        max_decode_tokens=int(
            os.getenv("GRSERVE_FANT_SLAB_LINEAR_MAX_TOKENS", "32")
        ),
        max_copies=int(os.getenv("GRSERVE_FANT_SLAB_LINEAR_MAX_COPIES", "8")),
    )

if os.getenv("GRSERVE_FANT_DOWN_INPLACE_LINEAR") == "1":
    from fant_down_inplace_linear import install_fant_down_inplace_linear_patch

    install_fant_down_inplace_linear_patch()
