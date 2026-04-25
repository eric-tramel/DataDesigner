# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math


def validate_finite_number(
    field_name: str,
    value: int | float,
    *,
    error_type: type[Exception],
    error_label: str,
) -> None:
    """Raise the module-specific Ray error when a numeric value is not finite."""
    if not math.isfinite(value):
        raise error_type(f"{error_label} {field_name} must be finite.")
