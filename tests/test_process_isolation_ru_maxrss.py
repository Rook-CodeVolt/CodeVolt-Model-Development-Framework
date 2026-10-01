"""Platform-specific regression coverage for ``ru_maxrss`` normalisation."""

from __future__ import annotations

import pytest

from codevolt_mdf.process_isolation import _ru_maxrss_to_mb


@pytest.mark.parametrize(
    ("platform_name", "ru_maxrss", "expected_mb"),
    [
        ("darwin", 6 * 1024 * 1024, 6.0),
        ("linux", 6 * 1024, 6.0),
    ],
)
def test_ru_maxrss_uses_platform_defined_units(
    platform_name: str,
    ru_maxrss: int,
    expected_mb: float,
):
    assert _ru_maxrss_to_mb(ru_maxrss, platform_name=platform_name) == expected_mb


def test_small_macos_byte_value_is_not_misclassified_as_kibibytes():
    ru_maxrss_bytes = 6_752 * 1024

    assert ru_maxrss_bytes < 10_000_000
    assert _ru_maxrss_to_mb(ru_maxrss_bytes, platform_name="darwin") == 6.59375
