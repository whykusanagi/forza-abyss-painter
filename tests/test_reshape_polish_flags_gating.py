"""Pin Tier-B feature flags at True after Tier B smoke-tested plumbing
landed (CLAUDE.md §1c). The two feature flags are now permanently True
in the EXE build. If you find yourself reverting to False, that means
the feature broke in production and you should also revert the wiring
(do not ship half-flipped flags).
"""
from forza_abyss_painter.gui import feature_flags


def test_reshape_gen_flag_default_is_true():
    assert feature_flags.RESHAPE_GEN_AVAILABLE is True


def test_polish_loaded_flag_default_is_true():
    assert feature_flags.POLISH_LOADED_AVAILABLE is True
