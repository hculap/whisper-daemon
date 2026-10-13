import pytest
from whisper_daemon.shortcuts import parse_shortcut


def test_shortcuts_are_physical_and_modifiers_exact():
    ns, code, carbon = parse_shortcut('cmd+shift+space')
    assert code == 49 and carbon == 0x300 and ns == (1 << 20 | 1 << 17)
    assert parse_shortcut('ctrl+alt+escape')[1] == 53


@pytest.mark.parametrize('shortcut', ['cmd+cmd+v', 'x', 'cmd+unsupported', 'wrong+space'])
def test_invalid_shortcut_rejected(shortcut):
    with pytest.raises(ValueError): parse_shortcut(shortcut)
