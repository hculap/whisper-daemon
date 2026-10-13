"""User-facing shortcut names mapped to macOS physical key codes."""
KEYS = dict(zip('asdfhgzxcvbqwerty', (0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16)))
KEYS.update(dict(zip('ouiplnjkm', (31,32,34,35,37,45,38,40,46))))
KEYS.update(space=49, enter=36, escape=53)
MODIFIERS = {'cmd': (1 << 20, 0x100), 'shift': (1 << 17, 0x200),
             'alt': (1 << 19, 0x800), 'ctrl': (1 << 18, 0x1000)}


def parse_shortcut(value):
    parts = [p.strip().lower() for p in value.split('+')]
    if len(parts) < 2 or parts[-1] not in KEYS or len(set(parts)) != len(parts):
        raise ValueError('Use modifiers and a key, for example cmd+shift+space')
    ns_flags = carbon_flags = 0
    for modifier in parts[:-1]:
        if modifier not in MODIFIERS: raise ValueError('Modifiers: cmd, shift, alt, ctrl')
        ns, carbon = MODIFIERS[modifier]
        ns_flags |= ns
        carbon_flags |= carbon
    return ns_flags, KEYS[parts[-1]], carbon_flags
