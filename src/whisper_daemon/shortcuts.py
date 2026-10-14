"""User-facing shortcut names mapped to macOS physical key codes."""
KEYS = {'a':0,'s':1,'d':2,'f':3,'h':4,'g':5,'z':6,'x':7,'c':8,'v':9,
        'b':11,'q':12,'w':13,'e':14,'r':15,'y':16,'t':17,'o':31,'u':32,
        'i':34,'p':35,'l':37,'j':38,'k':40,'n':45,'m':46}
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
