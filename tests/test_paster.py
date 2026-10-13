from whisper_daemon import paster


class Item:
    def __init__(self, values=None): self.values = values or {}
    @classmethod
    def alloc(cls): return cls()
    def init(self): return self
    def types(self): return list(self.values)
    def dataForType_(self, t): return self.values[t]
    def setData_forType_(self, data, t): self.values[t] = data; return True


class Board:
    def __init__(self):
        self.items = [Item({"public.png": b"image", "public.html": b"<b>text</b>"})]
        self.version = 0
    def pasteboardItems(self): return self.items
    def changeCount(self): return self.version
    def clearContents(self): self.items = []; self.version += 1
    def setString_forType_(self, s, t): self.items = [Item({t: s.encode()})]; self.version += 1
    def stringForType_(self, t): return None
    def writeObjects_(self, items): self.items = items; self.version += 1; return True


def setup(monkeypatch):
    pb = Board()
    monkeypatch.setattr(paster, "NSPasteboard", type("Pasteboard", (), {"generalPasteboard": lambda: pb}))
    monkeypatch.setattr(paster, "NSPasteboardItem", Item, raising=False)
    monkeypatch.setattr(paster, "_simulate_cmd_v", lambda: None)
    monkeypatch.setattr(paster.time, "sleep", lambda _: None)
    return pb


def test_paste_restores_all_clipboard_formats(monkeypatch):
    pb = setup(monkeypatch)
    paster.paste_text("dictation")
    assert len(pb.items) == 1
    assert pb.items[0].values == {"public.png": b"image", "public.html": b"<b>text</b>"}


def test_paste_does_not_overwrite_new_user_copy(monkeypatch):
    pb = setup(monkeypatch)
    def copy_during_paste(_):
        pb.clearContents()
        pb.setString_forType_("new copy", "public.text")
    monkeypatch.setattr(paster.time, "sleep", copy_during_paste)
    paster.paste_text("dictation")
    assert pb.items[0].values == {"public.text": b"new copy"}


def test_changed_input_prevents_any_clipboard_mutation(monkeypatch):
    from whisper_daemon import focus
    pb = setup(monkeypatch)
    monkeypatch.setattr(focus, 'target_matches', lambda _: False)
    assert paster.paste_text('should not paste', target=object()) is False
    assert pb.version == 0
