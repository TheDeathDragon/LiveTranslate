import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from subtitle_overlay import SubtitleOverlay
from subtitle_window import SubtitleWindow


@pytest.fixture(scope="module")
def qt():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def windows(qt):
    overlay = SubtitleOverlay({})
    subtitle = SubtitleWindow()
    yield overlay, subtitle
    overlay.close()
    subtitle.close()
    qt.processEvents()


def caption(windows, original, translation, revision=1, final=False, session=1, utterance=0):
    for w in windows:
        w.update_caption(session, utterance, revision, original, {"zh": translation}, final, 1)


def test_revision_is_immediate_without_history_or_animation(windows):
    overlay, subtitle = windows
    caption(windows, "Hello", "你好")
    caption(windows, "Hello there", "大家好", revision=2)
    assert not overlay._messages and not subtitle._sentences
    assert overlay._preview_message._original == "Hello there"
    assert subtitle._preview_sentence == ("Hello there", {"zh": "大家好"})
    assert not subtitle._pending_segment_timers
    assert any(w._text == "大家好" for w in subtitle._text_widgets)
    assert all(w._pending_text is None for w in subtitle._text_widgets)
    caption(windows, "Hello there", "大家好", revision=3, final=True)
    assert len(overlay._messages) == len(subtitle._sentences) == 1
    assert overlay._preview_message is None and subtitle._preview_sentence is None
    caption(windows, "late", "过期", revision=2)
    caption(windows, "Hello there", "大家好", revision=3, final=True)
    assert len(overlay._messages) == len(subtitle._sentences) == 1


def test_reset_rejects_old_session_and_does_not_clear_history(windows):
    overlay, subtitle = windows
    caption(windows, "final", "最终", final=True)
    for w in windows:
        w.reset_preview(2)
    caption(windows, "stale", "过期", session=1, utterance=1)
    assert overlay._preview_message is None and subtitle._preview_sentence is None
    assert len(overlay._messages) == len(subtitle._sentences) == 1


def test_empty_final_removes_preview_without_saving(windows):
    overlay, subtitle = windows
    caption(windows, "noise", "杂音")
    caption(windows, "", "", revision=2, final=True)
    assert not overlay._messages and not subtitle._sentences
    assert overlay._preview_message is None and subtitle._preview_sentence is None


def test_legacy_window_api_still_inserts_sentences(windows):
    overlay, subtitle = windows
    overlay.add_message(1, "12:00:00", "original", "en", 10)
    subtitle.update_text("original", {"zh": "原文"})
    assert len(overlay._messages) == len(subtitle._sentences) == 1


def test_live_revision_has_no_entry_animation(windows):
    for widget in windows[1]._text_widgets:
        widget._entry_animation = widget._exit_animation = "fade"
    caption(windows, "Hello", "你好")
    assert all(w._content_opacity_val == 1.0 for w in windows[1]._text_widgets)
    assert all(w._anim_group is None for w in windows[1]._text_widgets)


def test_revision_keeps_visible_text_without_flushing_old_pending_tokens(windows):
    overlay, _ = windows
    caption(windows, "old", "上一句")
    caption(windows, "old", "上一句完整")
    visible = overlay._preview_message._trans_label.text()
    caption(windows, "new", "", revision=2)
    QTest.qWait(80)
    assert overlay._preview_message._trans_label.text() == visible
    assert "上一句完整" not in overlay._preview_message._trans_label.text()


def test_pending_translation_does_not_blank_current_subtitle(windows):
    caption(windows, "Hello", "你好")
    caption(windows, "Hello there", "", revision=2)
    assert any(w._text == "你好" for w in windows[1]._text_widgets)
    caption(windows, "Hello there", "大家好", revision=2)
    assert any(w._text == "大家好" for w in windows[1]._text_widgets)


def test_next_sentence_keeps_previous_chinese_only_while_pending(windows):
    caption(windows, "Hello", "你好", final=True)
    caption(windows, "How are you", "", utterance=1)
    assert any(w._text == "How are you" for w in windows[1]._text_widgets)
    assert any(w._text == "你好" for w in windows[1]._text_widgets)
    caption(windows, "How are you", "你好吗", utterance=1)
    assert any(w._text == "你好吗" for w in windows[1]._text_widgets)
    caption(windows, "How are you", "", final=True, utterance=1)
    assert not any(w._text == "你好吗" for w in windows[1]._text_widgets)


def test_live_captions_roll_by_line_with_confirmed_sentence_unchanged(qt):
    subtitle = SubtitleWindow({"sentences": 2})
    try:
        subtitle.update_caption(1, 0, 1, "First", {"zh": "前一句"}, True)
        subtitle.update_caption(1, 1, 1, "Second", {"zh": "当前句"}, False)
        assert any(w._text == "前一句\n当前句" for w in subtitle._text_widgets)
        subtitle.update_caption(1, 1, 2, "Second revised", {"zh": "当前句修订"}, False)
        assert any(w._text == "前一句\n当前句修订" for w in subtitle._text_widgets)
        assert subtitle._sentences == [("First", {"zh": "前一句"})]
    finally:
        subtitle.close()


def test_older_final_preserves_newer_preview_at_bottom(windows):
    overlay, subtitle = windows
    caption(windows, "current", "当前", utterance=1)
    caption(windows, "previous", "上一句", final=True, utterance=0)
    assert overlay._preview_message._original == subtitle._preview_sentence[0] == "current"
    assert overlay._msg_layout.indexOf(overlay._preview_message) > overlay._msg_layout.indexOf(overlay._messages[1])


def test_failed_translation_finishes_without_a_permanent_pending_label(windows):
    caption(windows, "Hello", "你好")
    caption(windows, "Hello", "", revision=2, final=True)
    assert windows[0]._messages[1]._translated == ""
    assert windows[0]._messages[1]._trans_label.text() == ""
    assert windows[1]._sentences == [("Hello", {"zh": ""})]
