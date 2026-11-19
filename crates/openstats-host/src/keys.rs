//! Key codes on the wire: a code below `0x20_0000` is a Unicode scalar value (a typed
//! character); `SPECIAL + n` names a special key.

use openstats_ui::grid::Key;

/// First code of the special keys.
pub const SPECIAL: u32 = 0x20_0000;

/// Special key numbers, in order: Up, Down, Left, Right, Home, End, PageUp, PageDown, Enter, Tab,
/// Escape, Backspace, Delete, F2, F6, F10.
pub const NAMES: [&str; 16] = [
    "Up",
    "Down",
    "Left",
    "Right",
    "Home",
    "End",
    "PageUp",
    "PageDown",
    "Enter",
    "Tab",
    "Escape",
    "Backspace",
    "Delete",
    "F2",
    "F6",
    "F10",
];

/// The key for a wire code (`None` for a code that is neither a scalar value nor special).
pub fn decode(code: u32) -> Option<Key> {
    if code < SPECIAL {
        return char::from_u32(code).map(Key::Char);
    }
    Some(match code - SPECIAL {
        0 => Key::Up,
        1 => Key::Down,
        2 => Key::Left,
        3 => Key::Right,
        4 => Key::Home,
        5 => Key::End,
        6 => Key::PageUp,
        7 => Key::PageDown,
        8 => Key::Enter,
        9 => Key::Tab,
        10 => Key::Escape,
        11 => Key::Backspace,
        12 => Key::Delete,
        13 => Key::F2,
        14 => Key::F6,
        15 => Key::F10,
        16 => Key::F5,
        17 => Key::F8,
        18 => Key::F9,
        19 => Key::F1,
        _ => return None,
    })
}

/// The wire code of a key.
pub fn encode(k: Key) -> u32 {
    let n = match k {
        Key::Char(c) => return c as u32,
        Key::Up => 0,
        Key::Down => 1,
        Key::Left => 2,
        Key::Right => 3,
        Key::Home => 4,
        Key::End => 5,
        Key::PageUp => 6,
        Key::PageDown => 7,
        Key::Enter => 8,
        Key::Tab => 9,
        Key::Escape => 10,
        Key::Backspace => 11,
        Key::Delete => 12,
        Key::F2 => 13,
        Key::F6 => 14,
        Key::F10 => 15,
        Key::F5 => 16,
        Key::F8 => 17,
        Key::F9 => 18,
        Key::F1 => 19,
    };
    SPECIAL + n
}
