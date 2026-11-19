//! winit keys → the wire codes `openstats-host` understands (see `openstats_host::keys`).

use openstats_host::keys::SPECIAL;
use winit::event::KeyEvent;
use winit::keyboard::{Key, ModifiersState, NamedKey};

/// Modifier bits: 1 shift, 2 ctrl (Cmd on macOS), 4 alt.
pub fn mod_bits(m: ModifiersState) -> u32 {
    let command = if cfg!(target_os = "macos") { m.super_key() } else { m.control_key() };
    u32::from(m.shift_key()) | (u32::from(command) << 1) | (u32::from(m.alt_key()) << 2)
}

/// The wire code of a key press, or `None` for keys the workspace does not use (modifiers, media
/// keys…). Printable keys are sent as their character: the typed text when no command key is
/// held, the unshifted key otherwise so shortcuts (Cmd+S) match.
pub fn key_code(ev: &KeyEvent, mods: ModifiersState) -> Option<u32> {
    let named = |n: u32| Some(SPECIAL + n);
    match &ev.logical_key {
        Key::Named(NamedKey::ArrowUp) => named(0),
        Key::Named(NamedKey::ArrowDown) => named(1),
        Key::Named(NamedKey::ArrowLeft) => named(2),
        Key::Named(NamedKey::ArrowRight) => named(3),
        Key::Named(NamedKey::Home) => named(4),
        Key::Named(NamedKey::End) => named(5),
        Key::Named(NamedKey::PageUp) => named(6),
        Key::Named(NamedKey::PageDown) => named(7),
        Key::Named(NamedKey::Enter) => named(8),
        Key::Named(NamedKey::Tab) => named(9),
        Key::Named(NamedKey::Escape) => named(10),
        Key::Named(NamedKey::Backspace) => named(11),
        Key::Named(NamedKey::Delete) => named(12),
        Key::Named(NamedKey::F2) => named(13),
        Key::Named(NamedKey::F6) => named(14),
        Key::Named(NamedKey::F10) => named(15),
        Key::Named(NamedKey::F5) => named(16),
        Key::Named(NamedKey::F8) => named(17),
        Key::Named(NamedKey::F9) => named(18),
        Key::Named(NamedKey::F1) => named(19),
        Key::Named(NamedKey::Space) => Some(' ' as u32),
        Key::Character(s) => {
            let from_key = || single_char(s.as_str());
            if mod_bits(mods) & 2 != 0 {
                from_key()
            } else {
                ev.text.as_deref().and_then(single_char).or_else(from_key)
            }
        }
        _ => None,
    }
}

fn single_char(s: &str) -> Option<u32> {
    let mut it = s.chars();
    match (it.next(), it.next()) {
        (Some(c), None) if !c.is_control() => Some(c as u32),
        _ => None,
    }
}
