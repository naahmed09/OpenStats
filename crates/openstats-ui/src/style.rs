//! Interface colours for the light and dark themes.
//!
//! Charts take their colours from [`openstats_charts::core::Theme`]. The window chrome — desk,
//! menu bar, grid headers, buttons, selection and accent — needs a few more, and they must
//! change with the theme. [`Chrome::of`] derives them from a chart theme, so every component
//! (grid, menus, Navigator, Session window, dialogs) agrees on one definition instead of
//! carrying its own hexadecimal literals.
//!
//! The colours are chosen for legibility: [`contrast_ratio`] implements the WCAG 2.x relative
//! luminance formula, and the tests require body text to reach 4.5:1 and the accent and
//! alert colours 3:1 against the background in both themes.

use openstats_charts::core::{Color, Theme};

/// Which of the built-in themes is in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeKind {
    /// Dark text on a light background.
    Light,
    /// Light text on a dark background.
    Dark,
    /// Pure white on pure black with a yellow accent, for low vision and bright rooms.
    HighContrast,
}

impl ThemeKind {
    /// The chart theme for this kind.
    pub fn theme(self) -> Theme {
        match self {
            ThemeKind::Light => Theme::light(),
            ThemeKind::Dark => Theme::dark(),
            ThemeKind::HighContrast => Theme::high_contrast(),
        }
    }

    /// Stable lower-case name (`light`, `dark`, `high-contrast`) for saved settings.
    pub fn name(self) -> &'static str {
        match self {
            ThemeKind::Light => "light",
            ThemeKind::Dark => "dark",
            ThemeKind::HighContrast => "high-contrast",
        }
    }

    /// The kind called `name` (see [`ThemeKind::name`]).
    pub fn from_name(name: &str) -> Option<ThemeKind> {
        [ThemeKind::Light, ThemeKind::Dark, ThemeKind::HighContrast].into_iter().find(|k| k.name() == name.trim())
    }

    /// Menu label.
    pub fn label(self) -> &'static str {
        match self {
            ThemeKind::Light => "Light",
            ThemeKind::Dark => "Dark",
            ThemeKind::HighContrast => "High Contrast",
        }
    }

    /// The kind whose theme `theme` resembles: dark when its background is dark.
    pub fn of(theme: &Theme) -> ThemeKind {
        if theme.background == Color::BLACK && theme.ink == Color::WHITE {
            ThemeKind::HighContrast
        } else if luminance(theme.background) < 0.18 {
            ThemeKind::Dark
        } else {
            ThemeKind::Light
        }
    }

    /// The next theme in the cycle light → dark → high contrast → light.
    pub fn toggled(self) -> ThemeKind {
        match self {
            ThemeKind::Light => ThemeKind::Dark,
            ThemeKind::Dark => ThemeKind::HighContrast,
            ThemeKind::HighContrast => ThemeKind::Light,
        }
    }
}

/// Colours of the window chrome for one theme.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chrome {
    /// Whether this is the dark variant.
    pub dark: bool,
    /// The gap between panes.
    pub desk: Color,
    /// Menu bar, formula bar and grid headers.
    pub bar: Color,
    /// Selected grid header.
    pub header_selected: Color,
    /// An idle button or tab.
    pub button: Color,
    /// The accent: focus rings, title bars, primary buttons, caret.
    pub accent: Color,
    /// Text drawn on top of the accent.
    pub on_accent: Color,
    /// Cell selection and menu highlight fill (translucent accent).
    pub selection: Color,
    /// Secondary text.
    pub muted: Color,
    /// Navigator marker for worksheets.
    pub worksheet: Color,
    /// Navigator marker for graphs.
    pub graph: Color,
    /// Navigator marker for Session windows.
    pub session: Color,
    /// Navigator marker for macro scripts.
    pub script: Color,
    /// Navigator marker for folders and other rows.
    pub folder: Color,
}

impl Chrome {
    /// The chrome colours that go with `theme`.
    pub fn of(theme: &Theme) -> Chrome {
        match ThemeKind::of(theme) {
            ThemeKind::Light => Chrome::light(),
            ThemeKind::Dark => Chrome::dark(),
            ThemeKind::HighContrast => Chrome::high_contrast(),
        }
    }

    /// Light variant.
    pub fn light() -> Chrome {
        let accent = Color::hex(0x0072B2);
        Chrome {
            dark: false,
            desk: Color::hex(0xDADADA),
            bar: Color::hex(0xF0F0F0),
            header_selected: Color::hex(0xD6E4F5),
            button: Color::hex(0xE8E8E8),
            accent,
            on_accent: Color::WHITE,
            selection: accent.with_alpha(0.16),
            muted: Color::hex(0x555555),
            worksheet: accent,
            graph: Color::hex(0xD55E00),
            session: Color::hex(0x009E73),
            script: Color::hex(0x8E44AD),
            folder: Color::hex(0x7F7F7F),
        }
    }

    /// High contrast: black surfaces, white text, a yellow accent and bright markers.
    pub fn high_contrast() -> Chrome {
        let accent = Color::hex(0xFFE000);
        Chrome {
            dark: true,
            desk: Color::BLACK,
            bar: Color::hex(0x161616),
            header_selected: Color::hex(0x3A3A00),
            button: Color::hex(0x2A2A2A),
            accent,
            on_accent: Color::BLACK,
            selection: accent.with_alpha(0.35),
            muted: Color::hex(0xEDEDED),
            worksheet: Color::hex(0x56D4FF),
            graph: Color::hex(0xFFA040),
            session: Color::hex(0x00FF9C),
            script: Color::hex(0xE0A0FF),
            folder: Color::hex(0xD0D0D0),
        }
    }

    /// Dark variant: a lighter accent so it keeps its contrast on dark surfaces.
    pub fn dark() -> Chrome {
        let accent = Color::hex(0x56B4E9);
        Chrome {
            dark: true,
            desk: Color::hex(0x121212),
            bar: Color::hex(0x2B2B2B),
            header_selected: Color::hex(0x3A4A5E),
            button: Color::hex(0x3A3A3A),
            accent,
            on_accent: Color::hex(0x101010),
            selection: accent.with_alpha(0.28),
            muted: Color::hex(0xB0B0B0),
            worksheet: accent,
            graph: Color::hex(0xE69F00),
            session: Color::hex(0x5FD3A8),
            script: Color::hex(0xC792EA),
            folder: Color::hex(0x9A9A9A),
        }
    }
}

fn channel(v: u8) -> f64 {
    let c = f64::from(v) / 255.0;
    if c <= 0.03928 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// WCAG relative luminance of an opaque colour, 0 (black) to 1 (white).
pub fn luminance(c: Color) -> f64 {
    0.2126 * channel(c.r) + 0.7152 * channel(c.g) + 0.0722 * channel(c.b)
}

/// WCAG contrast ratio between two opaque colours, 1 (identical) to 21 (black on white).
pub fn contrast_ratio(a: Color, b: Color) -> f64 {
    let (la, lb) = (luminance(a), luminance(b));
    let (hi, lo) = if la >= lb { (la, lb) } else { (lb, la) };
    (hi + 0.05) / (lo + 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contrast_reference_values() {
        assert!((contrast_ratio(Color::BLACK, Color::WHITE) - 21.0).abs() < 1e-9);
        assert!((contrast_ratio(Color::WHITE, Color::WHITE) - 1.0).abs() < 1e-12);
        // #767676 on white is the classic 4.54:1 boundary grey.
        assert!((contrast_ratio(Color::hex(0x767676), Color::WHITE) - 4.54).abs() < 0.01);
    }

    #[test]
    fn kinds_round_trip() {
        assert_eq!(ThemeKind::of(&Theme::light()), ThemeKind::Light);
        assert_eq!(ThemeKind::of(&Theme::dark()), ThemeKind::Dark);
        assert_eq!(ThemeKind::Light.toggled(), ThemeKind::Dark);
        assert_eq!(ThemeKind::Dark.toggled(), ThemeKind::HighContrast);
        assert_eq!(ThemeKind::HighContrast.toggled(), ThemeKind::Light);
        assert_eq!(ThemeKind::of(&Theme::high_contrast()), ThemeKind::HighContrast);
        assert_eq!(Chrome::of(&Theme::high_contrast()), Chrome::high_contrast());
        for k in [ThemeKind::Light, ThemeKind::Dark, ThemeKind::HighContrast] {
            assert_eq!(ThemeKind::from_name(k.name()), Some(k));
        }
        assert_eq!(ThemeKind::from_name("sepia"), None);
        assert_eq!(ThemeKind::Dark.theme(), Theme::dark());
        assert_eq!(Chrome::of(&Theme::dark()), Chrome::dark());
        assert_eq!(Chrome::of(&Theme::light()), Chrome::light());
    }

    #[test]
    fn both_themes_are_legible() {
        for kind in [ThemeKind::Light, ThemeKind::Dark, ThemeKind::HighContrast] {
            let hc = kind == ThemeKind::HighContrast;
            let body = if hc { 7.0 } else { 4.5 };
            let t = kind.theme();
            let c = Chrome::of(&t);
            let text = [("ink", t.ink), ("muted", c.muted)];
            for (name, col) in text {
                for (surface, bg) in [("background", t.background), ("bar", c.bar), ("button", c.button)] {
                    let r = contrast_ratio(col, bg);
                    assert!(r >= body, "{kind:?}: {name} on {surface} is {r:.2}:1");
                }
            }
            for (name, col) in [("accent", c.accent), ("alert", t.alert), ("limit", t.limit)] {
                let r = contrast_ratio(col, t.background);
                assert!(r >= 3.0, "{kind:?}: {name} on background is {r:.2}:1");
            }
            let r = contrast_ratio(c.on_accent, c.accent);
            assert!(r >= 4.5, "{kind:?}: text on the accent is {r:.2}:1");
            for (name, col) in [("worksheet", c.worksheet), ("graph", c.graph), ("session", c.session)] {
                let r = contrast_ratio(col, t.background);
                assert!(r >= 3.0, "{kind:?}: {name} marker is {r:.2}:1");
            }
        }
    }
}
