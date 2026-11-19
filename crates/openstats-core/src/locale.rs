//! Regional conventions for showing and typing numbers and dates (Phase 6a). The data stay the same; only the
//! text the worksheet shows, the text typed into it and delimited files change.

use crate::datetime::DateOrder;

/// Decimal mark and day/month/year order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Locale {
    /// `.` or `,`.
    pub decimal: char,
    /// Order of the parts of a numeric date.
    pub date_order: DateOrder,
}

impl Default for Locale {
    fn default() -> Locale {
        Locale::UNITED_STATES
    }
}

impl Locale {
    /// 1.5 and 11/18/2026 (Minitab's default).
    pub const UNITED_STATES: Locale = Locale { decimal: '.', date_order: DateOrder::Mdy };
    /// 1.5 and 18/11/2026.
    pub const UNITED_KINGDOM: Locale = Locale { decimal: '.', date_order: DateOrder::Dmy };
    /// 1,5 and 18/11/2026 (Germany, France, Spain, Italy ...).
    pub const CONTINENTAL: Locale = Locale { decimal: ',', date_order: DateOrder::Dmy };
    /// 1.5 and 2026-11-18.
    pub const ISO: Locale = Locale { decimal: '.', date_order: DateOrder::Ymd };

    /// The presets in the order the Options dialog lists them.
    pub const PRESETS: [(&'static str, Locale); 4] = [
        ("United States (1.5, month/day/year)", Locale::UNITED_STATES),
        ("United Kingdom (1.5, day/month/year)", Locale::UNITED_KINGDOM),
        ("Continental Europe (1,5, day/month/year)", Locale::CONTINENTAL),
        ("ISO (1.5, year-month-day)", Locale::ISO),
    ];

    /// Index of this locale among [`Locale::PRESETS`].
    pub fn preset_index(&self) -> usize {
        Locale::PRESETS.iter().position(|(_, l)| l == self).unwrap_or(0)
    }

    /// Whether this is the default (United States) convention.
    pub fn is_default(&self) -> bool {
        *self == Locale::UNITED_STATES
    }

    /// `text` (formatted with `.`) with this locale's decimal mark.
    pub fn localize_number(&self, text: &str) -> String {
        if self.decimal == '.' {
            text.to_string()
        } else {
            text.replace('.', &self.decimal.to_string())
        }
    }

    /// Default display pattern of a date (and of a date with time).
    pub fn date_patterns(&self) -> (&'static str, &'static str) {
        match self.date_order {
            DateOrder::Mdy => (crate::datetime::DEFAULT_DATE_FORMAT, crate::datetime::DEFAULT_DATETIME_FORMAT),
            DateOrder::Dmy => ("d/M/yyyy", "d/M/yyyy H:mm:ss"),
            DateOrder::Ymd => ("yyyy-MM-dd", "yyyy-MM-dd H:mm:ss"),
        }
    }
}
