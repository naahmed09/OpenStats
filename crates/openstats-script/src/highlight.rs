//! Syntax classes for editor colouring.
//!
//! [`highlight_line`] classifies one line on its own; the editor keeps no state between lines
//! (inline data lines are coloured like any other text).

use crate::commands::canonical;

/// What a stretch of a line is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Ordinary text.
    Plain,
    /// `#` comment or `NOTE` line.
    Comment,
    /// Control-flow and declaration words (`DO`, `IF`, `MACRO`, …).
    Keyword,
    /// A command word.
    Command,
    /// A number.
    Number,
    /// `"text"`.
    Text,
    /// `'column name'`.
    Column,
    /// `C1`, `K2`, `M3`.
    Variable,
    /// An operator or punctuation.
    Operator,
    /// `%name` macro call or `MLABEL` name.
    Call,
}

/// A coloured stretch: byte range of the line and its class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// First byte.
    pub start: usize,
    /// One past the last byte.
    pub end: usize,
    /// Class.
    pub class: Class,
}

/// Words of the language that are not commands.
pub const KEYWORDS: &[&str] = &[
    "GMACRO",
    "MACRO",
    "ENDMACRO",
    "DO",
    "ENDDO",
    "WHILE",
    "ENDWHILE",
    "IF",
    "ELSEIF",
    "ELSE",
    "ENDIF",
    "GOTO",
    "MLABEL",
    "RETURN",
    "STOP",
    "END",
    "ARGS",
    "MCOLUMN",
    "MCONSTANT",
    "MMATRIX",
    "MTEXT",
    "MSTORE",
    "MEND",
    "LABEL",
    "LET",
    "BY",
    "AND",
    "OR",
    "NOT",
    "NARGS",
];

fn is_variable(w: &str) -> bool {
    let b = w.as_bytes();
    b.len() > 1
        && matches!(b[0].to_ascii_uppercase(), b'C' | b'K' | b'M' | b'S')
        && b[1..].iter().all(u8::is_ascii_digit)
}

/// Classify the line.
pub fn highlight_line(line: &str) -> Vec<Span> {
    let b = line.as_bytes();
    let mut out: Vec<Span> = Vec::new();
    let mut push = |start: usize, end: usize, class: Class| {
        if end > start {
            out.push(Span { start, end, class });
        }
    };
    let first = line.trim_start();
    let lead = line.len() - first.len();
    if first.starts_with('#')
        || first.starts_with("//")
        || first.split_whitespace().next().is_some_and(|w| w.eq_ignore_ascii_case("NOTE"))
    {
        push(lead, line.len(), Class::Comment);
        return out;
    }
    let mut i = 0;
    let mut first_word = true;
    while i < b.len() {
        let ch = b[i];
        if ch.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        match ch {
            b'#' => {
                push(i, b.len(), Class::Comment);
                break;
            }
            b'/' if b.get(i + 1) == Some(&b'/') => {
                push(i, b.len(), Class::Comment);
                break;
            }
            b'"' | b'\'' => {
                let q = ch;
                i += 1;
                while i < b.len() {
                    if b[i] == q {
                        if b.get(i + 1) == Some(&q) {
                            i += 2;
                            continue;
                        }
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                push(start, i, if q == b'"' { Class::Text } else { Class::Column });
            }
            b'%' => {
                i += 1;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || matches!(b[i], b'_' | b'.' | b'/' | b'\\')) {
                    i += 1;
                }
                push(start, i, Class::Call);
            }
            b'0'..=b'9' => {
                while i < b.len()
                    && (b[i].is_ascii_digit() || b[i] == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit))
                {
                    i += 1;
                }
                if i < b.len() && matches!(b[i], b'e' | b'E') {
                    let mut j = i + 1;
                    if j < b.len() && matches!(b[j], b'+' | b'-') {
                        j += 1;
                    }
                    if j < b.len() && b[j].is_ascii_digit() {
                        while j < b.len() && b[j].is_ascii_digit() {
                            j += 1;
                        }
                        i = j;
                    }
                }
                push(start, i, Class::Number);
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                let w = &line[start..i];
                let up = w.to_ascii_uppercase();
                let class = if is_variable(w) {
                    Class::Variable
                } else if KEYWORDS.contains(&up.as_str()) {
                    Class::Keyword
                } else if first_word && canonical(w).is_some() {
                    Class::Command
                } else {
                    Class::Plain
                };
                push(start, i, class);
            }
            _ => {
                i += 1;
                push(start, i, Class::Operator);
            }
        }
        first_word = false;
    }
    out
}
