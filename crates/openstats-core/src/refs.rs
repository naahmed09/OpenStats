//! Minitab-style references: `C3`, `C1-C5`, `'Part Number'`, `K2`, `M1`.

use crate::error::{CoreError, Result};

/// A parsed reference token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ref {
    /// Column by 1-based number (`C7`).
    Column(usize),
    /// Inclusive column range (`C1-C4`, `C4-C1` is a descending range).
    ColumnRange(usize, usize),
    /// Column by name (`'Weight (g)'` or bare `Weight`).
    Name(String),
    /// Stored constant (`K1`).
    Constant(usize),
    /// Stored matrix (`M1`).
    Matrix(usize),
}

fn parse_index(prefix: char, s: &str) -> Option<usize> {
    let mut it = s.chars();
    let c = it.next()?;
    if c.to_ascii_uppercase() != prefix {
        return None;
    }
    let rest = it.as_str();
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: usize = rest.parse().ok()?;
    if n == 0 {
        None
    } else {
        Some(n)
    }
}

/// Parse a single reference token.
pub fn parse_ref(s: &str) -> Result<Ref> {
    let t = s.trim();
    if t.is_empty() {
        return Err(CoreError::parse("reference", "empty reference", None));
    }
    if let Some(inner) = t.strip_prefix('\'') {
        let inner = inner
            .strip_suffix('\'')
            .ok_or_else(|| CoreError::parse("reference", format!("unterminated quote in {s:?}"), None))?;
        // '' inside quotes is an escaped quote
        return Ok(Ref::Name(inner.replace("''", "'")));
    }
    if let Some((a, b)) = t.split_once('-') {
        if let (Some(x), Some(y)) = (parse_index('C', a.trim()), parse_index('C', b.trim())) {
            return Ok(Ref::ColumnRange(x, y));
        }
    }
    if let Some(n) = parse_index('C', t) {
        return Ok(Ref::Column(n));
    }
    if let Some(n) = parse_index('K', t) {
        return Ok(Ref::Constant(n));
    }
    if let Some(n) = parse_index('M', t) {
        return Ok(Ref::Matrix(n));
    }
    Ok(Ref::Name(t.to_string()))
}

/// Split a reference list such as `C1-C3 'Batch No' C7` into tokens,
/// respecting quotes.
pub fn split_ref_list(s: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = s.chars().peekable();
    let mut in_quote = false;
    while let Some(c) = chars.next() {
        if in_quote {
            cur.push(c);
            if c == '\'' {
                if chars.peek() == Some(&'\'') {
                    cur.push(chars.next().expect("peeked"));
                } else {
                    in_quote = false;
                    out.push(std::mem::take(&mut cur));
                }
            }
        } else if c == '\'' {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            cur.push(c);
            in_quote = true;
        } else if c.is_whitespace() || c == ',' {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else if c == '-' && cur.is_empty() && out.last().is_some_and(|l| parse_index('C', l).is_some()) {
            // "C1 - C3" written with spaces around the dash
            let prev = out.pop().expect("checked");
            cur = format!("{prev}-");
        } else {
            cur.push(c);
        }
    }
    if in_quote {
        return Err(CoreError::parse("reference", format!("unterminated quote in {s:?}"), None));
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    // Join a dangling "C1-" with the next token.
    let mut merged: Vec<String> = Vec::with_capacity(out.len());
    for tok in out {
        if let Some(last) = merged.last_mut() {
            if last.ends_with('-') {
                last.push_str(&tok);
                continue;
            }
        }
        merged.push(tok);
    }
    Ok(merged)
}

/// Format a column name for use in a command: quoted if it is not a plain
/// identifier or could be mistaken for a reference.
pub fn quote_name(name: &str) -> String {
    let plain = !name.is_empty()
        && name.chars().all(|c| c.is_alphanumeric() || c == '_')
        && !name.chars().next().is_some_and(|c| c.is_ascii_digit())
        && parse_index('C', name).is_none()
        && parse_index('K', name).is_none()
        && parse_index('M', name).is_none();
    if plain {
        name.to_string()
    } else {
        format!("'{}'", name.replace('\'', "''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tokens() {
        assert_eq!(parse_ref("c12").unwrap(), Ref::Column(12));
        assert_eq!(parse_ref("C1-C3").unwrap(), Ref::ColumnRange(1, 3));
        assert_eq!(parse_ref("K2").unwrap(), Ref::Constant(2));
        assert_eq!(parse_ref("m1").unwrap(), Ref::Matrix(1));
        assert_eq!(parse_ref("'It''s'").unwrap(), Ref::Name("It's".into()));
        assert_eq!(parse_ref("Weight").unwrap(), Ref::Name("Weight".into()));
        assert_eq!(parse_ref("C0").unwrap(), Ref::Name("C0".into()));
        assert!(parse_ref("'oops").is_err());
    }

    #[test]
    fn split_lists() {
        assert_eq!(split_ref_list("C1-C3 'Batch No', C7").unwrap(), vec!["C1-C3", "'Batch No'", "C7"]);
        assert_eq!(split_ref_list("C1 - C3 x").unwrap(), vec!["C1-C3", "x"]);
        assert_eq!(split_ref_list("'a''b' c").unwrap(), vec!["'a''b'", "c"]);
    }

    #[test]
    fn quoting() {
        assert_eq!(quote_name("Weight"), "Weight");
        assert_eq!(quote_name("Part No"), "'Part No'");
        assert_eq!(quote_name("C3"), "'C3'");
        assert_eq!(quote_name("O'Neil"), "'O''Neil'");
    }
}
