//! Formula tokenizer.

use crate::error::{CoreError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Number(f64),
    /// `"double-quoted"` text literal.
    Text(String),
    /// `'single-quoted'` column name.
    QuotedName(String),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Pow,
    LParen,
    RParen,
    Comma,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Not,
}

/// A token with its byte offset.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub pos: usize,
}

fn err(msg: impl Into<String>, pos: usize) -> CoreError {
    CoreError::parse("formula", msg, Some(pos))
}

/// Tokenize a formula.
pub fn tokenize(src: &str) -> Result<Vec<Token>> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let pos = i;
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let tok = match c {
            b'0'..=b'9' | b'.' if c != b'.' || b.get(i + 1).is_some_and(|d| d.is_ascii_digit()) => {
                let st = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                if i < b.len() && b[i] == b'.' {
                    i += 1;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                    let mut j = i + 1;
                    if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
                        j += 1;
                    }
                    if j < b.len() && b[j].is_ascii_digit() {
                        i = j;
                        while i < b.len() && b[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                let s = &src[st..i];
                Tok::Number(s.parse().map_err(|_| err(format!("bad number {s:?}"), st))?)
            }
            b'"' | b'\'' => {
                let q = c;
                i += 1;
                let mut s = String::new();
                loop {
                    let rest = &src[i..];
                    let Some(off) = rest.find(q as char) else {
                        return Err(err("unterminated quote", pos));
                    };
                    s.push_str(&rest[..off]);
                    i += off + 1;
                    if b.get(i) == Some(&q) {
                        s.push(q as char);
                        i += 1;
                    } else {
                        break;
                    }
                }
                if q == b'"' {
                    Tok::Text(s)
                } else {
                    Tok::QuotedName(s)
                }
            }
            b'+' => {
                i += 1;
                Tok::Plus
            }
            b'-' => {
                i += 1;
                Tok::Minus
            }
            b'*' => {
                if b.get(i + 1) == Some(&b'*') {
                    i += 2;
                    Tok::Pow
                } else {
                    i += 1;
                    Tok::Star
                }
            }
            b'^' => {
                i += 1;
                Tok::Pow
            }
            b'/' => {
                i += 1;
                Tok::Slash
            }
            b'(' => {
                i += 1;
                Tok::LParen
            }
            b')' => {
                i += 1;
                Tok::RParen
            }
            b',' => {
                i += 1;
                Tok::Comma
            }
            b'&' => {
                i += if b.get(i + 1) == Some(&b'&') { 2 } else { 1 };
                Tok::And
            }
            b'|' => {
                i += if b.get(i + 1) == Some(&b'|') { 2 } else { 1 };
                Tok::Or
            }
            b'=' => {
                i += if b.get(i + 1) == Some(&b'=') { 2 } else { 1 };
                Tok::Eq
            }
            b'!' | b'~' => {
                if b.get(i + 1) == Some(&b'=') {
                    i += 2;
                    Tok::Ne
                } else {
                    i += 1;
                    Tok::Not
                }
            }
            b'<' => match b.get(i + 1) {
                Some(b'=') => {
                    i += 2;
                    Tok::Le
                }
                Some(b'>') => {
                    i += 2;
                    Tok::Ne
                }
                _ => {
                    i += 1;
                    Tok::Lt
                }
            },
            b'>' => {
                if b.get(i + 1) == Some(&b'=') {
                    i += 2;
                    Tok::Ge
                } else {
                    i += 1;
                    Tok::Gt
                }
            }
            c if c.is_ascii_alphabetic() || c == b'_' || c >= 0x80 => {
                let st = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.' || b[i] >= 0x80) {
                    i += 1;
                }
                let word = &src[st..i];
                match word.to_ascii_uppercase().as_str() {
                    "AND" => Tok::And,
                    "OR" => Tok::Or,
                    "NOT" => Tok::Not,
                    "EQ" => Tok::Eq,
                    "NE" => Tok::Ne,
                    "LT" => Tok::Lt,
                    "LE" => Tok::Le,
                    "GT" => Tok::Gt,
                    "GE" => Tok::Ge,
                    _ => Tok::Ident(word.to_string()),
                }
            }
            _ => {
                let ch = src[i..].chars().next().unwrap_or('?');
                return Err(err(format!("unexpected character {ch:?}"), pos));
            }
        };
        out.push(Token { tok, pos });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<Tok> {
        tokenize(s).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn tokens() {
        assert_eq!(
            toks("C1**2 * * <> 'My Col' ~= \"a\"\"b\""),
            vec![
                Tok::Ident("C1".into()),
                Tok::Pow,
                Tok::Number(2.0),
                Tok::Star,
                Tok::Star,
                Tok::Ne,
                Tok::QuotedName("My Col".into()),
                Tok::Ne,
                Tok::Text("a\"b".into())
            ]
        );
        assert_eq!(toks("x and Not y"), vec![Tok::Ident("x".into()), Tok::And, Tok::Not, Tok::Ident("y".into())]);
        assert_eq!(
            toks(".5e-1 1E3 2e"),
            vec![Tok::Number(0.05), Tok::Number(1000.0), Tok::Number(2.0), Tok::Ident("e".into())]
        );
        assert!(tokenize("'open").is_err());
        assert!(tokenize("x # y").is_err());
    }
}
