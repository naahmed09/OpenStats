//! Tokeniser shared by the formula parser and the `I(...)` expression parser.

use crate::error::{invalid, Result};

/// A lexical token.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Tok {
    /// Bare or quoted name.
    Ident(String),
    /// Non-negative number.
    Num(f64),
    /// One of `+ - * / ^ : . | ( ) ~ ,`.
    Op(char),
}

/// A token with the whitespace flag the parser needs (`A(B)` nests, `A (B)` does not).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Token {
    pub tok: Tok,
    pub space_before: bool,
    pub pos: usize,
}

pub(crate) fn tokenize(src: &str) -> Result<Vec<Token>> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    let mut space = false;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            space = true;
            i += 1;
            continue;
        }
        let pos = i;
        let tok = if c.is_alphabetic() || c == '_' {
            let st = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            Tok::Ident(chars[st..i].iter().collect())
        } else if c.is_ascii_digit() {
            let st = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let mut j = i + 1;
                if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                    j += 1;
                }
                if j < chars.len() && chars[j].is_ascii_digit() {
                    while j < chars.len() && chars[j].is_ascii_digit() {
                        j += 1;
                    }
                    i = j;
                }
            }
            let text: String = chars[st..i].iter().collect();
            match text.parse::<f64>() {
                Ok(v) => Tok::Num(v),
                Err(_) => return invalid(format!("formula: bad number '{text}' at position {st}")),
            }
        } else if c == '`' || c == '"' || c == '\'' {
            let q = c;
            i += 1;
            let st = i;
            while i < chars.len() && chars[i] != q {
                i += 1;
            }
            if i >= chars.len() {
                return invalid(format!("formula: unterminated quoted name starting at position {pos}"));
            }
            let name: String = chars[st..i].iter().collect();
            i += 1;
            if name.is_empty() {
                return invalid("formula: empty quoted name");
            }
            Tok::Ident(name)
        } else if "+-*/^:.|()~,".contains(c) {
            i += 1;
            Tok::Op(c)
        } else {
            return invalid(format!("formula: unexpected character '{c}' at position {pos}"));
        };
        out.push(Token { tok, space_before: space, pos });
        space = false;
    }
    Ok(out)
}
