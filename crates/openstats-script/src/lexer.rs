//! Lexer: source text → tokens.
//!
//! * `#` and `//` start a comment that runs to the end of the line; a statement that begins with
//!   `NOTE` is a comment too (its text is not tokenised, so it may contain quotes).
//! * A line that ends with `&` continues on the next line.
//! * `"text"` and `'text'` are quoted items (`""` / `''` is a quote inside); the lexer reports
//!   `'…'` as [`Tok::Quoted`] and the interpreter decides whether it names a column or is text.
//! * Numbers: `12`, `1.5`, `.5`, `2e-3`. A lone `*` is the missing-value marker or the multiply
//!   operator, which the parser decides from context.
//! * Words are letters, digits, `_` and `$`, starting with a letter or `_`; `K1`, `C12` and `M3`
//!   are ordinary words that the parser recognises.
//! * Every token records whether it was preceded by white space, which the command-argument
//!   parser uses to tell `C1-C5` (a range) from `C1 -5`.

use crate::error::ScriptError;

/// A token kind.
#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    /// A word (identifier or keyword), original case.
    Word(String),
    /// A number.
    Num(f64),
    /// `"text"`.
    Str(String),
    /// `'column name'`.
    Quoted(String),
    /// An operator or punctuation mark.
    Sym(&'static str),
    /// End of a logical line.
    Eol,
    /// End of input.
    Eof,
}

/// A token with its position.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// What it is.
    pub tok: Tok,
    /// 1-based line of the token's first character.
    pub line: usize,
    /// 1-based column of the first character.
    pub col: usize,
    /// Column just past the last character (same line).
    pub end: usize,
    /// White space (or the start of the line) came directly before it.
    pub space_before: bool,
}

const SYMS: &[&str] = &[
    "**", "==", "<>", "!=", "<=", ">=", "+", "-", "*", "/", "^", "(", ")", "=", "<", ">", ",", ":", ";", ".", "&", "|",
    "~", "%",
];

/// Tokenize `src`. In `lenient` mode (used for syntax highlighting) an unterminated string or an
/// unknown character never fails.
pub fn lex(src: &str, lenient: bool) -> Result<Vec<Token>, ScriptError> {
    let mut out = Vec::new();
    let mut line_no = 0usize;
    for raw in src.split('\n') {
        line_no += 1;
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0usize;
        let mut space = true;
        let mut cont_here = false;
        while i < chars.len() {
            let c = chars[i];
            if c == ' ' || c == '\t' {
                space = true;
                i += 1;
                continue;
            }
            if c == '#' || (c == '/' && chars.get(i + 1) == Some(&'/')) {
                break;
            }
            let start = i;
            let col = i + 1;
            let tok = if c == '"' || c == '\'' {
                let mut s = String::new();
                i += 1;
                let mut closed = false;
                while i < chars.len() {
                    if chars[i] == c {
                        if i + 1 < chars.len() && chars[i + 1] == c {
                            s.push(c);
                            i += 2;
                            continue;
                        }
                        i += 1;
                        closed = true;
                        break;
                    }
                    s.push(chars[i]);
                    i += 1;
                }
                if !closed && !lenient {
                    let what = if c == '"' { "text" } else { "column name" };
                    return Err(ScriptError::syntax(line_no, col, format!("the {what} is not closed")));
                }
                if c == '"' {
                    Tok::Str(s)
                } else {
                    Tok::Quoted(s)
                }
            } else if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(char::is_ascii_digit)) {
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                if i < chars.len() && chars[i] == '.' && chars.get(i + 1).is_some_and(char::is_ascii_digit) {
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
                let text: String = chars[start..i].iter().collect();
                match text.parse::<f64>() {
                    Ok(v) => Tok::Num(v),
                    Err(_) if lenient => Tok::Num(0.0),
                    Err(_) => return Err(ScriptError::syntax(line_no, col, format!("`{text}` is not a number"))),
                }
            } else if c.is_alphabetic() || c == '_' {
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '$') {
                    i += 1;
                }
                let w: String = chars[start..i].iter().collect();
                // `NOTE` as the first word of a statement comments out the rest of the line, so
                // an apostrophe in its text is never read as the start of a name.
                if w.eq_ignore_ascii_case("NOTE") && matches!(out.last().map(|t: &Token| &t.tok), None | Some(Tok::Eol))
                {
                    out.push(Token { tok: Tok::Word(w), line: line_no, col, end: i + 1, space_before: space });
                    break;
                }
                Tok::Word(w)
            } else {
                let rest: String = chars[i..chars.len().min(i + 2)].iter().collect();
                match SYMS.iter().find(|s| rest.starts_with(**s)) {
                    Some(s) => {
                        i += s.chars().count();
                        if *s == "&" && chars[i..].iter().all(|ch| ch.is_whitespace()) {
                            cont_here = true;
                            break;
                        }
                        Tok::Sym(s)
                    }
                    None if lenient => {
                        i += 1;
                        Tok::Sym("?")
                    }
                    None => return Err(ScriptError::syntax(line_no, col, format!("unexpected character `{c}`"))),
                }
            };
            out.push(Token { tok, line: line_no, col, end: i + 1, space_before: space });
            space = false;
        }
        if cont_here {
            continue;
        }
        out.push(Token {
            tok: Tok::Eol,
            line: line_no,
            col: chars.len() + 1,
            end: chars.len() + 1,
            space_before: true,
        });
    }
    out.push(Token { tok: Tok::Eof, line: line_no.max(1), col: 1, end: 1, space_before: true });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Tok> {
        lex(src, false).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn words_numbers_strings_and_names() {
        let t = kinds("LET K1 = 2.5e1 + C3 # note\nPRINT \"a \"\"b\"\" c\" 'Weight (g)'");
        assert_eq!(
            t,
            vec![
                Tok::Word("LET".into()),
                Tok::Word("K1".into()),
                Tok::Sym("="),
                Tok::Num(25.0),
                Tok::Sym("+"),
                Tok::Word("C3".into()),
                Tok::Eol,
                Tok::Word("PRINT".into()),
                Tok::Str("a \"b\" c".into()),
                Tok::Quoted("Weight (g)".into()),
                Tok::Eol,
                Tok::Eof
            ]
        );
    }

    #[test]
    fn continuation_joins_lines_and_keeps_positions() {
        let toks = lex("LET K1 = 1 + &\n   2\nLET K2 = 3", false).unwrap();
        let eols = toks.iter().filter(|t| t.tok == Tok::Eol).count();
        assert_eq!(eols, 2);
        let two = toks.iter().find(|t| t.tok == Tok::Num(2.0)).unwrap();
        assert_eq!((two.line, two.col), (2, 4));
        // `&` inside a line is the AND operator.
        assert!(kinds("IF K1 & K2").contains(&Tok::Sym("&")));
    }

    #[test]
    fn symbols_and_two_character_operators() {
        let t = kinds("a ** b <= c <> d != e == f");
        let syms: Vec<&str> = t.iter().filter_map(|x| if let Tok::Sym(s) = x { Some(*s) } else { None }).collect();
        assert_eq!(syms, ["**", "<=", "<>", "!=", "=="]);
    }

    #[test]
    fn spacing_is_recorded() {
        let t = lex("C1-C5 C6 -5", false).unwrap();
        let sp: Vec<bool> = t.iter().take(6).map(|x| x.space_before).collect();
        assert_eq!(sp, [true, false, false, true, true, false]);
    }

    #[test]
    fn numbers_edge_cases() {
        // A trailing dot is the command terminator, not part of the number.
        assert_eq!(kinds("5."), vec![Tok::Num(5.0), Tok::Sym("."), Tok::Eol, Tok::Eof]);
        assert_eq!(kinds(".5")[0], Tok::Num(0.5));
        assert_eq!(kinds("2e")[0], Tok::Num(2.0));
        assert_eq!(kinds("2e")[1], Tok::Word("e".into()));
    }

    #[test]
    fn errors_are_positioned_and_lenient_mode_recovers() {
        let e = lex("LET K1 = 1\nPRINT \"oops", false).unwrap_err();
        assert_eq!((e.line, e.col), (2, 7));
        assert!(lex("PRINT \"oops", true).is_ok());
        let e = lex("LET K1 = 1 @ 2", false).unwrap_err();
        assert_eq!((e.line, e.col), (1, 12));
        assert!(lex("LET K1 = 1 @ 2", true).is_ok());
    }

    #[test]
    fn crlf_and_unicode_names() {
        let t = kinds("PRINT 'Größe'\r\nPRINT C1\r\n");
        assert_eq!(t[1], Tok::Quoted("Größe".into()));
        assert_eq!(t.iter().filter(|x| **x == Tok::Eol).count(), 3);
    }
}
