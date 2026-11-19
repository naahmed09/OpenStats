//! Clipboard text: tab-separated values as produced and consumed by spreadsheets.

/// Quote a field when it contains a tab, newline or quote (Excel/Minitab convention).
pub fn quote_field(s: &str) -> String {
    if s.contains(['\t', '\n', '\r', '"']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Join rows of fields into clipboard text (`\t` between fields, `\n` after every row).
pub fn to_tsv(rows: &[Vec<String>]) -> String {
    let mut out = String::new();
    for r in rows {
        let line: Vec<String> = r.iter().map(|f| quote_field(f)).collect();
        out.push_str(&line.join("\t"));
        out.push('\n');
    }
    out
}

/// Parse clipboard text into rows of fields. Handles `\n`, `\r\n` and `\r` row ends, quoted
/// fields with embedded tabs, newlines and doubled quotes; one trailing row end is ignored.
pub fn parse_tsv(text: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut field_started = false;
    let mut chars = text.chars().peekable();
    let mut pending_row = false;
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if !field_started => {
                in_quotes = true;
                field_started = true;
                pending_row = true;
            }
            '\t' => {
                row.push(std::mem::take(&mut field));
                field_started = false;
                pending_row = true;
            }
            '\r' | '\n' => {
                if c == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
                field_started = false;
                pending_row = false;
            }
            _ => {
                field.push(c);
                field_started = true;
                pending_row = true;
            }
        }
    }
    if pending_row || !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(v: &[&[&str]]) -> Vec<Vec<String>> {
        v.iter().map(|x| x.iter().map(|s| (*s).to_string()).collect()).collect()
    }

    #[test]
    fn round_trips_awkward_fields() {
        let rows = r(&[&["1", "a\tb", "say \"hi\""], &["multi\nline", "", "*"], &["", "", ""]]);
        let text = to_tsv(&rows);
        assert_eq!(parse_tsv(&text), rows);
    }

    #[test]
    fn parses_line_endings_and_trailing_breaks() {
        assert_eq!(parse_tsv("1\t2\r\n3\t4\r\n"), r(&[&["1", "2"], &["3", "4"]]));
        assert_eq!(parse_tsv("1\t2\n3\t4"), r(&[&["1", "2"], &["3", "4"]]));
        assert_eq!(parse_tsv("1\r2"), r(&[&["1"], &["2"]]));
        assert_eq!(parse_tsv("x"), r(&[&["x"]]));
        assert_eq!(parse_tsv(""), Vec::<Vec<String>>::new());
        assert_eq!(parse_tsv("\n"), r(&[&[""]]));
        assert_eq!(parse_tsv("a\t\tb\n"), r(&[&["a", "", "b"]]));
        assert_eq!(parse_tsv("\t\n"), r(&[&["", ""]]));
        // A quote inside an unquoted field is literal.
        assert_eq!(parse_tsv("5\" pipe\t2"), r(&[&["5\" pipe", "2"]]));
        // Unterminated quote swallows the rest rather than panicking.
        assert_eq!(parse_tsv("\"abc\tdef").len(), 1);
    }

    #[test]
    fn quoting_rules() {
        assert_eq!(quote_field("plain"), "plain");
        assert_eq!(quote_field("a\"b"), "\"a\"\"b\"");
        assert_eq!(quote_field("a\tb"), "\"a\tb\"");
    }
}
