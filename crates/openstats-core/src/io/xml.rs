//! Minimal conforming XML 1.0 pull parser.
//!
//! Produces start/end/text events, decodes the predefined entities and
//! numeric character references, handles CDATA sections, skips comments,
//! processing instructions and DOCTYPE declarations (including an internal
//! subset), checks that tags are properly nested, and rejects duplicate
//! attributes. Namespaces are not resolved: names keep their prefix, and
//! [`local_name`] strips it.

use crate::error::{CoreError, Result};

/// A parse event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Start { name: String, attrs: Vec<(String, String)>, empty: bool },
    End { name: String },
    Text(String),
}

impl Event {
    /// Attribute value by (local) name on a start event.
    pub fn attr(&self, key: &str) -> Option<&str> {
        match self {
            Event::Start { attrs, .. } => attrs
                .iter()
                .find(|(k, _)| k == key)
                .or_else(|| attrs.iter().find(|(k, _)| local_name(k) == key))
                .map(|(_, v)| v.as_str()),
            _ => None,
        }
    }
}

/// Name without its namespace prefix (`x:row` → `row`).
pub fn local_name(name: &str) -> &str {
    name.rsplit_once(':').map_or(name, |(_, l)| l)
}

/// Decode XML bytes to a string: UTF-8 (with or without BOM) or UTF-16 with BOM.
pub fn decode(bytes: &[u8]) -> Result<String> {
    let bad = |e: String| CoreError::Format(format!("xml: {e}"));
    if let Some(b) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8(b.to_vec()).map_err(|e| bad(e.to_string()));
    }
    let utf16 = |b: &[u8], le: bool| -> Result<String> {
        if b.len() % 2 != 0 {
            return Err(bad("odd-length UTF-16 data".into()));
        }
        let units: Vec<u16> = b
            .chunks(2)
            .map(|c| if le { u16::from_le_bytes([c[0], c[1]]) } else { u16::from_be_bytes([c[0], c[1]]) })
            .collect();
        String::from_utf16(&units).map_err(|e| bad(e.to_string()))
    };
    if let Some(b) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return utf16(b, true);
    }
    if let Some(b) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return utf16(b, false);
    }
    String::from_utf8(bytes.to_vec()).map_err(|e| bad(e.to_string()))
}

/// Replace entity and character references.
pub fn unescape(s: &str, pos: usize) -> Result<String> {
    if !s.contains('&') {
        return Ok(s.to_string());
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i + 1..];
        let end = tail.find(';').ok_or_else(|| CoreError::parse("XML", "unterminated entity reference", Some(pos)))?;
        let ent = &tail[..end];
        match ent {
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "amp" => out.push('&'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            _ => {
                let code = if let Some(h) = ent.strip_prefix("#x").or_else(|| ent.strip_prefix("#X")) {
                    u32::from_str_radix(h, 16).ok()
                } else if let Some(d) = ent.strip_prefix('#') {
                    d.parse::<u32>().ok()
                } else {
                    return Err(CoreError::parse("XML", format!("unknown entity &{ent};"), Some(pos)));
                };
                let c = code.and_then(char::from_u32).ok_or_else(|| {
                    CoreError::parse("XML", format!("invalid character reference &{ent};"), Some(pos))
                })?;
                out.push(c);
            }
        }
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Streaming reader over a string.
#[derive(Debug)]
pub struct Reader<'a> {
    s: &'a str,
    p: usize,
    stack: Vec<String>,
    pending_end: Option<String>,
    seen_root: bool,
    /// Skip text events that are entirely whitespace (default true).
    pub skip_whitespace: bool,
}

impl<'a> Reader<'a> {
    pub fn new(s: &'a str) -> Self {
        Reader { s, p: 0, stack: Vec::new(), pending_end: None, seen_root: false, skip_whitespace: true }
    }

    fn err<T>(&self, m: impl Into<String>) -> Result<T> {
        Err(CoreError::parse("XML", m, Some(self.p)))
    }

    fn skip_past(&mut self, pat: &str, what: &str) -> Result<()> {
        match self.s[self.p..].find(pat) {
            Some(i) => {
                self.p += i + pat.len();
                Ok(())
            }
            None => self.err(format!("unterminated {what}")),
        }
    }

    /// Current element nesting depth.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Next event, or `None` at end of document.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<Event>> {
        if let Some(name) = self.pending_end.take() {
            return Ok(Some(Event::End { name }));
        }
        loop {
            if self.p >= self.s.len() {
                if let Some(open) = self.stack.last() {
                    return self.err(format!("unclosed element <{open}>"));
                }
                return Ok(None);
            }
            let rest = &self.s[self.p..];
            if !rest.starts_with('<') {
                let end = rest.find('<').unwrap_or(rest.len());
                let raw = &rest[..end];
                let start = self.p;
                self.p += end;
                if self.stack.is_empty() {
                    if raw.trim().is_empty() {
                        continue;
                    }
                    return Err(CoreError::parse("XML", "text outside the root element", Some(start)));
                }
                if self.skip_whitespace && raw.trim().is_empty() {
                    continue;
                }
                return Ok(Some(Event::Text(unescape(raw, start)?)));
            }
            if rest.starts_with("<!--") {
                self.skip_past("-->", "comment")?;
                continue;
            }
            if rest.starts_with("<?") {
                self.skip_past("?>", "processing instruction")?;
                continue;
            }
            if let Some(cdata) = rest.strip_prefix("<![CDATA[") {
                let end =
                    cdata.find("]]>").ok_or_else(|| CoreError::parse("XML", "unterminated CDATA", Some(self.p)))?;
                let text = cdata[..end].to_string();
                self.p += 9 + end + 3;
                if self.stack.is_empty() {
                    return self.err("CDATA outside the root element");
                }
                return Ok(Some(Event::Text(text)));
            }
            if rest.starts_with("<!DOCTYPE") || rest.starts_with("<!doctype") {
                // Skip, honouring an internal subset in [ ].
                let b = rest.as_bytes();
                let mut depth = 0i32;
                let mut i = 0;
                let mut quote: Option<u8> = None;
                while i < b.len() {
                    match (quote, b[i]) {
                        (Some(q), c) if c == q => quote = None,
                        (Some(_), _) => {}
                        (None, b'"' | b'\'') => quote = Some(b[i]),
                        (None, b'[') => depth += 1,
                        (None, b']') => depth -= 1,
                        (None, b'>') if depth == 0 => break,
                        _ => {}
                    }
                    i += 1;
                }
                if i >= b.len() {
                    return self.err("unterminated DOCTYPE");
                }
                self.p += i + 1;
                continue;
            }
            if let Some(after) = rest.strip_prefix("</") {
                let end =
                    after.find('>').ok_or_else(|| CoreError::parse("XML", "unterminated end tag", Some(self.p)))?;
                let name = after[..end].trim().to_string();
                match self.stack.pop() {
                    Some(open) if open == name => {}
                    Some(open) => return self.err(format!("</{name}> does not match <{open}>")),
                    None => return self.err(format!("unexpected </{name}>")),
                }
                self.p += 2 + end + 1;
                return Ok(Some(Event::End { name }));
            }
            return self.start_tag().map(Some);
        }
    }

    fn start_tag(&mut self) -> Result<Event> {
        let b = self.s.as_bytes();
        let mut i = self.p + 1;
        let name_start = i;
        while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'/' && b[i] != b'>' {
            i += 1;
        }
        let name = self.s[name_start..i].to_string();
        if name.is_empty() {
            return self.err("empty element name");
        }
        let mut attrs: Vec<(String, String)> = Vec::new();
        loop {
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= b.len() {
                return self.err(format!("unterminated start tag <{name}>"));
            }
            if b[i] == b'>' || (b[i] == b'/' && b.get(i + 1) == Some(&b'>')) {
                let empty = b[i] == b'/';
                self.p = i + if empty { 2 } else { 1 };
                if self.stack.is_empty() {
                    if self.seen_root {
                        return self.err("more than one root element");
                    }
                    self.seen_root = true;
                }
                if empty {
                    self.pending_end = Some(name.clone());
                } else {
                    self.stack.push(name.clone());
                }
                return Ok(Event::Start { name, attrs, empty });
            }
            let ks = i;
            while i < b.len() && b[i] != b'=' && !b[i].is_ascii_whitespace() && b[i] != b'>' && b[i] != b'/' {
                i += 1;
            }
            let key = self.s[ks..i].to_string();
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            if key.is_empty() || b.get(i) != Some(&b'=') {
                self.p = i;
                return self.err(format!("malformed attribute in <{name}>"));
            }
            i += 1;
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            let q = match b.get(i) {
                Some(&c @ (b'"' | b'\'')) => c,
                _ => {
                    self.p = i;
                    return self.err("attribute value must be quoted");
                }
            };
            let vs = i + 1;
            let ve = self.s[vs..]
                .find(q as char)
                .map(|k| vs + k)
                .ok_or_else(|| CoreError::parse("XML", "unterminated attribute value", Some(vs)))?;
            let raw = &self.s[vs..ve];
            if raw.contains('<') {
                self.p = vs;
                return self.err("'<' in attribute value");
            }
            // Attribute-value normalisation of literal whitespace characters.
            let v = unescape(&raw.replace(['\t', '\n', '\r'], " "), vs)?;
            if attrs.iter().any(|(k, _)| *k == key) {
                self.p = ks;
                return self.err(format!("duplicate attribute {key:?}"));
            }
            attrs.push((key, v));
            i = ve + 1;
        }
    }

    /// Concatenated text content until the end of the current element,
    /// skipping nested markup (call right after a non-empty start event).
    pub fn read_text(&mut self) -> Result<String> {
        let depth = self.depth();
        let mut out = String::new();
        let saved = self.skip_whitespace;
        self.skip_whitespace = false;
        let r = loop {
            match self.next()? {
                Some(Event::Text(t)) => out.push_str(&t),
                Some(Event::End { .. }) if self.depth() < depth => break Ok(out),
                Some(_) => {}
                None => break self.err("unexpected end of document"),
            }
        };
        self.skip_whitespace = saved;
        r
    }

    /// Skip the remainder of the current element (after its start event).
    pub fn skip_element(&mut self) -> Result<()> {
        let depth = self.depth();
        loop {
            match self.next()? {
                Some(Event::End { .. }) if self.depth() < depth => return Ok(()),
                Some(_) => {}
                None => return self.err("unexpected end of document"),
            }
        }
    }
}

/// Collect all events (convenience for small documents and tests).
pub fn events(s: &str) -> Result<Vec<Event>> {
    let mut r = Reader::new(s);
    let mut out = Vec::new();
    while let Some(e) = r.next()? {
        out.push(e);
    }
    Ok(out)
}

/// Escape text for inclusion in XML content or attributes.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_and_entities() {
        let doc = r#"<?xml version="1.0"?>
<!DOCTYPE r [ <!ENTITY x "y"> ]>
<!-- c -->
<r a="1 &amp; 2" b='&#x41;&#66;'>
  <x:c/>text &lt;here&gt;<![CDATA[<raw & stuff>]]><d e = "q"></d>
</r>"#;
        let ev = events(doc).unwrap();
        assert_eq!(
            ev[0],
            Event::Start {
                name: "r".into(),
                attrs: vec![("a".into(), "1 & 2".into()), ("b".into(), "AB".into())],
                empty: false
            }
        );
        assert_eq!(ev[1], Event::Start { name: "x:c".into(), attrs: vec![], empty: true });
        assert_eq!(ev[2], Event::End { name: "x:c".into() });
        assert_eq!(ev[3], Event::Text("text <here>".into()));
        assert_eq!(ev[4], Event::Text("<raw & stuff>".into()));
        assert_eq!(ev[5].attr("e"), Some("q"));
        assert_eq!(ev.last(), Some(&Event::End { name: "r".into() }));
        assert_eq!(local_name("x:c"), "c");
    }

    #[test]
    fn read_text_and_skip() {
        let mut r =
            Reader::new("<si><r><t>Hello</t></r><r><t xml:space=\"preserve\"> World</t></r><rPh><t>x</t></rPh></si>");
        r.next().unwrap();
        assert_eq!(r.read_text().unwrap(), "Hello Worldx");
        let mut r = Reader::new("<a><skip><deep/>t</skip><keep>k</keep></a>");
        r.next().unwrap();
        r.next().unwrap();
        r.skip_element().unwrap();
        assert_eq!(r.next().unwrap().unwrap(), Event::Start { name: "keep".into(), attrs: vec![], empty: false });
    }

    #[test]
    fn malformed_documents() {
        for bad in [
            "<a>",
            "<a></b>",
            "</a>",
            "<a b=1/>",
            "<a b='1' b='2'/>",
            "<a>&bogus;</a>",
            "<a>&#xZZ;</a>",
            "<a/><b/>",
            "text<a/>",
            "<a><!-- x</a>",
            "<a b='<'/>",
            "<a><![CDATA[x</a>",
            "<>",
            "<a",
        ] {
            assert!(events(bad).is_err(), "should reject {bad:?}");
        }
    }

    #[test]
    fn encodings() {
        assert_eq!(decode(b"\xEF\xBB\xBF<a/>").unwrap(), "<a/>");
        let utf16: Vec<u8> =
            [0xFF, 0xFE].into_iter().chain("<é/>".encode_utf16().flat_map(|u| u.to_le_bytes())).collect();
        assert_eq!(decode(&utf16).unwrap(), "<é/>");
        assert_eq!(escape("<a & 'b'>"), "&lt;a &amp; &apos;b&apos;&gt;");
    }
}
