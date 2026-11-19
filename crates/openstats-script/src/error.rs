//! Script errors with source positions and a macro call trace.

use std::fmt;

/// What went wrong, broadly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// The text is not valid script.
    Syntax,
    /// A statement failed while running.
    Runtime,
    /// A resource limit (steps, recursion depth, output size) was reached.
    Limit,
}

/// An error with its location. `line` and `col` are 1-based; `col` is 0 when unknown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptError {
    /// Category.
    pub kind: ErrorKind,
    /// Source line.
    pub line: usize,
    /// Source column (0 = whole line).
    pub col: usize,
    /// Description.
    pub message: String,
    /// Macro calls leading to the error, innermost first: `(macro name, line where it was called)`.
    pub trace: Vec<(String, usize)>,
    /// The text of the offending source line, when known (shown under the message).
    pub snippet: Option<String>,
}

impl ScriptError {
    /// A syntax error.
    pub fn syntax(line: usize, col: usize, message: impl Into<String>) -> Self {
        ScriptError { kind: ErrorKind::Syntax, line, col, message: message.into(), trace: Vec::new(), snippet: None }
    }

    /// A runtime error.
    pub fn runtime(line: usize, message: impl Into<String>) -> Self {
        ScriptError {
            kind: ErrorKind::Runtime,
            line,
            col: 0,
            message: message.into(),
            trace: Vec::new(),
            snippet: None,
        }
    }

    /// A limit error.
    pub fn limit(line: usize, message: impl Into<String>) -> Self {
        ScriptError { kind: ErrorKind::Limit, line, col: 0, message: message.into(), trace: Vec::new(), snippet: None }
    }
}

impl ScriptError {
    /// Attach the text of source line `self.line` taken from `lines` (0-based list of lines),
    /// unless one is attached already or the line is out of range.
    pub fn with_snippet(mut self, lines: &[String]) -> Self {
        if self.snippet.is_none() && self.line >= 1 {
            if let Some(t) = lines.get(self.line - 1) {
                let t = t.trim();
                if !t.is_empty() {
                    self.snippet = Some(t.to_string());
                }
            }
        }
        self
    }
}

impl fmt::Display for ScriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = match self.kind {
            ErrorKind::Syntax => "syntax error",
            ErrorKind::Runtime => "error",
            ErrorKind::Limit => "limit reached",
        };
        if self.col > 0 {
            write!(f, "line {}, column {}: {what}: {}", self.line, self.col, self.message)?;
        } else {
            write!(f, "line {}: {what}: {}", self.line, self.message)?;
        }
        if let Some(t) = &self.snippet {
            write!(f, "\n    {} | {t}", self.line)?;
        }
        for (name, line) in &self.trace {
            write!(f, "\n  in macro {name}, called at line {line}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ScriptError {}
