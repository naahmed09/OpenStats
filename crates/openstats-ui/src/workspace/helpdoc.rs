//! In-app help (Phase 6a): the help of one dialog, and a search across every command.

use crate::dialogs::analysis::Analysis;
use crate::dialogs::spec::{FieldKind, Page};
use crate::dialogs::Registry;
use openstats_stats::report::{Cell, Column, ColumnKind, Section, Table};

fn add(t: &mut Table, row: Vec<Cell>) {
    let _ = t.push_row(row);
}

/// `Stat > Basic Statistics > 1-Sample t` for an analysis.
pub fn menu_path(a: &Analysis) -> String {
    a.menu.iter().map(|s| s.trim_end_matches("...")).collect::<Vec<_>>().join(" > ")
}

fn describe(kind: &FieldKind) -> String {
    match kind {
        FieldKind::Variables { kind, min, max } => {
            let n = if *min == 0 {
                format!("optional, up to {max}")
            } else if min == max {
                format!("{min}")
            } else {
                format!("{min} to {max}")
            };
            format!("column(s): {} ({n})", format!("{kind:?}").to_lowercase())
        }
        FieldKind::Number { default, min, max, integer, optional } => {
            let range = match (min.is_finite() && *min > -1e100, max.is_finite() && *max < 1e100) {
                (true, true) => format!(", from {min} to {max}"),
                (true, false) => format!(", at least {min}"),
                (false, true) => format!(", at most {max}"),
                (false, false) => String::new(),
            };
            format!(
                "{}{} (default {default}{range})",
                if *integer { "whole number" } else { "number" },
                if *optional { ", may be blank" } else { "" }
            )
        }
        FieldKind::Text { default } if default.is_empty() => "text".into(),
        FieldKind::Text { default } => format!("text (default \"{default}\")"),
        FieldKind::Choice { options, default } => {
            format!("one of: {} (default {})", options.join(" | "), options.get(*default).map_or("", String::as_str))
        }
        FieldKind::Check { default } => format!("check box (default {})", if *default { "on" } else { "off" }),
    }
}

/// Everything the dialog `a` asks for, as a Session section.
pub fn dialog_help(a: &Analysis) -> Section {
    let mut s = Section::new(format!("Help: {}", a.spec.title));
    s.text(a.spec.help.clone());
    if !a.menu.is_empty() {
        s.text(format!("Menu: {}", menu_path(a)));
    }
    let mut t = Table::new(
        "Fields",
        vec![
            Column::new("Field", ColumnKind::Text),
            Column::new("Page", ColumnKind::Text),
            Column::new("Value", ColumnKind::Text),
        ],
    );
    for f in &a.spec.fields {
        let page = match f.page {
            Page::Main => "Main",
            Page::Options => "Options",
            Page::Graphs => "Graphs",
            Page::Storage => "Storage",
        };
        add(&mut t, vec![Cell::text(f.label.clone()), Cell::text(page), Cell::text(describe(&f.kind))]);
    }
    if !a.spec.fields.is_empty() {
        s.table(t);
    }
    s
}

/// Commands matching every word of `query` (in the title, menu path, help text, id or a field label); an empty
/// query lists the number of commands in each menu.
pub fn search(reg: &Registry, query: &str) -> Section {
    let words: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
    if words.is_empty() {
        let mut s = Section::new("Help: commands by menu");
        let mut t = Table::new(
            "Menus",
            vec![Column::new("Menu", ColumnKind::Text), Column::new("Commands", ColumnKind::Integer)],
        );
        let mut seen: Vec<(String, i64)> = Vec::new();
        for a in reg.all() {
            let top = a.menu.first().copied().unwrap_or("(no menu)").to_string();
            match seen.iter_mut().find(|(n, _)| *n == top) {
                Some((_, c)) => *c += 1,
                None => seen.push((top, 1)),
            }
        }
        for (n, c) in seen {
            add(&mut t, vec![Cell::text(n), Cell::Int(c)]);
        }
        s.table(t);
        s.text("Type words in Help > Search Commands to find a command, or press F1 in a dialog for its fields.");
        return s;
    }
    let mut s = Section::new(format!("Help: commands matching \"{}\"", query.trim()));
    let hits: Vec<&Analysis> = reg
        .all()
        .iter()
        .filter(|a| {
            let hay = format!(
                "{} {} {} {} {}",
                a.spec.title,
                menu_path(a),
                a.spec.help,
                a.spec.id,
                a.spec.fields.iter().map(|f| f.label.as_str()).collect::<Vec<_>>().join(" ")
            )
            .to_lowercase();
            words.iter().all(|w| hay.contains(w))
        })
        .collect();
    if hits.is_empty() {
        s.text("No command matches. Try fewer or different words.");
        return s;
    }
    let mut t = Table::new(
        "Commands",
        vec![
            Column::new("Command", ColumnKind::Text),
            Column::new("Menu", ColumnKind::Text),
            Column::new("What it does", ColumnKind::Text),
        ],
    );
    for a in hits.iter().take(40) {
        add(&mut t, vec![Cell::text(a.spec.title.clone()), Cell::text(menu_path(a)), Cell::text(a.spec.help.clone())]);
    }
    s.table(t);
    if hits.len() > 40 {
        s.text(format!("{} more match; add words to narrow the search.", hits.len() - 40));
    }
    s
}
