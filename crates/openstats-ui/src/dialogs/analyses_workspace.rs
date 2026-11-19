//! Dialogs applied by the workspace itself because they act on the project rather than on worksheet columns
//! (Phase 6a): help search, custom shortcuts, related documents and options.

use super::analyses::add;
use super::analysis::{col_title, Output, Registry, Res, Runner};
use super::spec::{DialogSpec, Page, Values, VarKind};
use crate::grid::GridController;
use openstats_core::locale::Locale;
use openstats_core::{ColumnKind, Worksheet};

/// Ids of the dialogs the workspace runs itself.
pub const OWNED: [&str; 4] = ["help.search", "tools.keys", "tools.related", "tools.options"];

fn owned(_: &Worksheet, _: &Values) -> Res<Output> {
    Err("this command is applied by the workspace".into())
}

pub(crate) fn register(r: &mut Registry) {
    add(
        r,
        DialogSpec::new(
            "help.search",
            "Search Commands",
            "Find commands by words in their name, menu, description or field labels. Leave blank to list the menus.",
        )
        .text(Page::Main, "query", "Words to look for", ""),
        &["Help", "Search Commands..."],
        Runner::Report(owned),
    );
    add(
        r,
        DialogSpec::new(
            "tools.keys",
            "Customize Shortcuts",
            "Bind a key combination to a command or dialog. Shortcuts are saved with the project. A blank command removes the shortcut.",
        )
        .text(Page::Main, "chord", "Shortcut, e.g. Ctrl+Shift+K or F2", "")
        .text(Page::Main, "command", "Command name (e.g. Undo) or dialog id (e.g. stat.basic.ttest1)", "")
        .check(Page::Options, "list", "Only list the custom shortcuts", false)
        .check(Page::Options, "reset", "Remove all custom shortcuts", false),
        &["Tools", "Customize Shortcuts..."],
        Runner::Report(owned),
    );
    add(
        r,
        DialogSpec::new(
            "tools.related",
            "Related Documents",
            "Keep a list of files related to this project (reports, source data, notes) with the project.",
        )
        .choice(Page::Main, "action", "Action", &["Add", "Remove", "List"], 0)
        .text(Page::Main, "path", "File name or address", "")
        .text(Page::Main, "note", "Note (optional)", ""),
        &["Tools", "Related Documents..."],
        Runner::Report(owned),
    );
    add(
        r,
        DialogSpec::new("tools.options", "Options", "Settings of this project that are not tied to a command.")
            .integer(Page::Main, "undo", "Undo steps to keep", 1000.0, 1.0, 100_000.0)
            .choice(
                Page::Main,
                "locale",
                "Regional settings (decimal mark and date order)",
                &Locale::PRESETS.map(|(n, _)| n),
                0,
            ),
        &["Tools", "Options..."],
        Runner::Report(owned),
    );
    add(
        r,
        DialogSpec::new(
            "data.valuelabels",
            "Value Labels",
            "Show a label in the worksheet in place of a value (for example 1 = Low, 2 = High). The data keep their values.",
        )
        .vars("col", "Column", VarKind::Any, 1, 1)
        .text(Page::Main, "labels", "Labels as value=label, separated by commas (e.g. 1=Low, 2=High)", "")
        .check(Page::Main, "clear", "Remove the labels of the column instead", false),
        &["Data", "Value Labels..."],
        Runner::Mutate(value_labels),
    );
}

/// The `value=label` pairs of `text`; numeric columns match values by their number, not their spelling.
fn parse_labels(text: &str, numeric: bool) -> Res<Vec<(String, String)>> {
    let mut out: Vec<(String, String)> = Vec::new();
    for part in text.split([',', ';', '\n']).map(str::trim).filter(|p| !p.is_empty()) {
        let (k, l) = part.split_once('=').ok_or_else(|| format!("\"{part}\" needs the form value=label"))?;
        let (k, l) = (k.trim(), l.trim());
        if l.is_empty() {
            return Err(format!("the label for {k} is empty").into());
        }
        let key = if numeric {
            let x = openstats_core::value::parse_number(k).map_err(|_| format!("\"{k}\" is not a number"))?;
            openstats_core::value::format_number(x)
        } else {
            k.to_string()
        };
        if out.iter().any(|(o, _)| *o == key) {
            return Err(format!("{k} is labelled twice").into());
        }
        out.push((key, l.to_string()));
    }
    Ok(out)
}

fn value_labels(g: &mut GridController, v: &Values) -> Res<String> {
    let c = v.var("col").ok_or("select a column")?;
    let name = col_title(g.worksheet(), c);
    let kind = g.worksheet().column(c)?.kind();
    if kind == ColumnKind::DateTime {
        return Err("value labels are for numeric and text columns".into());
    }
    if v.flag("clear") {
        g.set_value_labels(c, Vec::new())?;
        return Ok(format!("Removed the value labels of {name}."));
    }
    let labels = parse_labels(v.text("labels"), kind == ColumnKind::Numeric)?;
    if labels.is_empty() {
        return Err("type the labels, for example 1=Low, 2=High".into());
    }
    let n = labels.len();
    g.set_value_labels(c, labels)?;
    Ok(format!("{name} now shows {n} value label(s)."))
}
