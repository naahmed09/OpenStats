//! Phase 6a: help search, F1, custom shortcuts, related documents and options.

use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_ui::dialogs::{Command, Registry, Value};
use openstats_ui::grid::{CellRef, Key, Mods};
use openstats_ui::workspace::helpdoc;
use openstats_ui::workspace::keymap::{chord_of, normalize, KeyMap};
use openstats_ui::workspace::{NavAction, Outcome, Target, Workspace};

fn loaded() -> Workspace {
    let mut w = Workspace::new(1200.0, 800.0);
    let mut wb = Workbook::new();
    let x: Vec<f64> = (0..10).map(f64::from).collect();
    wb.worksheets[0].add_column(Column::numeric("x", x)).unwrap();
    *w.grid_mut() = openstats_ui::grid::GridController::new(wb, 800.0, 300.0);
    w.set_size(1200.0, 800.0);
    w
}

fn set(w: &mut Workspace, field: &str, val: Value) {
    let d = w.dialog_mut().expect("dialog open");
    let mut v = d.values().clone();
    v.set(field, val);
    d.set_values(v);
}

fn session_text(w: &Workspace) -> String {
    w.sessions().iter().map(|s| s.report.render_text()).collect()
}

#[test]
fn help_search_finds_commands_by_any_word() {
    let reg = Registry::standard();
    let s = helpdoc::search(&reg, "regression logistic");
    let mut r = openstats_stats::report::SessionReport::new("t");
    r.push(s);
    let t = r.render_text();
    assert!(t.contains("Logistic") && t.contains("Stat >"), "{t}");
    let none = helpdoc::search(&reg, "zzzzqqq");
    assert!(none.blocks.len() == 1);
    let menus = helpdoc::search(&reg, "");
    let mut r = openstats_stats::report::SessionReport::new("t");
    r.push(menus);
    let t = r.render_text();
    assert!(t.contains("Stat") && t.contains("Graph") && t.contains("Help"), "{t}");
    // Every dialog has help text and every field a label.
    for a in reg.all() {
        assert!(!a.spec.help.trim().is_empty(), "{} has no help text", a.spec.id);
        assert!(a.spec.fields.iter().all(|f| !f.label.trim().is_empty()), "{}", a.spec.id);
    }
}

#[test]
fn help_search_dialog_and_f1() {
    let mut w = loaded();
    // F1 with no dialog opens the search; running it writes the result to the Session window.
    w.key_down(Key::F1, Mods::NONE);
    assert!(w.dialog().is_some());
    set(&mut w, "query", Value::Text("histogram".into()));
    let o = w.run_dialog();
    assert_eq!(o, Outcome::None, "{:?} {}", w.dialog().map(|d| d.problems()), w.status());
    assert!(session_text(&w).contains("Histogram"), "{}", session_text(&w));
    // F1 inside a dialog adds that dialog's fields to the Session window.
    w.execute(Command::OpenDialog("stat.basic.ttest1".into()));
    w.key_down(Key::F1, Mods::NONE);
    assert!(w.dialog().is_some(), "the dialog stays open");
    let t = session_text(&w);
    assert!(t.contains("Help: 1-Sample t") && t.contains("Test mean") && t.contains("Confidence level"), "{t}");
}

#[test]
fn shortcut_names_are_checked() {
    assert_eq!(normalize("Shift+Ctrl+K").unwrap(), "ctrl+shift+k");
    assert_eq!(normalize("f2").unwrap(), "f2");
    assert!(normalize("k").unwrap_err().contains("Ctrl or Alt"));
    assert!(normalize("Ctrl+C").unwrap_err().contains("text-editing"));
    assert!(normalize("Ctrl+K+L").is_err());
    assert!(normalize("Ctrl+Home").is_err());
    assert!(normalize("").is_err());
    assert_eq!(chord_of(Key::Char('K'), Mods { shift: true, ctrl: true, alt: false }).unwrap(), "ctrl+shift+k");
    let reg = Registry::standard();
    let mut k = KeyMap::default();
    assert!(k.set(&reg, "ctrl+k", "no such thing").is_err());
    assert!(k.set(&reg, "ctrl+k", "Undo").is_ok());
    assert!(k.set(&reg, "ctrl+k", "").is_ok(), "a blank command removes");
    assert!(k.set(&reg, "ctrl+k", "").is_err(), "removing twice is reported");
    assert!(k.is_empty());
}

#[test]
fn custom_shortcuts_run_commands_and_dialogs() {
    let mut w = loaded();
    w.execute(Command::OpenDialog("tools.keys".into()));
    set(&mut w, "chord", Value::Text("Ctrl+Alt+U".into()));
    set(&mut w, "command", Value::Text("Undo".into()));
    assert_eq!(w.run_dialog(), Outcome::None);
    w.execute(Command::OpenDialog("tools.keys".into()));
    set(&mut w, "chord", Value::Text("F9".into()));
    set(&mut w, "command", Value::Text("stat.basic.ttest1".into()));
    assert_eq!(w.run_dialog(), Outcome::None);
    assert_eq!(w.keymap().entries().count(), 2);
    // Ctrl+Alt+U undoes an edit.
    w.grid_mut().enter_cell(CellRef { row: 0, col: 0 }, "99").unwrap();
    assert_eq!(w.grid().worksheet().columns[0].data.to_f64_vec()[0], 99.0);
    w.key_down(Key::Char('u'), Mods { shift: false, ctrl: true, alt: true });
    assert_eq!(w.grid().worksheet().columns[0].data.to_f64_vec()[0], 0.0);
    // The custom F9 opens the dialog instead of toggling a macro breakpoint.
    w.key_down(Key::F9, Mods::NONE);
    assert_eq!(w.dialog().map(|d| d.spec().id.as_str()), Some("stat.basic.ttest1"));
    w.cancel_dialog();
    // Reserved and unknown bindings are refused and the dialog says why.
    w.execute(Command::OpenDialog("tools.keys".into()));
    set(&mut w, "chord", Value::Text("Ctrl+C".into()));
    set(&mut w, "command", Value::Text("Undo".into()));
    assert!(matches!(w.run_dialog(), Outcome::Error(_)));
    w.cancel_dialog();
    // Saved with the project.
    let bytes = w.save_project();
    let mut w2 = Workspace::new(1200.0, 800.0);
    w2.open_project(&bytes);
    assert_eq!(w2.keymap(), w.keymap());
    // The shortcut list is in Help > Keyboard Shortcuts.
    w2.execute(Command::ShowShortcuts);
    assert!(session_text(&w2).contains("ctrl+alt+u  undo"));
    // Reset.
    w2.execute(Command::OpenDialog("tools.keys".into()));
    set(&mut w2, "reset", Value::Bool(true));
    w2.run_dialog();
    assert!(w2.keymap().is_empty());
}

#[test]
fn related_documents_are_listed_in_the_navigator_and_saved() {
    let mut w = loaded();
    assert!(!w.navigator().visible(&Default::default()).iter().any(|r| r.1.contains("Related")));
    for (p, n) in [("report.docx", "final report"), ("raw/data.csv", "")] {
        w.execute(Command::OpenDialog("tools.related".into()));
        set(&mut w, "path", Value::Text(p.into()));
        set(&mut w, "note", Value::Text(n.into()));
        assert_eq!(w.run_dialog(), Outcome::None);
    }
    assert_eq!(w.related().len(), 2);
    w.execute(Command::OpenDialog("tools.related".into()));
    set(&mut w, "path", Value::Text("report.docx".into()));
    assert!(matches!(w.run_dialog(), Outcome::Error(_)), "duplicates are refused");
    w.cancel_dialog();
    let rows = w.render();
    assert!(rows
        .items
        .iter()
        .any(|i| matches!(&i.shape, openstats_charts::core::Shape::Text { text, .. } if text == "Related Documents")));
    w.open_item(Target::Related(0));
    assert!(session_text(&w).contains("final report"));
    // Save, reopen, delete one through the navigator, remove the other by name.
    let bytes = w.save_project();
    let mut w2 = Workspace::new(1200.0, 800.0);
    w2.open_project(&bytes);
    assert_eq!(w2.related(), w.related());
    w2.navigator_action(Target::Related(0), NavAction::Delete);
    assert_eq!(w2.related().len(), 1);
    w2.execute(Command::OpenDialog("tools.related".into()));
    set(&mut w2, "action", Value::Choice(1));
    set(&mut w2, "path", Value::Text("raw/data.csv".into()));
    assert_eq!(w2.run_dialog(), Outcome::None);
    assert!(w2.related().is_empty());
}

#[test]
fn undo_depth_is_an_option_and_is_saved() {
    let mut w = loaded();
    w.execute(Command::OpenDialog("tools.options".into()));
    set(&mut w, "undo", Value::Num(3.0));
    assert_eq!(w.run_dialog(), Outcome::None);
    for i in 0..6 {
        w.grid_mut().enter_cell(CellRef { row: i, col: 0 }, "50").unwrap();
    }
    let mut steps = 0;
    while w.grid().history().can_undo() {
        w.execute(Command::Undo);
        steps += 1;
        assert!(steps < 20);
    }
    assert_eq!(steps, 3);
    let bytes = w.save_project();
    let mut w2 = Workspace::new(1200.0, 800.0);
    w2.open_project(&bytes);
    for i in 0..6 {
        w2.grid_mut().enter_cell(CellRef { row: i, col: 0 }, "7").unwrap();
    }
    assert_eq!(w2.grid().history().undo_len(), 3, "the limit comes back with the project");
    w2.execute(Command::OpenDialog("tools.options".into()));
    set(&mut w2, "undo", Value::Num(0.0));
    w2.run_dialog();
    assert!(w2.dialog().is_some_and(|d| !d.problems().is_empty()), "0 steps is refused by the dialog");
}
