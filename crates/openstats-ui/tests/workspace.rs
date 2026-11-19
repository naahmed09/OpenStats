//! End-to-end: the whole window driven the way a host drives it.

use openstats_charts::core::{Scene, Shape};
use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_ui::dialogs::{Command, Topic, Value};
use openstats_ui::grid::{CellRef, Key, Mods};
use openstats_ui::workspace::{Button, Focus, NavAction, Outcome, Target, Top, Workspace};

fn loaded() -> Workspace {
    let mut w = Workspace::new(1200.0, 800.0);
    let mut wb = Workbook::new();
    let x: Vec<f64> = (0..30).map(f64::from).collect();
    let y: Vec<f64> = (0..30).map(|i| 1.0 + 2.0 * f64::from(i) + f64::from((i * 7) % 5)).collect();
    for c in [Column::numeric("x", x), Column::numeric("y", y)] {
        wb.worksheets[0].add_column(c).unwrap();
    }
    *w.grid_mut() = openstats_ui::grid::GridController::new(wb, 800.0, 300.0);
    w.set_size(1200.0, 800.0);
    w
}

fn texts(s: &Scene) -> Vec<String> {
    s.items
        .iter()
        .filter_map(|i| if let Shape::Text { text, .. } = &i.shape { Some(text.clone()) } else { None })
        .collect()
}

fn has(s: &Scene, needle: &str) -> bool {
    texts(s).iter().any(|t| t.contains(needle))
}

fn pick(w: &mut Workspace, field: &str, cols: &[usize]) {
    let d = w.dialog_mut().expect("dialog open");
    let mut v = d.values().clone();
    v.set(field, Value::Vars(cols.to_vec()));
    d.set_values(v);
}

fn session_text(w: &Workspace) -> String {
    w.sessions().iter().map(|s| s.report.render_text()).collect()
}

#[test]
fn menu_dialog_run_puts_output_in_session_and_graphs_in_navigator() {
    let mut w = loaded();
    assert!(w.menu_mut().open_by_label("Graph"));
    assert!(w.menu().is_open());
    assert_eq!(w.execute(Command::OpenDialog("graph.histogram".into())), Outcome::None);
    assert!(!w.menu().is_open(), "opening a dialog closes the menu");
    assert!(w.dialog().is_some());
    assert!(has(&w.render(), "Histogram"), "dialog is drawn");
    // OK with nothing chosen keeps the dialog open and reports the problem.
    pick(&mut w, "vars", &[]);
    assert_eq!(w.key_down(Key::Enter, Mods::NONE), Outcome::None);
    assert!(w.dialog().is_some());
    assert!(!w.dialog().unwrap().problems().is_empty());
    pick(&mut w, "vars", &[0, 1]);
    w.key_down(Key::Enter, Mods::NONE);
    assert!(w.dialog().is_none());
    assert_eq!(w.graphs().len(), 2);
    assert_eq!(w.graphs()[0].name, "Histogram 1");
    assert_eq!(w.graphs()[1].name, "Histogram 2");
    assert_eq!(w.top(), Top::Session(w.active_session()));
    let scene = w.render();
    assert!(has(&scene, "Histogram 1") && has(&scene, "Histogram 2"), "Navigator lists graphs");
    assert!(has(&scene, "Session 1") && has(&scene, "Worksheet 1"));
}

#[test]
fn regression_end_to_end_with_storage_undo_and_redo() {
    let mut w = loaded();
    w.execute(Command::OpenDialog("stat.regression.fit".into()));
    pick(&mut w, "response", &[1]);
    pick(&mut w, "preds", &[0]);
    let d = w.dialog_mut().unwrap();
    let mut v = d.values().clone();
    v.set("store_fits", Value::Bool(true));
    v.set("store_resid", Value::Bool(true));
    d.set_values(v);
    assert_eq!(w.run_dialog(), Outcome::None);
    assert!(w.dialog().is_none());
    let text = session_text(&w);
    assert!(text.contains("Regression") && text.contains("Coef"), "{text}");
    assert_eq!(w.grid().worksheet().n_cols(), 4);
    assert!(w.render().items.len() > 50);
    w.execute(Command::Undo);
    assert_eq!(w.grid().worksheet().n_cols(), 2, "Edit > Undo removes the stored columns in one step");
    w.execute(Command::Redo);
    assert_eq!(w.grid().worksheet().n_cols(), 4);
}

#[test]
fn last_values_are_remembered_and_stale_columns_dropped() {
    let mut w = loaded();
    w.execute(Command::OpenDialog("stat.basic.ttest1".into()));
    pick(&mut w, "var", &[1]);
    let d = w.dialog_mut().unwrap();
    let mut v = d.values().clone();
    v.set("mu", Value::Num(12.5));
    d.set_values(v);
    w.run_dialog();
    w.execute(Command::OpenDialog("stat.basic.ttest1".into()));
    assert_eq!(w.dialog().unwrap().values().num("mu"), 12.5);
    assert_eq!(w.dialog().unwrap().values().vars("var"), &[1]);
    w.cancel_dialog();
    assert!(w.dialog().is_none());
    assert!(w.open_dialog("nope").is_err());
}

#[test]
fn run_errors_stay_in_the_dialog() {
    let mut w = loaded();
    w.execute(Command::OpenDialog("calc.calculator".into()));
    let d = w.dialog_mut().unwrap();
    let mut v = d.values().clone();
    v.set("target", Value::Num(3.0));
    v.set("expr", Value::Text("x +".into()));
    d.set_values(v);
    let o = w.run_dialog();
    assert!(matches!(o, Outcome::Error(_)), "{o:?}");
    assert!(w.dialog().is_some());
    assert!(!w.status().is_empty());
    assert!(!w.dialog().unwrap().problems().is_empty());
    assert_eq!(w.grid().worksheet().n_cols(), 2);
    let d = w.dialog_mut().unwrap();
    let mut v = d.values().clone();
    v.set("expr", Value::Text("x * 2".into()));
    d.set_values(v);
    assert_eq!(w.run_dialog(), Outcome::None);
    assert_eq!(w.grid().worksheet().n_cols(), 3);
    assert!(w.status().is_empty());
}

#[test]
fn navigator_rename_delete_export() {
    let mut w = loaded();
    for id in ["graph.histogram", "graph.boxplot"] {
        w.open_dialog(id).unwrap();
        pick(&mut w, "vars", &[1]);
        w.run_dialog();
    }
    let g1 = w.graphs()[0].id;
    let g2 = w.graphs()[1].id;
    // rename through the in-place editor
    assert_eq!(w.navigator_action(Target::Graph(g1), NavAction::Rename), Outcome::None);
    assert!(w.navigator().renaming().is_some());
    w.set_focus(Focus::Navigator);
    for _ in 0..20 {
        w.key_down(Key::Backspace, Mods::NONE);
    }
    for c in "Fit plot".chars() {
        w.key_down(Key::Char(c), Mods::NONE);
    }
    w.key_down(Key::Enter, Mods::NONE);
    assert_eq!(w.graphs()[0].name, "Fit plot");
    // duplicate names are refused
    w.navigator_action(Target::Graph(g2), NavAction::Rename);
    for _ in 0..20 {
        w.key_down(Key::Backspace, Mods::NONE);
    }
    for c in "fit plot".chars() {
        w.key_down(Key::Char(c), Mods::NONE);
    }
    assert!(w.key_down(Key::Enter, Mods::NONE) != Outcome::None);
    assert_ne!(w.graphs()[1].name, "fit plot");
    // exports
    match w.navigator_action(Target::Graph(g1), NavAction::Export) {
        Outcome::File(f) => {
            assert_eq!(f.name, "Fit plot.svg");
            assert!(String::from_utf8(f.bytes).unwrap().starts_with("<svg"));
        }
        o => panic!("{o:?}"),
    }
    match w.navigator_action(Target::Graph(g1), NavAction::ExportPng) {
        Outcome::File(f) => assert_eq!(&f.bytes[1..4], b"PNG"),
        o => panic!("{o:?}"),
    }
    match w.navigator_action(Target::Worksheet(0), NavAction::Export) {
        Outcome::File(f) => {
            assert_eq!(f.name, "Worksheet 1.csv");
            let t = String::from_utf8(f.bytes).unwrap();
            assert!(t.starts_with("x,y") && t.lines().nth(1) == Some("0,1"), "{t:?}");
        }
        o => panic!("{o:?}"),
    }
    let sid = w.active_session();
    match w.navigator_action(Target::Session(sid), NavAction::Export) {
        Outcome::File(f) => assert!(String::from_utf8(f.bytes).unwrap().contains("Histogram")),
        o => panic!("{o:?}"),
    }
    // delete
    w.navigator_action(Target::Graph(g2), NavAction::Delete);
    assert_eq!(w.graphs().len(), 1);
    assert!(
        matches!(w.navigator_action(Target::Session(sid), NavAction::Delete), Outcome::Error(_)),
        "last Session window is kept"
    );
    assert!(
        matches!(w.navigator_action(Target::Worksheet(0), NavAction::Delete), Outcome::Error(_)),
        "last worksheet is kept"
    );
}

#[test]
fn sheets_and_sessions_can_be_added_renamed_and_switched() {
    let mut w = loaded();
    w.navigator_action(Target::Folder(openstats_ui::workspace::Folder::Worksheets), NavAction::NewWorksheet);
    assert_eq!(w.grid().sheet_count(), 2);
    assert_eq!(w.grid().sheet_index(), 1);
    assert_eq!(w.grid().worksheet().n_cols(), 0, "new sheet is empty and active");
    w.navigator_action(Target::Worksheet(0), NavAction::Open);
    assert_eq!(w.grid().sheet_index(), 0);
    w.navigator_action(Target::Worksheet(1), NavAction::Duplicate);
    assert_eq!(w.grid().sheet_count(), 3);
    w.navigator_action(Target::Folder(openstats_ui::workspace::Folder::Sessions), NavAction::NewSession);
    assert_eq!(w.sessions().len(), 2);
    assert_eq!(w.sessions()[1].name, "Session 2");
    assert_eq!(w.active_session(), w.sessions()[1].id, "new Session window becomes the output target");
    w.open_dialog("graph.histogram").unwrap();
    pick(&mut w, "vars", &[1]);
    w.grid_mut().set_sheet(0).unwrap();
    w.run_dialog();
    assert_eq!(w.sessions()[1].report.sections.len(), 1);
    assert_eq!(w.sessions()[0].report.sections.len(), 0);
    let sid = w.sessions()[1].id;
    w.navigator_action(Target::Session(sid), NavAction::Delete);
    assert!(w.graphs().is_empty(), "graphs follow the Session window they came from");
    assert_eq!(w.sessions().len(), 1);
    w.navigator_action(Target::Worksheet(0), NavAction::Rename);
    w.set_focus(Focus::Navigator);
    for c in "Data".chars() {
        w.key_down(Key::Char(c), Mods::NONE);
    }
    w.key_down(Key::Enter, Mods::NONE);
    assert!(w.grid().workbook().worksheets.iter().any(|s| s.name.ends_with("Data")));
}

#[test]
fn copy_and_print_session() {
    let mut w = loaded();
    w.open_dialog("stat.basic.descriptive").unwrap();
    pick(&mut w, "vars", &[0, 1]);
    w.run_dialog();
    w.set_focus(Focus::Session);
    w.key_down(Key::Char('a'), Mods::CTRL);
    match w.execute(Command::Copy) {
        Outcome::Clipboard(t) => assert!(t.contains("Descriptive Statistics") && t.contains("Mean"), "{t}"),
        o => panic!("{o:?}"),
    }
    match w.key_down(Key::Char('p'), Mods::CTRL) {
        Outcome::Print(pages) => {
            assert!(!pages.is_empty());
            assert!(pages.iter().all(|p| (p.width - 612.0).abs() < 1e-9 && (p.height - 792.0).abs() < 1e-9));
            assert!(pages.iter().any(|p| has(p, "Mean")));
        }
        o => panic!("{o:?}"),
    }
    w.execute(Command::ClearSession);
    assert!(w.sessions()[0].report.sections.is_empty());
}

#[test]
fn save_and_open_round_trip_keeps_sessions_graphs_and_data() {
    let mut w = loaded();
    w.open_dialog("graph.scatter").unwrap();
    pick(&mut w, "y", &[1]);
    pick(&mut w, "x", &[0]);
    w.run_dialog();
    w.open_dialog("stat.basic.ttest1").unwrap();
    pick(&mut w, "var", &[1]);
    w.run_dialog();
    let before_text = session_text(&w);
    let before_graph = w.graphs()[0].clone();
    let bytes = w.save_project();
    assert!(
        !w.grid().workbook().metadata.keys().any(|k| k.starts_with("ui.")),
        "UI state is not left in the live workbook"
    );
    let mut w2 = Workspace::new(1000.0, 700.0);
    assert_eq!(w2.open_project(&bytes), Outcome::None);
    assert_eq!(session_text(&w2), before_text);
    assert_eq!(w2.graphs().len(), 1);
    assert_eq!(w2.graphs()[0].spec, before_graph.spec);
    assert_eq!(w2.graphs()[0].name, before_graph.name);
    assert_eq!(w2.grid().worksheet().n_cols(), 2);
    assert_eq!(w2.grid().worksheet().columns[1].get(3).as_f64(), w.grid().worksheet().columns[1].get(3).as_f64());
    assert!(has(&w2.render(), "Scatterplot"));
    assert!(matches!(w2.open_project(b"not a project"), Outcome::Error(_)));
    assert_eq!(w2.graphs().len(), 1, "a failed open changes nothing");
    // keyboard shortcuts
    assert!(matches!(w.key_down(Key::Char('s'), Mods::CTRL), Outcome::File(_)));
    assert_eq!(w.key_down(Key::Char('o'), Mods::CTRL), Outcome::NeedOpenFile);
    w.key_down(Key::Char('n'), Mods::CTRL);
    assert!(w.graphs().is_empty() && w.grid().worksheet().n_cols() == 0);
    assert_eq!(w.sessions().len(), 1);
}

#[test]
fn input_routes_to_the_focused_pane() {
    let mut w = loaded();
    // pointer: grid area
    let x = 400.0;
    let y = 700.0;
    w.pointer_down(x, y, Button::Left, Mods::NONE, 1);
    w.pointer_up();
    assert_eq!(w.focus(), Focus::Grid);
    w.key_down(Key::Char('7'), Mods::NONE);
    w.key_down(Key::Enter, Mods::NONE);
    let a = w.grid().active();
    assert!(a.row >= 1 || a.col > 0, "grid moved: {a:?}");
    // pointer: navigator
    w.pointer_down(30.0, 60.0, Button::Left, Mods::NONE, 1);
    assert_eq!(w.focus(), Focus::Navigator);
    // pointer: session pane scrolls with the wheel
    w.open_dialog("stat.basic.descriptive").unwrap();
    pick(&mut w, "vars", &[0, 1]);
    w.run_dialog();
    w.pointer_down(600.0, 200.0, Button::Left, Mods::NONE, 1);
    assert_eq!(w.focus(), Focus::Session);
    let top0 = w.session_view().scroll();
    w.wheel(600.0, 200.0, 0.0, 30.0);
    w.key_down(Key::Home, Mods::NONE);
    assert!(w.session_view().scroll() <= top0 + 1e-9);
    // a modal dialog swallows everything and Escape closes it
    w.open_dialog("data.sort").unwrap();
    w.pointer_down(10.0, 700.0, Button::Left, Mods::NONE, 1);
    w.key_down(Key::Char('q'), Mods::NONE);
    assert!(w.dialog().is_some());
    w.key_down(Key::Escape, Mods::NONE);
    assert!(w.dialog().is_none());
}

#[test]
fn menu_bar_clicks_open_menus_and_run_commands() {
    let mut w = loaded();
    let rects = w.menu().bar_rects();
    let labels: Vec<&str> = w.menu().menus.iter().map(|m| m.label.as_str()).collect();
    assert_eq!(labels, ["File", "Edit", "Data", "Calc", "Stat", "Graph", "View", "Tools", "Help", "Assistant"]);
    // click "Stat"
    let r = rects[4];
    w.pointer_down(r.x + 3.0, r.y + 3.0, Button::Left, Mods::NONE, 1);
    assert!(w.menu().is_open());
    let panels = w.menu().panels();
    assert!(!panels.is_empty());
    // click the first row of the panel opens a submenu or a dialog without panicking
    let (row, _) = panels[0].rows[0];
    w.pointer_down(row.x + 5.0, row.y + 3.0, Button::Left, Mods::NONE, 1);
    w.key_down(Key::Escape, Mods::NONE);
    w.key_down(Key::Escape, Mods::NONE);
    assert!(!w.menu().is_open());
    // Data > Insert Rows through the command path
    let n = w.grid().data_rows();
    w.grid_mut().select_cell(CellRef { row: 2, col: 0 });
    w.execute(Command::InsertRows);
    assert_eq!(w.grid().data_rows(), n + 1);
    w.execute(Command::Undo);
    assert_eq!(w.grid().data_rows(), n);
}

#[test]
fn assistant_recommends_and_opens_the_matching_dialog() {
    let mut w = loaded();
    w.grid_mut().select_cell(CellRef { row: 0, col: 1 });
    w.execute(Command::Assistant(Topic::Regression));
    assert!(w.dialog().is_some());
    assert!(w.sessions()[0].report.render_text().contains("Assistant"));
}

#[test]
fn resize_keeps_every_pane_inside_the_window() {
    let mut w = loaded();
    for (a, b) in [(1600.0, 900.0), (800.0, 500.0), (500.0, 300.0)] {
        w.set_size(a, b);
        let s = w.render();
        assert!((s.width - a).abs() < 1e-9 && (s.height - b).abs() < 1e-9);
        assert!(!s.items.is_empty());
    }
    w.set_split(0.7);
    w.set_split(5.0);
    let _ = w.render();
}

#[test]
fn every_menu_command_is_reachable_and_survives_execution() {
    let mut w = loaded();
    let ids: Vec<String> = w.registry().all().iter().map(|a| a.spec.id.clone()).collect();
    for id in ids {
        assert_eq!(w.execute(Command::OpenDialog(id.clone())), Outcome::None, "{id}");
        let d = w.dialog().unwrap();
        assert_eq!(d.spec().id, id);
        let _ = w.render();
        w.cancel_dialog();
    }
}

#[test]
fn editor_clipboard_ime_and_dialog_text_through_the_workspace() {
    let mut w = loaded();
    w.set_focus(Focus::Grid);
    w.grid_mut().begin_edit(Some("alpha beta"), false);
    w.key_down(Key::Char('a'), Mods::CTRL);
    assert_eq!(w.key_down(Key::Char('c'), Mods::CTRL), Outcome::Clipboard("alpha beta".into()));
    assert_eq!(w.key_down(Key::Char('x'), Mods::CTRL), Outcome::Clipboard("alpha beta".into()));
    assert_eq!(w.grid().editor().unwrap().buffer, "");
    assert_eq!(w.paste_text("pasted"), Outcome::None);
    assert_eq!(w.grid().editor().unwrap().buffer, "pasted");
    w.ime_update("か", 1);
    assert!(w.ime_anchor().is_some());
    w.ime_commit("か");
    assert_eq!(w.grid().editor().unwrap().buffer, "pastedか");
    w.ime_cancel();
    w.key_down(Key::Escape, Mods::NONE);
    assert!(w.grid().editor().is_none() && w.ime_anchor().is_none());

    // Into a dialog text field.
    w.open_dialog("calc.calculator").unwrap();
    w.dialog_mut().unwrap().focus_field("expr");
    w.paste_text("x * 2");
    w.ime_commit("+1");
    assert_eq!(w.dialog().unwrap().values().text("expr"), "x * 2+1");
}

#[test]
fn dialog_list_takes_ctrl_and_shift_clicks_through_the_workspace() {
    let mut w = loaded();
    w.open_dialog("stat.basic.descriptive").unwrap();
    let (dw, dh) = (openstats_ui::dialogs::dialog::WIDTH, openstats_ui::dialogs::dialog::HEIGHT);
    let ox = (1200.0 - dw) / 2.0;
    let oy = (800.0 - dh) / 2.0;
    w.pointer_down(ox + 30.0, oy + 44.0 + 5.0, Button::Left, Mods::NONE, 1);
    w.pointer_down(ox + 30.0, oy + 44.0 + 25.0, Button::Left, Mods::SHIFT, 1);
    assert_eq!(w.dialog().unwrap().selected_columns(), &[0, 1]);
    w.dialog_mut().unwrap().focus_field("vars");
    assert!(w.dialog_mut().unwrap().add_selected());
    assert_eq!(w.dialog().unwrap().values().vars("vars"), &[0, 1]);
}

fn tiny_xlsx() -> Vec<u8> {
    use openstats_core::io::zip::ZipWriter;
    let mut z = ZipWriter::new();
    let files: [(&str, &str); 6] = [
        (
            "[Content_Types].xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>"#,
        ),
        (
            "_rels/.rels",
            r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#,
        ),
        (
            "xl/workbook.xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Imported" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
        ),
        (
            "xl/_rels/workbook.xml.rels",
            r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/></Relationships>"#,
        ),
        (
            "xl/sharedStrings.xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2"><si><t>height</t></si><si><t>weight</t></si></sst>"#,
        ),
        (
            "xl/worksheets/sheet1.xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row><row r="2"><c r="A2"><v>1.5</v></c><c r="B2"><v>60</v></c></row><row r="3"><c r="A3"><v>1.7</v></c><c r="B3"><v>72</v></c></row></sheetData></worksheet>"#,
        ),
    ];
    for (n, c) in files {
        z.add(n, c.as_bytes(), 6).unwrap();
    }
    z.finish()
}

#[test]
fn importing_csv_xlsx_and_refusing_projects() {
    let mut w = loaded();
    assert_eq!(w.execute(Command::ImportData), Outcome::NeedImportFile);
    assert_eq!(w.import_data("weights.csv", b"a,b\n1,2\n3,4\n5,6\n"), Outcome::None);
    assert_eq!(w.grid().sheet_count(), 2);
    assert_eq!(w.grid().worksheet().name, "weights");
    assert_eq!(w.grid().worksheet().n_cols(), 2);
    assert_eq!(w.grid().worksheet().columns[1].get(2).as_f64(), 6.0);
    assert_eq!(w.import_data("weights.csv", b"a\n1\n"), Outcome::None);
    assert_eq!(w.grid().worksheet().name, "weights (2)", "names stay unique");
    assert_eq!(w.import_data("t.tsv", b"x\ty\n1\t2\n"), Outcome::None);
    assert_eq!(w.grid().worksheet().n_cols(), 2);
    assert_eq!(w.import_data("data.xlsx", &tiny_xlsx()), Outcome::None);
    assert_eq!(w.grid().worksheet().name, "Imported");
    assert_eq!(w.grid().worksheet().columns[0].name(), "height");
    assert_eq!(w.grid().worksheet().columns[1].get(1).as_f64(), 72.0);
    let project = w.save_project();
    assert!(matches!(w.import_data("p.otb", &project), Outcome::Error(_)));
    assert!(matches!(w.import_data("broken.xlsx", b"PK\x03\x04not really"), Outcome::Error(_)));
    assert!(matches!(w.import_data("x.mpx", b""), Outcome::Error(_) | Outcome::None));
    // imported sheets appear in the Navigator and can be analysed
    assert!(has(&w.render(), "Imported"));
}

#[test]
fn graph_tear_off_carries_the_chart_and_round_trips() {
    let mut w = loaded();
    w.open_dialog("graph.histogram").unwrap();
    pick(&mut w, "vars", &[1]);
    w.run_dialog();
    let id = w.graphs()[0].id;
    assert!(Navigator_actions_include_tear_off(id));
    match w.navigator_action(Target::Graph(id), NavAction::TearOff) {
        Outcome::TearOffGraph { name, json } => {
            assert_eq!(name, "Histogram 1");
            let r = openstats_stats::report::SessionReport::from_json(&json).unwrap();
            assert_eq!(r.sections.len(), 1);
            assert!(
                matches!(&r.sections[0].blocks[0], openstats_stats::report::Block::Chart(c) if *c == w.graphs()[0].spec)
            );
        }
        o => panic!("{o:?}"),
    }
    assert!(w.graph_json(openstats_ui::workspace::ItemId(9999)).is_none());
    assert!(matches!(w.navigator_action(Target::Session(w.active_session()), NavAction::TearOff), Outcome::None));
}

#[allow(non_snake_case)]
fn Navigator_actions_include_tear_off(id: openstats_ui::workspace::ItemId) -> bool {
    openstats_ui::workspace::Navigator::actions_for(Target::Graph(id)).contains(&NavAction::TearOff)
}

#[test]
fn command_ids_round_trip_and_the_menu_exports_as_json() {
    let w = loaded();
    let mut n = 0;
    let mut seen = std::collections::BTreeSet::new();
    fn walk(es: &[openstats_ui::dialogs::Entry], f: &mut dyn FnMut(&Command)) {
        for e in es {
            match e {
                openstats_ui::dialogs::Entry::Item { command, .. } => f(command),
                openstats_ui::dialogs::Entry::Submenu { entries, .. } => walk(entries, f),
                openstats_ui::dialogs::Entry::Separator => {}
            }
        }
    }
    for m in &w.menu().menus {
        walk(&m.entries, &mut |c| {
            n += 1;
            assert_eq!(Command::from_id(&c.id()).as_ref(), Some(c), "{}", c.id());
            assert!(seen.insert(c.id()), "ids are unique: {}", c.id());
        });
    }
    assert!(n > 40);
    assert_eq!(Command::from_id("nonsense"), None);
    assert_eq!(Command::from_id("assistant:nope"), None);
    let json = w.menu().to_json(&|c| *c != Command::Redo);
    assert!(
        json.starts_with("[{\"label\":\"File\"")
            && json.contains("\"id\":\"file.save\"")
            && json.contains("\"separator\":true")
    );
    assert!(json.contains("\"id\":\"edit.redo\",\"shortcut\":\"Ctrl+Y / Ctrl+Shift+Z\",\"enabled\":false"));
    assert!(json.contains("\"id\":\"dialog:stat.anova.glm\"") && json.contains("Hotelling"));
    // balanced braces/brackets (a cheap structural check; the host's JSON.parse is the real one)
    let (mut depth, mut in_str, mut esc) = (0i32, false, false);
    for ch in json.chars() {
        if in_str {
            match (esc, ch) {
                (true, _) => esc = false,
                (false, '\\') => esc = true,
                (false, '"') => in_str = false,
                _ => {}
            }
        } else {
            match ch {
                '"' => in_str = true,
                '{' | '[' => depth += 1,
                '}' | ']' => depth -= 1,
                _ => {}
            }
            assert!(depth >= 0);
        }
    }
    assert_eq!((depth, in_str), (0, false));
}
