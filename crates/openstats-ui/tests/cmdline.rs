//! The command line, the History list and the HTML / Word exports of the Session window.

use openstats_charts::core::{Scene, Shape};
use openstats_core::column::Column;
use openstats_core::io::xml;
use openstats_core::io::zip::ZipArchive;
use openstats_core::Workbook;
use openstats_ui::dialogs::{Command, Value};
use openstats_ui::grid::{GridController, Key, Mods};
use openstats_ui::workspace::{Focus, HistoryKind, Outcome, Workspace};

fn loaded() -> Workspace {
    let mut w = Workspace::new(1200.0, 800.0);
    let mut wb = Workbook::new();
    let x: Vec<f64> = (1..=8).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|v| 3.0 + 2.0 * v + if (*v as i64) % 2 == 0 { 0.5 } else { -0.5 }).collect();
    for c in [Column::numeric("x", x), Column::numeric("y", y)] {
        wb.worksheets[0].add_column(c).unwrap();
    }
    *w.grid_mut() = GridController::new(wb, 800.0, 300.0);
    w.set_size(1200.0, 800.0);
    w
}

fn texts(s: &Scene) -> Vec<String> {
    s.items
        .iter()
        .filter_map(|i| if let Shape::Text { text, .. } = &i.shape { Some(text.clone()) } else { None })
        .collect()
}

fn session_text(w: &Workspace) -> String {
    w.sessions().iter().map(|s| s.report.render_text()).collect()
}

fn constant(w: &Workspace, k: usize) -> f64 {
    w.grid().workbook().worksheets[0].constants.get(&k).map_or(f64::NAN, |v| v.as_f64())
}

fn col(w: &Workspace, c: usize) -> Vec<f64> {
    match w.grid().workbook().worksheets[0].column(c) {
        Ok(col) => (0..col.len()).map(|i| col.get(i).as_f64()).collect(),
        Err(_) => Vec::new(),
    }
}

fn type_line(w: &mut Workspace, line: &str) -> Outcome {
    for c in line.chars() {
        w.key_down(Key::Char(c), Mods::default());
    }
    w.key_down(Key::Enter, Mods::default())
}

#[test]
fn ctrl_l_shows_the_command_line_and_runs_typed_commands() {
    let mut w = loaded();
    assert!(!w.command_line().visible);
    assert!(!texts(&w.render()).iter().any(|t| t == "MTB >"));
    w.key_down(Key::Char('l'), Mods::CTRL);
    assert!(w.command_line().visible);
    assert_eq!(w.focus(), Focus::Command);
    assert!(texts(&w.render()).iter().any(|t| t == "MTB >"));

    assert_eq!(type_line(&mut w, "MEAN x K1"), Outcome::None);
    assert!((constant(&w, 1) - 4.5).abs() < 1e-12);
    let t = session_text(&w);
    assert!(t.contains("MTB > MEAN x K1"), "{t}");
    assert_eq!(w.command_line().text(), "");
    let h = w.command_line().entries();
    assert_eq!((h.len(), h[0].kind, h[0].ok), (1, HistoryKind::Command, true));

    // One undo step per command line; formula columns recalculated.
    assert_eq!(type_line(&mut w, "LET C3 = x * 2"), Outcome::None);
    assert_eq!(col(&w, 2)[7], 16.0);
    w.execute(Command::Undo);
    assert!(col(&w, 2).is_empty());
    w.execute(Command::Redo);
    assert_eq!(col(&w, 2)[7], 16.0);

    // Ctrl+L again hides it; the keyboard goes back to the worksheet.
    w.key_down(Key::Char('l'), Mods::CTRL);
    assert!(!w.command_line().visible);
    assert_eq!(w.focus(), Focus::Grid);
}

#[test]
fn errors_are_reported_and_kept_in_the_history() {
    let mut w = loaded();
    w.execute(Command::CommandLine);
    let o = type_line(&mut w, "FROBNICATE x");
    assert!(matches!(o, Outcome::Error(ref m) if m.contains("not a command")), "{o:?}");
    let t = session_text(&w);
    assert!(t.contains("MTB > FROBNICATE x") && t.contains("not a command"), "{t}");
    let h = w.command_line().entries();
    assert_eq!((h[0].ok, h[0].text.as_str()), (false, "FROBNICATE x"));
    // A failing script leaves the data as it was, and the next command still works.
    let o = w.run_command_line("SET C5\n1 2 3\nEND\nMEAN C9 K1");
    assert!(matches!(o, Outcome::Error(_)), "{o:?}");
    assert_eq!(type_line(&mut w, "SUM y K2"), Outcome::None);
    assert!(constant(&w, 2) > 0.0);
    // Syntax errors are caught before anything runs.
    assert!(matches!(w.run_command_line("DO i = 1:3"), Outcome::Error(_)));
    // Blank lines do nothing and are not recorded.
    let n = w.command_line().entries().len();
    assert_eq!(w.run_command_line("   \n "), Outcome::None);
    assert_eq!(w.command_line().entries().len(), n);
}

#[test]
fn up_and_down_recall_commands_and_escape_clears() {
    let mut w = loaded();
    w.execute(Command::CommandLine);
    type_line(&mut w, "MEAN x K1");
    type_line(&mut w, "SUM x K2");
    for c in "ab".chars() {
        w.key_down(Key::Char(c), Mods::default());
    }
    w.key_down(Key::Up, Mods::default());
    assert_eq!(w.command_line().text(), "SUM x K2");
    w.key_down(Key::Up, Mods::default());
    assert_eq!(w.command_line().text(), "MEAN x K1");
    w.key_down(Key::Up, Mods::default());
    assert_eq!(w.command_line().text(), "MEAN x K1", "stays at the oldest");
    w.key_down(Key::Down, Mods::default());
    assert_eq!(w.command_line().text(), "SUM x K2");
    w.key_down(Key::Down, Mods::default());
    assert_eq!(w.command_line().text(), "ab", "the draft comes back");
    w.key_down(Key::Escape, Mods::default());
    assert_eq!(w.command_line().text(), "");
    assert_eq!(w.focus(), Focus::Command);
    w.key_down(Key::Escape, Mods::default());
    assert_eq!(w.focus(), Focus::Grid);
}

#[test]
fn editing_keys_and_clipboard() {
    let mut w = loaded();
    w.execute(Command::CommandLine);
    for c in "MEAN yy".chars() {
        w.key_down(Key::Char(c), Mods::default());
    }
    w.key_down(Key::Left, Mods::default());
    w.key_down(Key::Backspace, Mods::default());
    assert_eq!(w.command_line().text(), "MEAN y");
    w.key_down(Key::Home, Mods::default());
    w.key_down(Key::Delete, Mods::default());
    assert_eq!(w.command_line().text(), "EAN y");
    w.key_down(Key::End, Mods::default());
    w.key_down(Key::Char('a'), Mods::CTRL);
    assert_eq!(w.key_down(Key::Char('c'), Mods::CTRL), Outcome::Clipboard("EAN y".into()));
    assert_eq!(w.key_down(Key::Char('x'), Mods::CTRL), Outcome::Clipboard("EAN y".into()));
    assert_eq!(w.command_line().text(), "");
}

#[test]
fn history_lists_dialogs_macros_and_commands_and_survives_save() {
    let mut w = loaded();
    w.open_dialog("stat.basic.descriptive").unwrap();
    {
        let d = w.dialog_mut().unwrap();
        let mut v = d.values().clone();
        v.set("vars", Value::Vars(vec![0]));
        d.set_values(v);
    }
    assert_eq!(w.run_dialog(), Outcome::None);
    let m = w.add_macro(Some("Setup"), "K5 = 55\n");
    w.open_macro(m);
    w.execute(Command::MacroRun);
    w.execute(Command::CommandLine);
    type_line(&mut w, "EXECUTE 'Setup'");
    type_line(&mut w, "MEAN x K1");
    let kinds: Vec<HistoryKind> = w.command_line().entries().iter().map(|e| e.kind).collect();
    assert_eq!(
        kinds,
        [HistoryKind::Dialog, HistoryKind::Macro, HistoryKind::Command, HistoryKind::Command],
        "{kinds:?}"
    );
    assert_eq!(w.command_line().entries()[0].text, "Display Descriptive Statistics");
    assert!((constant(&w, 5) - 55.0).abs() < 1e-12);

    w.execute(Command::HistoryPane);
    assert!(w.command_line().history_open);
    let t = texts(&w.render());
    assert!(t.iter().any(|s| s.starts_with("History (")), "{t:?}");
    assert!(t.iter().any(|s| s == "EXECUTE 'Setup'"), "{t:?}");

    // Save and reopen: the history and the multi-line text round trip.
    w.run_command_line("SET C6\n1 2\nEND");
    let bytes = w.save_project();
    let mut w2 = Workspace::new(1200.0, 800.0);
    assert_eq!(w2.open_project(&bytes), Outcome::None);
    assert_eq!(w2.command_line().entries(), w.command_line().entries());
    assert!(w2.command_line().entries().last().unwrap().text.contains('\n'));
    // A new project forgets it.
    w2.execute(Command::NewProject);
    assert!(w2.command_line().entries().is_empty());
}

#[test]
fn clicking_a_history_row_recalls_the_command() {
    let mut w = loaded();
    w.execute(Command::CommandLine);
    type_line(&mut w, "MEAN x K1");
    type_line(&mut w, "SUM x K2");
    w.execute(Command::HistoryPane);
    // Find the drawn row and click it.
    let s = w.render();
    let (x, y) = s
        .items
        .iter()
        .find_map(|i| match &i.shape {
            Shape::Text { text, at, .. } if text == "MEAN x K1" => Some((at.x, at.y)),
            _ => None,
        })
        .expect("row drawn");
    w.pointer_down(x + 4.0, y - 4.0, openstats_ui::workspace::Button::Left, Mods::default(), 1);
    assert_eq!(w.command_line().text(), "MEAN x K1");
    assert_eq!(w.focus(), Focus::Command);
}

#[test]
fn outfile_returns_a_text_file_and_stack_and_execute_work_from_the_command_line() {
    let mut w = loaded();
    let o = w.run_command_line("OUTFILE 'report.txt'\nDESCRIBE x\nNOOUTFILE");
    match o {
        Outcome::File(f) => {
            assert_eq!(f.name, "report.txt");
            let t = String::from_utf8(f.bytes).unwrap();
            assert!(t.contains("Descriptive"), "{t}");
        }
        o => panic!("{o:?}"),
    }
    assert_eq!(w.run_command_line("STACK x y INTO C10; SUBSCRIPTS C11."), Outcome::None);
    assert_eq!(col(&w, 9).len(), 16);
    assert_eq!(col(&w, 10)[8], 2.0);
    // Two files: the first is returned and the rest are reported.
    let o = w.run_command_line("OUTFILE 'a.txt'\nTEXT 'A'\nOUTFILE 'b.txt'\nTEXT 'B'");
    assert!(matches!(o, Outcome::File(ref f) if f.name == "a.txt"));
    assert!(w.status().contains("b.txt"), "{}", w.status());
}

#[test]
fn pasting_into_the_command_line() {
    let mut w = loaded();
    w.execute(Command::CommandLine);
    w.paste_text("MEAN y K3\n");
    assert_eq!(w.command_line().text(), "MEAN y K3");
    w.key_down(Key::Enter, Mods::default());
    assert!(constant(&w, 3) > 0.0);
    w.ime_commit("TEXT 'hi'");
    assert_eq!(w.command_line().text(), "TEXT 'hi'");
}

// ------------------------------------------------------------------------------- exports

fn report_workspace() -> Workspace {
    let mut w = loaded();
    w.run_command_line("DESCRIBE x y");
    let _ = w.open_dialog("graph.bar");
    if let Some(d) = w.dialog_mut() {
        let mut v = d.values().clone();
        v.set("cat", Value::Vars(vec![0]));
        d.set_values(v);
    }
    let _ = w.run_dialog();
    w
}

fn exported(w: &mut Workspace, cmd: Command) -> openstats_ui::workspace::ExportFile {
    match w.execute(cmd) {
        Outcome::File(f) => f,
        o => panic!("{o:?}"),
    }
}

#[test]
fn html_export_has_tables_charts_and_escapes_text() {
    let mut w = report_workspace();
    w.run_command_line("TEXT '<b>bold</b> & more'");
    let f = exported(&mut w, Command::ExportSessionHtml);
    assert_eq!(f.mime, "text/html");
    assert!(f.name.ends_with(".html"));
    let h = String::from_utf8(f.bytes).unwrap();
    assert!(
        h.starts_with("<!DOCTYPE html>") && h.contains("<table>") && h.contains("<svg"),
        "{}",
        &h[..300.min(h.len())]
    );
    assert!(h.contains("&lt;b&gt;bold&lt;/b&gt; &amp; more"));
    assert!(!h.contains("<b>bold</b>"));
    assert!(h.contains("scope=\"col\"") && h.contains("class=\"num\"") && h.contains("class=\"txt\""));
    assert!(h.contains("role=\"img\" aria-label="));
    // Well-formed apart from the doctype: every tag the report produced is closed.
    assert_eq!(h.matches("<table>").count(), h.matches("</table>").count());
    assert_eq!(h.matches("<section>").count(), h.matches("</section>").count());
}

#[test]
fn docx_export_is_a_valid_package() {
    let mut w = report_workspace();
    let f = exported(&mut w, Command::ExportSessionDocx);
    assert!(f.name.ends_with(".docx"));
    assert!(f.mime.contains("wordprocessingml"));
    let z = ZipArchive::new(&f.bytes).expect("a zip file");
    for name in
        ["[Content_Types].xml", "_rels/.rels", "word/document.xml", "word/styles.xml", "word/_rels/document.xml.rels"]
    {
        let xml_text = String::from_utf8(z.read(name).unwrap()).unwrap();
        xml::events(&xml_text).unwrap_or_else(|e| panic!("{name} is not well-formed XML: {e}"));
    }
    let doc = String::from_utf8(z.read("word/document.xml").unwrap()).unwrap();
    assert!(doc.contains("<w:tbl>") && doc.contains("<w:drawing>") && doc.contains("Heading2"));
    let png = z.read("word/media/chart1.png").unwrap();
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    let rels = String::from_utf8(z.read("word/_rels/document.xml.rels").unwrap()).unwrap();
    assert!(rels.contains("rIdImg1") && rels.contains("media/chart1.png"));
    // Optional: write the file for an external check (python-docx, LibreOffice).
    if let Ok(dir) = std::env::var("OPENSTATS_DOCX_DIR") {
        std::fs::write(std::path::Path::new(&dir).join("session.docx"), &f.bytes).unwrap();
        let h = exported(&mut w, Command::ExportSessionHtml);
        std::fs::write(std::path::Path::new(&dir).join("session.html"), &h.bytes).unwrap();
    }
}

#[test]
fn control_characters_never_reach_the_xml() {
    let mut w = loaded();
    w.run_command_line("TEXT 'tab\there \u{1}bell'");
    let f = exported(&mut w, Command::ExportSessionDocx);
    let z = ZipArchive::new(&f.bytes).unwrap();
    let doc = String::from_utf8(z.read("word/document.xml").unwrap()).unwrap();
    assert!(xml::events(&doc).is_ok());
    assert!(!doc.contains('\u{1}'));
}

#[test]
fn every_new_command_id_round_trips() {
    for c in [Command::CommandLine, Command::HistoryPane, Command::ExportSessionHtml, Command::ExportSessionDocx] {
        assert_eq!(Command::from_id(&c.id()), Some(c));
    }
}
