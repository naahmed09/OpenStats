//! The Macro Editor, Navigator folder, menus, persistence and run controls of the workspace.

use openstats_charts::core::{Scene, Shape};
use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_ui::dialogs::Command;
use openstats_ui::grid::{GridController, Key, Mods};
use openstats_ui::workspace::{Button, Focus, NavAction, Outcome, RunState, Target, Top, Workspace};

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

fn has(s: &Scene, needle: &str) -> bool {
    texts(s).iter().any(|t| t.contains(needle))
}

fn session_text(w: &Workspace) -> String {
    w.sessions().iter().map(|s| s.report.render_text()).collect()
}

fn open(w: &mut Workspace, name: &str, src: &str) {
    let id = w.add_macro(Some(name), src);
    w.open_macro(id);
}

fn col(w: &Workspace, c: usize) -> Vec<f64> {
    match w.grid().workbook().worksheets[0].column(c) {
        Ok(col) => (0..col.len()).map(|i| col.get(i).as_f64()).collect(),
        Err(_) => Vec::new(),
    }
}

fn constant(w: &Workspace, k: usize) -> f64 {
    w.grid().workbook().worksheets[0].constants.get(&k).map_or(f64::NAN, |v| v.as_f64())
}

#[test]
fn new_macro_opens_the_editor_above_the_session_window() {
    let mut w = loaded();
    assert_eq!(w.execute(Command::MacroNew), Outcome::None);
    assert_eq!(w.macros().len(), 1);
    assert_eq!(w.macros()[0].name, "Macro 1");
    assert_eq!(w.top(), Top::Macro(w.macros()[0].id));
    assert_eq!(w.focus(), Focus::Session);
    let scene = w.render();
    for needle in ["Run", "Step", "Stop", "Macros", "Macro 1", "Session 1"] {
        assert!(has(&scene, needle), "missing {needle}");
    }
    assert_eq!(w.execute(Command::MacroNew), Outcome::None);
    assert_eq!(w.macros()[1].name, "Macro 2");
}

#[test]
fn typing_in_the_editor_updates_the_macro() {
    let mut w = loaded();
    w.execute(Command::MacroNew);
    for ch in "LET K1 = 5".chars() {
        w.key_down(Key::Char(ch), Mods::NONE);
    }
    assert_eq!(w.macros()[0].source, "LET K1 = 5");
    w.key_down(Key::Enter, Mods::NONE);
    w.paste_text("LET K2 = 6");
    assert_eq!(w.macros()[0].source, "LET K1 = 5\nLET K2 = 6");
}

#[test]
fn run_writes_to_the_worksheet_and_the_session_and_is_one_undo_step() {
    let mut w = loaded();
    open(
        &mut w,
        "Fit",
        "# regression on the worksheet\nREGRESS 'y' 1 'x';\n FITS C3;\n RESIDUALS C4;\n RSQUARED K1.\nLET C5 = C3 + C4\nDESCRIBE C5\nNAME C3 'fits'\n",
    );
    assert_eq!(w.key_down(Key::F5, Mods::NONE), Outcome::None, "{}", w.status());
    assert!(w.status().starts_with("Finished"), "{}", w.status());
    assert!(constant(&w, 1) > 0.98 && constant(&w, 1) < 1.0);
    assert_eq!(col(&w, 2).len(), 8);
    assert_eq!(col(&w, 4).len(), 8);
    let y = col(&w, 1);
    for (a, b) in col(&w, 4).iter().zip(&y) {
        assert!((a - b).abs() < 1e-9, "fits + residuals reproduce y");
    }
    let out = session_text(&w);
    assert!(out.contains("Regression Analysis: y versus x"), "{out}");
    assert!(out.contains("Descriptive Statistics: C5"), "{out}");
    assert_eq!(w.top(), Top::Macro(w.active_macro().unwrap()), "the editor stays on screen");
    assert!(has(&w.render(), "Descriptive Statistics"), "the newest output shows below the editor");
    // One Edit > Undo reverses the whole run.
    assert_eq!(w.execute(Command::Undo), Outcome::None);
    assert!(col(&w, 2).is_empty());
    assert!(w.grid().workbook().worksheets[0].constants.is_empty());
    assert!(!w.grid().history().can_undo(), "the run was a single history entry");
    assert_eq!(w.execute(Command::Redo), Outcome::None);
    assert_eq!(col(&w, 2).len(), 8);
}

#[test]
fn run_recalculates_formula_columns() {
    let mut w = loaded();
    w.grid_mut().set_formula(2, Some("C1 * 10")).unwrap();
    assert_eq!(col(&w, 2)[1], 20.0);
    open(&mut w, "Edit", "LET C1(2) = 100\n");
    w.execute(Command::MacroRun);
    assert_eq!(col(&w, 2)[1], 1000.0, "the formula column follows the macro's change");
}

#[test]
fn step_pauses_between_lines_and_marks_the_next_line() {
    let mut w = loaded();
    open(&mut w, "Steps", "LET K1 = 1\nLET K1 = K1 + 1\nLET K1 = K1 * 10\n");
    assert_eq!(w.execute(Command::MacroStep), Outcome::None);
    assert!(w.macro_paused());
    assert_eq!(w.macro_editor().exec_line(), Some(2));
    assert_eq!(w.macro_editor().run_state(), RunState::Paused);
    assert_eq!(constant(&w, 1), 1.0);
    w.execute(Command::MacroStep);
    assert_eq!(constant(&w, 1), 2.0);
    assert_eq!(w.macro_editor().exec_line(), Some(3));
    // Continue with Run.
    w.execute(Command::MacroRun);
    assert_eq!(constant(&w, 1), 20.0);
    assert!(!w.macro_paused());
    assert_eq!(w.macro_editor().exec_line(), None);
    assert!(w.status().starts_with("Finished"));
}

#[test]
fn breakpoints_pause_a_run_and_stop_abandons_it() {
    let mut w = loaded();
    open(&mut w, "Break", "LET K1 = 1\nLET K2 = 2\nLET K3 = 3\n");
    w.macro_editor_mut().set_cursor(3, 1);
    w.key_down(Key::F9, Mods::NONE);
    assert!(w.macro_editor().breakpoints().contains(&3));
    w.key_down(Key::F5, Mods::NONE);
    assert!(w.macro_paused());
    assert_eq!(w.macro_editor().exec_line(), Some(3));
    assert_eq!(constant(&w, 2), 2.0);
    assert!(constant(&w, 3).is_nan());
    assert!(w.status().contains("breakpoint"), "{}", w.status());
    w.key_down(Key::F5, Mods { shift: true, ..Mods::NONE });
    assert!(!w.macro_paused());
    assert_eq!(w.macro_editor().exec_line(), None);
    assert_eq!(w.status(), "Macro stopped");
    assert!(constant(&w, 3).is_nan(), "the rest of the macro never ran");
}

#[test]
fn stopping_erases_the_temporaries_of_active_macros() {
    let mut w = loaded();
    open(
        &mut w,
        "Temps",
        "MACRO\nWork X\nMCOLUMN X T\nLET T = X * 2\nLET K1 = SUM(T)\nLET K2 = 5\nENDMACRO\n%Work C1\n",
    );
    w.macro_editor_mut().set_cursor(6, 1);
    w.key_down(Key::F9, Mods::NONE);
    w.execute(Command::MacroRun);
    assert!(w.macro_paused(), "{}", w.status());
    assert_eq!(constant(&w, 1), 72.0);
    assert!(!col(&w, 2).is_empty(), "the macro's temporary column exists while paused");
    w.execute(Command::MacroStop);
    assert!(col(&w, 2).is_empty(), "stopping erased it");
}

#[test]
fn runtime_error_marks_the_line_and_writes_to_the_session() {
    let mut w = loaded();
    open(&mut w, "Bad", "LET K1 = 1\nLET K2 = NOSUCH + 1\nLET K3 = 3\n");
    let o = w.execute(Command::MacroRun);
    assert!(matches!(&o, Outcome::Error(m) if m.contains("NOSUCH")), "{o:?}");
    assert_eq!(w.macro_editor().error().map(|e| e.0), Some(2));
    assert_eq!(constant(&w, 1), 1.0, "work before the error stays");
    assert!(constant(&w, 3).is_nan());
    assert!(session_text(&w).contains("Macro error: Bad"));
    assert!(!w.macro_paused());
    assert!(has(&w.render(), "Error on line 2"));
    // Fixing the text clears the error marker.
    w.macro_editor_mut().set_cursor(2, 1);
    w.key_down(Key::Char('#'), Mods::NONE);
    assert!(w.macro_editor().error().is_none());
}

#[test]
fn syntax_error_is_reported_before_anything_runs() {
    let mut w = loaded();
    open(&mut w, "Syn", "LET K1 = 1\nDO K2 = 1:3\nLET K3 = 1\n");
    let o = w.execute(Command::MacroRun);
    assert!(matches!(o, Outcome::Error(_)), "{o:?}");
    assert!(constant(&w, 1).is_nan(), "a script with a syntax error runs nothing");
    assert!(w.macro_editor().error().is_some());
    assert!(!w.grid().history().can_undo());
}

#[test]
fn editing_a_paused_macro_abandons_the_run() {
    let mut w = loaded();
    open(&mut w, "Edit", "LET K1 = 1\nLET K2 = 2\n");
    w.execute(Command::MacroStep);
    assert!(w.macro_paused());
    w.key_down(Key::Char('x'), Mods::NONE);
    assert!(!w.macro_paused());
    assert!(w.status().contains("abandoned"), "{}", w.status());
    assert_eq!(w.macro_editor().exec_line(), None);
}

#[test]
fn a_macro_calls_another_stored_macro_by_name() {
    let mut w = loaded();
    open(&mut w, "Helper", "MACRO\nTriple A\nMCONSTANT A\nLET A = A * 3\nENDMACRO\n");
    open(&mut w, "Main", "LET K1 = 7\n%Triple K1\n");
    w.execute(Command::MacroRun);
    assert!(w.status().starts_with("Finished"), "{}", w.status());
    assert_eq!(constant(&w, 1), 21.0);
}

#[test]
fn changing_worksheet_while_paused_is_refused() {
    let mut w = loaded();
    open(&mut w, "Sheet", "LET K1 = 1\nLET K2 = 2\n");
    w.execute(Command::MacroStep);
    w.execute(Command::NewWorksheet);
    let o = w.execute(Command::MacroStep);
    assert!(matches!(o, Outcome::Error(ref m) if m.contains("worksheet")), "{o:?}");
}

#[test]
fn running_with_no_macro_open_explains_what_to_do() {
    let mut w = loaded();
    let o = w.execute(Command::MacroRun);
    assert!(matches!(o, Outcome::Error(ref m) if m.contains("New Macro")), "{o:?}");
    assert!(matches!(w.execute(Command::MacroExport), Outcome::Error(_)));
    assert_eq!(w.execute(Command::MacroStop), Outcome::None);
}

#[test]
fn macros_are_saved_inside_the_otb_project_and_restored() {
    let mut w = loaded();
    open(&mut w, "Alpha", "LET K1 = 1\n# line two\n");
    open(&mut w, "Beta", "SET C9\n1 2 3\nEND\n");
    let bytes = w.save_project();
    assert!(
        w.grid().workbook().metadata.keys().all(|k| !k.starts_with("ui.macro.")),
        "no leftovers in the live workbook"
    );
    let mut w2 = Workspace::new(1200.0, 800.0);
    assert_eq!(w2.open_project(&bytes), Outcome::None);
    let names: Vec<&str> = w2.macros().iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["Alpha", "Beta"]);
    assert_eq!(w2.macros()[0].source, "LET K1 = 1\n# line two\n");
    assert_eq!(w2.macros()[1].source, "SET C9\n1 2 3\nEND\n");
    assert!(w2.grid().workbook().metadata.keys().all(|k| !k.starts_with("ui.macro.")));
    // The restored macro opens and runs.
    let id = w2.macros()[1].id;
    assert_eq!(w2.open_item(Target::Macro(id)), Outcome::None);
    assert_eq!(w2.macro_editor().text(), "SET C9\n1 2 3\nEND\n");
    w2.execute(Command::MacroRun);
    assert_eq!(col(&w2, 8), vec![1.0, 2.0, 3.0]);
}

#[test]
fn unsaved_editor_text_is_included_when_saving() {
    let mut w = loaded();
    w.execute(Command::MacroNew);
    w.paste_text("LET K1 = 2");
    let bytes = w.save_project();
    let mut w2 = Workspace::new(1200.0, 800.0);
    w2.open_project(&bytes);
    assert_eq!(w2.macros()[0].source, "LET K1 = 2");
}

#[test]
fn a_new_project_clears_the_macros() {
    let mut w = loaded();
    open(&mut w, "Gone", "LET K1 = 1\n");
    w.execute(Command::NewProject);
    assert!(w.macros().is_empty());
    assert_eq!(w.active_macro(), None);
    assert!(matches!(w.top(), Top::Session(_)));
}

#[test]
fn mac_files_import_as_macros_and_export_back() {
    let mut w = loaded();
    assert_eq!(w.execute(Command::MacroImport), Outcome::NeedImportFile);
    assert_eq!(w.import_data("C:\\scripts\\clean.mac", b"LET K1 = 9\r\nLET K2 = 8\r\n"), Outcome::None);
    assert_eq!(w.macros()[0].name, "clean");
    assert_eq!(w.macros()[0].source, "LET K1 = 9\nLET K2 = 8\n");
    assert_eq!(w.top(), Top::Macro(w.macros()[0].id));
    w.execute(Command::MacroRun);
    assert_eq!(constant(&w, 2), 8.0);
    match w.execute(Command::MacroExport) {
        Outcome::File(f) => {
            assert_eq!(f.name, "clean.mac");
            assert_eq!(String::from_utf8(f.bytes).unwrap(), "LET K1 = 9\nLET K2 = 8\n");
        }
        o => panic!("{o:?}"),
    }
    assert!(matches!(w.import_data("bad.mac", &[0xff, 0xfe]), Outcome::Error(_)));
    // A second file with the same name gets a distinct macro name.
    w.import_data("clean.mac", b"LET K1 = 1\n");
    assert_eq!(w.macros()[1].name, "clean 1");
}

#[test]
fn navigator_lists_opens_renames_exports_and_deletes_macros() {
    let mut w = loaded();
    open(&mut w, "First", "LET K1 = 1\n");
    open(&mut w, "Second", "LET K1 = 2\n");
    let id1 = w.macros()[0].id;
    let id2 = w.macros()[1].id;
    let rows = w.navigator().visible(&openstats_ui::workspace::NavModel {
        worksheets: vec![],
        graphs: vec![],
        sessions: vec![],
        macros: vec![(id1, "First".into()), (id2, "Second".into())],
        ..Default::default()
    });
    assert!(rows.iter().any(|r| r.1 == "Macros"));
    // Open the first: the editor shows its text.
    w.navigator_action(Target::Macro(id1), NavAction::Open);
    assert_eq!(w.macro_editor().text(), "LET K1 = 1\n");
    // Rename.
    w.navigator_action(Target::Macro(id1), NavAction::Rename);
    for _ in 0..5 {
        w.key_down(Key::Backspace, Mods::NONE);
    }
    for ch in "Clean".chars() {
        w.key_down(Key::Char(ch), Mods::NONE);
    }
    // The navigator only has the keyboard when focused.
    w.set_focus(Focus::Navigator);
    w.navigator_action(Target::Macro(id1), NavAction::Rename);
    for _ in 0..5 {
        w.key_down(Key::Backspace, Mods::NONE);
    }
    for ch in "Clean".chars() {
        w.key_down(Key::Char(ch), Mods::NONE);
    }
    w.key_down(Key::Enter, Mods::NONE);
    assert_eq!(w.macros()[0].name, "Clean");
    // A duplicate name is refused.
    w.navigator_action(Target::Macro(id2), NavAction::Rename);
    for _ in 0..6 {
        w.key_down(Key::Backspace, Mods::NONE);
    }
    for ch in "clean".chars() {
        w.key_down(Key::Char(ch), Mods::NONE);
    }
    w.key_down(Key::Enter, Mods::NONE);
    assert_eq!(w.macros()[1].name, "Second");
    assert!(w.status().contains("already exists"), "{}", w.status());
    // Export and delete.
    assert!(
        matches!(w.navigator_action(Target::Macro(id2), NavAction::Export), Outcome::File(f) if f.name == "Second.mac")
    );
    w.navigator_action(Target::Macro(id1), NavAction::Delete);
    assert_eq!(w.macros().len(), 1);
    assert_eq!(w.active_macro(), None);
    assert!(matches!(w.top(), Top::Session(_)));
    // Folder context menu creates a macro.
    w.navigator_action(Target::Folder(openstats_ui::workspace::Folder::Macros), NavAction::NewMacro);
    assert_eq!(w.macros().len(), 2);
}

#[test]
fn menus_offer_the_macro_commands_with_shortcuts() {
    let w = loaded();
    let json = w.menu().to_json(&|_| true);
    for id in [
        "file.macro_new",
        "file.macro_open",
        "file.macro_save",
        "tools.macro_editor",
        "tools.run_macro",
        "tools.step_macro",
        "tools.stop_macro",
        "tools.toggle_breakpoint",
    ] {
        assert!(json.contains(&format!("\"id\":\"{id}\"")), "{id} is in the menu");
        assert_eq!(Command::from_id(id).map(|c| c.id()).as_deref(), Some(id));
    }
    assert!(json.contains("\"id\":\"tools.run_macro\",\"shortcut\":\"F5\""));
    assert!(json.contains("\"id\":\"tools.stop_macro\",\"shortcut\":\"Shift+F5\""));
    assert!(w.menu().menus.iter().any(|m| m.label == "Tools"));
}

#[test]
fn tools_menu_runs_the_macro_through_a_click() {
    let mut w = loaded();
    open(&mut w, "Click", "LET K1 = 4\n");
    let idx = w.menu().menus.iter().position(|m| m.label == "Tools").unwrap();
    let r = w.menu().bar_rects()[idx];
    w.pointer_down(r.x + 3.0, r.y + 3.0, Button::Left, Mods::NONE, 1);
    assert!(w.menu().is_open());
    let row = w.menu().panels()[0].rows[0].0;
    w.pointer_down(row.x + 5.0, row.y + 5.0, Button::Left, Mods::NONE, 1);
    assert_eq!(constant(&w, 1), 4.0);
}

#[test]
fn editor_toolbar_and_mouse_work_inside_the_workspace() {
    let mut w = loaded();
    open(&mut w, "Mouse", "LET K1 = 1\nLET K2 = 2\n");
    // The pane origin: x = nav width + gap, y = menu bar. Click the Step button.
    let scene = w.render();
    assert!(has(&scene, "Step"));
    let g_x = 210.0 + 4.0;
    let g_y = openstats_ui::dialogs::menu::BAR_HEIGHT;
    w.pointer_down(g_x + 6.0 + 64.0 + 4.0 + 10.0, g_y + 10.0, Button::Left, Mods::NONE, 1);
    assert!(w.macro_paused());
    assert_eq!(constant(&w, 1), 1.0);
    // Click in the text: the cursor moves, focus stays with the editor.
    w.pointer_down(g_x + 60.0 + 12.0, g_y + 30.0 + 3.0, Button::Left, Mods::NONE, 1);
    w.pointer_up();
    assert_eq!(w.focus(), Focus::Session);
    assert_eq!(w.macro_editor().cursor().0, 1);
}

#[test]
fn clipboard_commands_go_to_the_editor_when_it_has_focus() {
    let mut w = loaded();
    open(&mut w, "Clip", "LET K1 = 1\nLET K2 = 2\n");
    w.key_down(Key::Char('a'), Mods::CTRL);
    match w.key_down(Key::Char('c'), Mods::CTRL) {
        Outcome::Clipboard(t) => assert_eq!(t, "LET K1 = 1\nLET K2 = 2\n"),
        o => panic!("{o:?}"),
    }
    match w.key_down(Key::Char('x'), Mods::CTRL) {
        Outcome::Clipboard(t) => assert_eq!(t, "LET K1 = 1\nLET K2 = 2\n"),
        o => panic!("{o:?}"),
    }
    assert_eq!(w.macros()[0].source, "");
    // Ctrl+Z in the editor undoes the text, not the worksheet.
    w.key_down(Key::Char('z'), Mods::CTRL);
    assert_eq!(w.macros()[0].source, "LET K1 = 1\nLET K2 = 2\n");
}

#[test]
fn panes_stay_inside_the_window_when_resized_with_the_editor_open() {
    let mut w = loaded();
    open(&mut w, "Size", "LET K1 = 1\n");
    for (width, height) in [(1200.0, 800.0), (900.0, 600.0), (700.0, 300.0), (400.0, 200.0)] {
        w.set_size(width, height);
        let scene = w.render();
        assert!(scene.width == width && scene.height == height);
        assert!(has(&scene, "Run"));
    }
}

#[test]
fn all_themes_draw_the_editor() {
    let mut w = loaded();
    open(&mut w, "Theme", "DO K1 = 1:3\n  LET K2 = K1 # c\nENDDO\n");
    for k in [
        openstats_ui::style::ThemeKind::Light,
        openstats_ui::style::ThemeKind::Dark,
        openstats_ui::style::ThemeKind::HighContrast,
    ] {
        w.set_theme_kind(k);
        assert!(has(&w.render(), "ENDDO"));
    }
}
