//! Keyboard shortcuts and the light/dark themes, driven through the whole window.

use openstats_charts::core::{Scene, Shape};
use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_ui::dialogs::{Command, Value};
use openstats_ui::grid::{Key, Mods};
use openstats_ui::style::{contrast_ratio, luminance, Chrome, ThemeKind};
use openstats_ui::workspace::{Focus, Outcome, Workspace};

fn loaded() -> Workspace {
    let mut w = Workspace::new(1200.0, 800.0);
    let mut wb = Workbook::new();
    let x: Vec<f64> = (0..20).map(f64::from).collect();
    let y: Vec<f64> = (0..20).map(|i| 3.0 + 1.5 * f64::from(i) + f64::from((i * 5) % 4)).collect();
    for c in [Column::numeric("x", x), Column::numeric("y", y)] {
        wb.worksheets[0].add_column(c).unwrap();
    }
    *w.grid_mut() = openstats_ui::grid::GridController::new(wb, 800.0, 300.0);
    w.set_size(1200.0, 800.0);
    w
}

fn ctrl(c: char) -> (Key, Mods) {
    (Key::Char(c), Mods::CTRL)
}

fn press(w: &mut Workspace, (k, m): (Key, Mods)) -> Outcome {
    w.key_down(k, m)
}

#[test]
fn file_shortcuts_raise_the_same_outcomes_as_the_menu() {
    let mut w = loaded();
    assert_eq!(press(&mut w, ctrl('o')), Outcome::NeedOpenFile);
    assert_eq!(press(&mut w, ctrl('i')), Outcome::NeedImportFile);
    assert!(matches!(press(&mut w, ctrl('s')), Outcome::File(f) if f.name.ends_with(".otb")));
    assert!(matches!(press(&mut w, ctrl('p')), Outcome::Print(_) | Outcome::Error(_)));
    // Ctrl+Shift+N adds a worksheet, Ctrl+N starts a new project.
    let before = w.grid().workbook().worksheets.len();
    press(&mut w, (Key::Char('N'), Mods::CTRL | Mods::SHIFT));
    assert_eq!(w.grid().workbook().worksheets.len(), before + 1);
    press(&mut w, ctrl('n'));
    assert_eq!(w.grid().workbook().worksheets.len(), 1);
    assert_eq!(w.grid().worksheet().n_cols(), 0, "a new project is empty");
}

#[test]
fn undo_redo_shortcuts_including_both_redo_spellings() {
    let mut w = loaded();
    w.set_focus(Focus::Grid);
    w.grid_mut().begin_edit(Some("99"), false);
    w.key_down(Key::Enter, Mods::NONE);
    let v = w.grid().worksheet().columns[0].display(0);
    assert_eq!(v, "99");
    press(&mut w, ctrl('z'));
    assert_eq!(w.grid().worksheet().columns[0].display(0), "0");
    press(&mut w, ctrl('y'));
    assert_eq!(w.grid().worksheet().columns[0].display(0), "99");
    press(&mut w, ctrl('z'));
    press(&mut w, (Key::Char('Z'), Mods::CTRL | Mods::SHIFT));
    assert_eq!(w.grid().worksheet().columns[0].display(0), "99", "Ctrl+Shift+Z also redoes");
}

#[test]
fn ctrl_f_moves_to_the_formula_bar_and_f6_cycles_panes() {
    let mut w = loaded();
    w.set_focus(Focus::Navigator);
    press(&mut w, ctrl('f'));
    assert_eq!(w.focus(), Focus::Grid);
    assert!(w.grid().editor().is_some(), "the formula bar is editing the active cell");
    w.key_down(Key::Escape, Mods::NONE);
    assert!(w.grid().editor().is_none());
    w.set_focus(Focus::Navigator);
    let mut seen = vec![];
    for _ in 0..3 {
        w.key_down(Key::F6, Mods::NONE);
        seen.push(w.focus());
    }
    assert_eq!(seen, [Focus::Session, Focus::Grid, Focus::Navigator]);
}

#[test]
fn alt_letters_and_f10_open_menus_and_escape_closes_them() {
    let mut w = loaded();
    for (c, label) in [
        ('f', "File"),
        ('e', "Edit"),
        ('d', "Data"),
        ('c', "Calc"),
        ('s', "Stat"),
        ('g', "Graph"),
        ('v', "View"),
        ('a', "Assistant"),
    ] {
        w.key_down(Key::Char(c), Mods { shift: false, ctrl: false, alt: true });
        assert!(w.menu().is_open(), "Alt+{c} opens {label}");
        assert_eq!(w.menu().open_label(), Some(label.to_string()));
        w.key_down(Key::Escape, Mods::NONE);
        assert!(!w.menu().is_open());
    }
    w.key_down(Key::F10, Mods::NONE);
    assert_eq!(w.menu().open_label(), Some("File".to_string()));
}

#[test]
fn every_shortcut_shown_in_a_menu_is_bound() {
    // Each "Ctrl+X" label in the menu bar must do something when pressed.
    let json = loaded().menu().to_json(&|_| true);
    let mut checked = 0;
    for (label, key, mods) in [
        ("Ctrl+N", 'n', Mods::CTRL),
        ("Ctrl+O", 'o', Mods::CTRL),
        ("Ctrl+S", 's', Mods::CTRL),
        ("Ctrl+P", 'p', Mods::CTRL),
        ("Ctrl+Z", 'z', Mods::CTRL),
        ("Ctrl+F", 'f', Mods::CTRL),
        ("Ctrl+I", 'i', Mods::CTRL),
        ("Ctrl+Shift+D", 'd', Mods::CTRL | Mods::SHIFT),
        ("Ctrl+Shift+N", 'n', Mods::CTRL | Mods::SHIFT),
    ] {
        assert!(json.contains(label), "menu lists {label}");
        let mut w = loaded();
        let before = w.render();
        let out = w.key_down(Key::Char(key), mods);
        let after = w.render();
        let acted = out != Outcome::None
            || before.items.len() != after.items.len()
            || w.status() != ""
            || w.focus() != Focus::Navigator
            || w.grid().editor().is_some();
        assert!(acted || label == "Ctrl+Z", "{label} does nothing");
        checked += 1;
    }
    assert_eq!(checked, 9);
}

fn fills(s: &Scene) -> Vec<openstats_charts::core::Color> {
    s.items.iter().filter(|i| matches!(i.shape, Shape::Rect(_))).filter_map(|i| i.fill).filter(|c| c.a == 255).collect()
}

#[test]
fn toggling_the_theme_recolours_every_component() {
    let mut w = loaded();
    w.execute(Command::OpenDialog("stat.regression.fit".into()));
    {
        let d = w.dialog_mut().unwrap();
        let mut v = d.values().clone();
        v.set("response", Value::Vars(vec![1]));
        v.set("preds", Value::Vars(vec![0]));
        d.set_values(v);
    }
    w.run_dialog();
    w.menu_mut().open_by_label("Stat");
    let light = w.render();
    assert_eq!(w.theme_kind(), ThemeKind::Light);
    assert_eq!(light.background, Chrome::light().desk);
    // Ctrl+Shift+D flips to dark; the dialog opened afterwards picks the dark theme up too.
    w.key_down(Key::Char('D'), Mods::CTRL | Mods::SHIFT);
    assert_eq!(w.theme_kind(), ThemeKind::Light, "an open menu has the keyboard");
    w.key_down(Key::Escape, Mods::NONE);
    w.key_down(Key::Char('D'), Mods::CTRL | Mods::SHIFT);
    assert_eq!(w.theme_kind(), ThemeKind::Dark);
    w.menu_mut().open_by_label("Stat");
    let dark = w.render();
    assert_eq!(dark.background, Chrome::dark().desk);
    assert!(luminance(dark.background) < 0.1 && luminance(light.background) > 0.5);
    // Panes, menus, the grid and the Session window must not keep light-theme surfaces.
    let bright = fills(&dark).into_iter().filter(|c| luminance(*c) > 0.45).count();
    let total = fills(&dark).len();
    assert!(bright * 20 < total.max(1), "{bright} of {total} opaque fills are still light in the dark theme");
    w.execute(Command::OpenDialog("stat.basic.descriptive".into()));
    let dialog = w.render();
    let bright = fills(&dialog).into_iter().filter(|c| luminance(*c) > 0.45).count();
    assert!(bright * 20 < fills(&dialog).len(), "dialog keeps dark surfaces ({bright} light fills)");
    w.cancel_dialog();
    // The session window text stays legible against its pane background in both themes.
    for k in [ThemeKind::Dark, ThemeKind::Light] {
        w.set_theme_kind(k);
        let t = k.theme();
        assert!(contrast_ratio(t.ink, t.background) >= 7.0, "{k:?}");
    }
    w.execute(Command::ToggleTheme);
    assert_eq!(w.theme_kind(), ThemeKind::Dark);
    assert_eq!(w.status(), "");
}

#[test]
fn graph_thumbnails_follow_the_theme() {
    let mut w = loaded();
    w.execute(Command::OpenDialog("graph.histogram".into()));
    let d = w.dialog_mut().unwrap();
    let mut v = d.values().clone();
    v.set("vars", Value::Vars(vec![0]));
    d.set_values(v);
    w.run_dialog();
    let a = w.render();
    w.set_theme_kind(ThemeKind::Dark);
    let b = w.render();
    assert_ne!(fills(&a), fills(&b));
    assert_eq!(w.graphs().len(), 1, "switching theme keeps the project's graphs");
}
