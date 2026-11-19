//! Integration tests: the grid as a host would drive it.

use openstats_charts::core::{Item, Scene, Shape};
use openstats_core::column::Column;
use openstats_core::{NumericFormat, Workbook};
use openstats_ui::grid::{CellRef, GridController, Hit, Key, Mods};
use std::time::Instant;

fn grid_with(cols: Vec<Column>) -> GridController {
    let mut wb = Workbook::new();
    for c in cols {
        wb.worksheets[0].add_column(c).unwrap();
    }
    GridController::new(wb, 800.0, 400.0)
}

fn sample() -> GridController {
    grid_with(vec![Column::numeric("x", vec![1.0, 2.0, f64::NAN, 4.0]), Column::text("name", vec!["a", "b", "c", "d"])])
}

fn texts(s: &Scene) -> Vec<String> {
    s.items
        .iter()
        .filter_map(|Item { shape, .. }| if let Shape::Text { text, .. } = shape { Some(text.clone()) } else { None })
        .collect()
}

fn type_str(g: &mut GridController, s: &str) {
    for ch in s.chars() {
        g.key_down(Key::Char(ch), Mods::NONE);
    }
}

fn num(g: &GridController, col: usize, row: usize) -> f64 {
    g.workbook().worksheets[0].columns[col].data.get(row).as_f64()
}

#[test]
fn headers_and_missing_display() {
    let mut g = sample();
    let t = texts(&g.render());
    assert!(t.contains(&"C1".to_string()) && t.contains(&"C2-T".to_string()));
    assert!(t.contains(&"x".to_string()) && t.contains(&"name".to_string()));
    assert!(t.contains(&"*".to_string()), "missing numeric shows *");
    assert!(t.contains(&"1".to_string()) && t.contains(&"4".to_string()));
    g.select_cell(CellRef::new(0, 0));
    g.set_number_format(NumericFormat::Fixed(2)).unwrap();
    let t = texts(&g.render());
    assert!(t.contains(&"1.00".to_string()));
    // The formula bar still shows full precision.
    assert_eq!(g.cell_content(CellRef::new(0, 0)), "1");
}

#[test]
fn typing_editing_and_undo() {
    let mut g = sample();
    g.select_cell(CellRef::new(2, 0));
    type_str(&mut g, "7.5");
    assert!(g.editor().is_some());
    g.key_down(Key::Enter, Mods::NONE);
    assert_eq!(num(&g, 0, 2), 7.5);
    assert_eq!(g.active(), CellRef::new(3, 0), "Enter moves down");
    g.key_down(Key::Char('z'), Mods::CTRL);
    assert!(num(&g, 0, 2).is_nan());
    g.key_down(Key::Char('y'), Mods::CTRL);
    assert_eq!(num(&g, 0, 2), 7.5);
    // Escape discards.
    g.select_cell(CellRef::new(0, 0));
    type_str(&mut g, "99");
    g.key_down(Key::Escape, Mods::NONE);
    assert_eq!(num(&g, 0, 0), 1.0);
    // Invalid text keeps the editor open and reports the error.
    type_str(&mut g, "abc");
    g.key_down(Key::Enter, Mods::NONE);
    assert!(g.editor().is_some() && !g.status().is_empty());
    assert_eq!(num(&g, 0, 0), 1.0);
    g.key_down(Key::Escape, Mods::NONE);
}

#[test]
fn star_and_blank_enter_missing() {
    let mut g = sample();
    g.select_cell(CellRef::new(0, 0));
    type_str(&mut g, "*");
    g.key_down(Key::Enter, Mods::NONE);
    assert!(num(&g, 0, 0).is_nan());
    g.select_cell(CellRef::new(1, 0));
    g.key_down(Key::Delete, Mods::NONE);
    assert!(num(&g, 0, 1).is_nan());
    assert_eq!(g.history().undo_len(), 2);
}

#[test]
fn typing_into_empty_columns_adopts_a_kind() {
    let mut g = sample();
    g.select_cell(CellRef::new(0, 4));
    type_str(&mut g, "12");
    g.key_down(Key::Enter, Mods::NONE);
    let ws = &g.workbook().worksheets[0];
    assert_eq!(ws.n_cols(), 5);
    assert_eq!(ws.column_label(4), "C5");
    assert_eq!(num(&g, 4, 0), 12.0);
    g.select_cell(CellRef::new(0, 5));
    type_str(&mut g, "2024-03-05");
    g.key_down(Key::Enter, Mods::NONE);
    assert_eq!(g.workbook().worksheets[0].column_label(5), "C6-D");
    g.select_cell(CellRef::new(0, 6));
    type_str(&mut g, "hello");
    g.key_down(Key::Enter, Mods::NONE);
    assert_eq!(g.workbook().worksheets[0].column_label(6), "C7-T");
    // One undo removes the typed cell and the columns it created.
    g.undo().unwrap();
    assert_eq!(g.workbook().worksheets[0].n_cols(), 6);
}

#[test]
fn formula_bar_and_formula_columns() {
    let mut g = sample();
    assert_eq!(g.formula_bar_text(), "1");
    assert_eq!(g.name_box(), "C1[1]");
    g.select_cell(CellRef::new(0, 2));
    g.begin_edit(None, true);
    g.editor_mut().unwrap().insert("='x' * 10");
    assert_eq!(g.formula_bar_text(), "='x' * 10");
    g.commit_edit().unwrap();
    assert_eq!(num(&g, 2, 0), 10.0);
    assert_eq!(num(&g, 2, 1), 20.0);
    assert!(num(&g, 2, 2).is_nan());
    assert!(g.cell_content(CellRef::new(1, 2)).starts_with('='));
    // Source edits propagate; calculated cells reject direct edits.
    g.select_cell(CellRef::new(0, 0));
    type_str(&mut g, "5");
    g.key_down(Key::Enter, Mods::NONE);
    assert_eq!(num(&g, 2, 0), 50.0);
    g.select_cell(CellRef::new(0, 2));
    type_str(&mut g, "3");
    g.key_down(Key::Enter, Mods::NONE);
    assert!(g.editor().is_some() && g.status().contains("formula"));
    g.key_down(Key::Escape, Mods::NONE);
    assert_eq!(num(&g, 2, 0), 50.0);
    // Undo reverts the recalculation together with the edit.
    g.undo().unwrap();
    assert_eq!(num(&g, 2, 0), 10.0);
    // A bad formula is rejected and leaves the sheet alone.
    assert!(g.set_formula(2, Some("'nope' +")).is_err());
    assert_eq!(num(&g, 2, 0), 10.0);
}

#[test]
fn selection_clipboard_roundtrip() {
    let mut g = sample();
    g.select_cell(CellRef::new(0, 0));
    g.key_down(Key::Down, Mods::SHIFT);
    g.key_down(Key::Right, Mods::SHIFT);
    g.key_down(Key::Down, Mods::SHIFT);
    assert_eq!(g.selection().current().n_cells(), 6);
    let t = g.copy().unwrap();
    assert_eq!(t, "1\ta\n2\tb\n*\tc\n");
    g.select_cell(CellRef::new(0, 3));
    let n = g.paste_text(&t).unwrap();
    assert_eq!(n, 6);
    assert_eq!(num(&g, 3, 1), 2.0);
    assert!(num(&g, 3, 2).is_nan());
    assert_eq!(g.workbook().worksheets[0].column_label(4), "C5-T");
    g.undo().unwrap();
    assert_eq!(g.workbook().worksheets[0].n_cols(), 2);
    // Paste of a single value fills the selection; text into a numeric column fails atomically.
    g.select_cell(CellRef::new(0, 0));
    g.key_down(Key::Down, Mods::SHIFT);
    g.paste_text("9").unwrap();
    assert_eq!((num(&g, 0, 0), num(&g, 0, 1)), (9.0, 9.0));
    assert!(g.paste_text("oops\n5").is_err());
    assert_eq!((num(&g, 0, 0), num(&g, 0, 1)), (9.0, 9.0));
    // Cut clears.
    g.key_down(Key::Char('x'), Mods::CTRL);
    assert!(num(&g, 0, 0).is_nan() && num(&g, 0, 1).is_nan());
    g.key_down(Key::Char('z'), Mods::CTRL);
    assert_eq!(num(&g, 0, 0), 9.0);
    g.key_down(Key::Char('v'), Mods::CTRL);
    assert_eq!(g.clipboard_text(), Some("9\n9\n"));
}

#[test]
fn keyboard_navigation_and_jumps() {
    let mut g = grid_with(vec![Column::numeric("a", vec![1.0, 2.0, f64::NAN, 4.0, 5.0])]);
    g.select_cell(CellRef::new(0, 0));
    g.key_down(Key::Down, Mods::CTRL);
    assert_eq!(g.active().row, 1, "end of the block");
    g.key_down(Key::Down, Mods::CTRL);
    assert_eq!(g.active().row, 3, "next filled cell");
    g.key_down(Key::Down, Mods::CTRL);
    assert_eq!(g.active().row, 4);
    g.key_down(Key::Down, Mods::CTRL);
    assert_eq!(g.active().row, g.display_rows() - 1, "edge of the sheet");
    g.key_down(Key::Home, Mods::CTRL);
    assert_eq!(g.active(), CellRef::new(0, 0));
    g.key_down(Key::End, Mods::CTRL);
    assert_eq!(g.active(), CellRef::new(4, 0));
    g.key_down(Key::Up, Mods::NONE);
    g.key_down(Key::Left, Mods::NONE);
    assert_eq!(g.active(), CellRef::new(3, 0));
    g.key_down(Key::PageDown, Mods::NONE);
    assert!(g.active().row > 10);
    assert!(g.is_visible(g.active()));
    g.key_down(Key::Char('a'), Mods::CTRL);
    assert!(g.selection().contains(CellRef::new(5000, 3)));
    // Tab moves right and Shift+Enter moves up.
    g.select_cell(CellRef::new(2, 0));
    g.key_down(Key::Tab, Mods::NONE);
    assert_eq!(g.active(), CellRef::new(2, 1));
    g.key_down(Key::Enter, Mods::SHIFT);
    assert_eq!(g.active(), CellRef::new(1, 1));
}

#[test]
fn pointer_selection_resize_and_headers() {
    let mut g = sample();
    let l = g.layout();
    assert!(matches!(g.hit_test(2.0, 2.0), Hit::Corner));
    let cell = g.cell_rect(CellRef::new(1, 1));
    assert_eq!(g.hit_test(cell.x + 5.0, cell.y + 5.0), Hit::Cell(CellRef::new(1, 1)));
    assert_eq!(g.hit_test(cell.x + 5.0, 10.0), Hit::ColumnHeader(1));
    assert_eq!(g.hit_test(3.0, cell.y + 5.0), Hit::RowHeader(1));
    // Drag-select.
    g.pointer_down(l.gutter + 5.0, l.header + 5.0, Mods::NONE, 1);
    g.pointer_move(cell.x + 5.0, cell.y + 5.0);
    g.pointer_up();
    assert_eq!(g.selection().current().n_cells(), 4);
    // Column header selects the column; shift extends.
    g.pointer_down(cell.x + 5.0, 10.0, Mods::NONE, 1);
    g.pointer_up();
    assert!(g.selection().touches_column(1) && !g.selection().touches_column(0));
    // Resize by dragging the edge.
    let edge = cell.right();
    assert_eq!(g.hit_test(edge, 10.0), Hit::ColumnEdge(1));
    let w0 = g.column_width(1);
    g.pointer_down(edge, 10.0, Mods::NONE, 1);
    g.pointer_move(edge + 40.0, 10.0);
    g.pointer_up();
    assert_eq!(g.column_width(1), w0 + 40.0);
    g.pointer_down(edge + 40.0, 10.0, Mods::NONE, 2);
    assert!(g.column_width(1) < w0 + 40.0, "double click autosizes");
    // Double click renames.
    let cell = g.cell_rect(CellRef::new(0, 0));
    g.pointer_down(cell.x + 5.0, 10.0, Mods::NONE, 2);
    assert_eq!(g.editor().unwrap().buffer, "x");
    g.editor_mut().unwrap().insert("yz");
    g.commit_edit().unwrap();
    assert_eq!(g.workbook().worksheets[0].columns[0].name(), "xyz");
    // Double-click a cell edits it.
    let cell = g.cell_rect(CellRef::new(0, 0));
    g.pointer_down(cell.x + 5.0, cell.y + 5.0, Mods::NONE, 2);
    assert_eq!(g.editor().unwrap().buffer, "1");
    g.cancel_edit();
    // Ctrl-click adds a range.
    g.pointer_down(cell.x + 5.0, cell.y + 5.0, Mods::NONE, 1);
    let c2 = g.cell_rect(CellRef::new(3, 1));
    g.pointer_down(c2.x + 5.0, c2.y + 5.0, Mods::CTRL, 1);
    g.pointer_up();
    assert_eq!(g.selection().ranges().len(), 2);
}

#[test]
fn rows_formats_and_sheet_switch() {
    let mut g = sample();
    g.select_cell(CellRef::new(1, 0));
    g.insert_rows().unwrap();
    assert_eq!(g.workbook().worksheets[0].n_rows(), 5);
    assert!(num(&g, 0, 1).is_nan());
    g.delete_rows().unwrap();
    assert_eq!(g.workbook().worksheets[0].n_rows(), 4);
    assert_eq!(num(&g, 0, 1), 2.0);
    g.select_cell(CellRef::new(0, 0));
    g.change_decimals(1).unwrap();
    assert!(texts(&g.render()).contains(&"1.00".to_string()) || texts(&g.render()).contains(&"1.0".to_string()));
    g.set_number_format(NumericFormat::Scientific(1)).unwrap();
    assert!(texts(&g.render()).iter().any(|t| t.contains('E')));
    assert!(g.set_sheet(3).is_err());
    g.set_sheet(0).unwrap();
}

#[test]
fn datetime_cells_display_and_format() {
    let mut g = grid_with(vec![Column::datetime(
        "when",
        vec![openstats_core::datetime::parse("2024-03-05 10:30:00").unwrap(), i64::MIN],
    )]);
    g.autosize_column(0);
    let t = texts(&g.render());
    assert!(t.contains(&"C1-D".to_string()));
    assert!(t.iter().any(|s| s.starts_with("2024")) || t.iter().any(|s| s.contains("2024")), "{t:?}");
    assert!(t.contains(&"*".to_string()));
    g.select_cell(CellRef::new(0, 0));
    g.set_datetime_format(Some("yyyy")).unwrap();
    assert!(texts(&g.render()).contains(&"2024".to_string()));
    g.enter_cell(CellRef::new(1, 0), "2020-01-02 00:00:00").unwrap();
    assert!(!g.workbook().worksheets[0].columns[0].data.is_missing(1));
}

#[test]
fn million_rows_virtualized() {
    let n = 1_000_000;
    let v: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let t: Vec<String> = (0..n).map(|i| format!("row {i}")).collect();
    let mut g = grid_with(vec![Column::numeric("v", v), Column::text("t", t)]);
    g.set_viewport(1200.0, 800.0);
    assert!(g.display_rows() >= n);
    let first = g.render();
    let bound = first.items.len();
    assert!(bound < 4000, "primitives for one frame: {bound}");
    // Scroll to the bottom, the middle and back: each frame draws only the window.
    let (_, max_y) = g.max_scroll();
    assert!(max_y > 20_000_000.0);
    let mut worst = 0.0_f64;
    for k in 0..240 {
        let y = max_y * (f64::from(k % 5) / 4.0) + f64::from(k) * 3.0;
        g.scroll_to(0.0, y);
        let start = Instant::now();
        let s = g.render();
        worst = worst.max(start.elapsed().as_secs_f64());
        assert!(s.items.len() < 4000);
    }
    g.scroll_to_row(n - 5);
    let last = texts(&g.render());
    assert!(last.contains(&"1000000".to_string()), "{last:?}");
    assert!(last.contains(&"row 999999".to_string()));
    // Wheel scrolling accumulates and clamps.
    g.scroll_to(0.0, 0.0);
    g.wheel(0.0, 22.0 * 500_000.0);
    assert_eq!(g.visible_rows().0, 500_000);
    g.wheel(0.0, -1e12);
    assert_eq!(g.visible_rows().0, 0);
    // Keyboard navigation to the end of a million-row column is immediate.
    g.select_cell(CellRef::new(0, 0));
    g.key_down(Key::End, Mods::CTRL);
    assert_eq!(g.active(), CellRef::new(n - 1, 1));
    assert!(g.is_visible(g.active()));
    // The 60 fps budget (16.7 ms) is generous in debug builds on any machine we test on.
    assert!(worst < 0.1, "worst frame {worst}s");
}

#[test]
fn bulk_paste_is_one_undo_step() {
    let mut g = grid_with(vec![Column::numeric("a", vec![])]);
    let text: String = (0..5000).map(|i| format!("{i}\n")).collect();
    assert_eq!(g.paste_text(&text).unwrap(), 5000);
    assert_eq!(g.workbook().worksheets[0].n_rows(), 5000);
    assert_eq!(num(&g, 0, 4999), 4999.0);
    g.undo().unwrap();
    assert_eq!(g.workbook().worksheets[0].n_rows(), 0);
    g.redo().unwrap();
    assert_eq!(g.workbook().worksheets[0].n_rows(), 5000);
}

#[test]
fn scrollbars_follow_scroll_state() {
    let mut g = grid_with(vec![Column::numeric("a", (0..100_000).map(f64::from).collect())]);
    let (v, h) = g.scrollbar_thumbs();
    assert!(v.is_some());
    assert!(h.is_some(), "20 blank columns are wider than the window");
    let t0 = v.unwrap().y;
    g.scroll_to(0.0, g.max_scroll().1);
    let t1 = g.scrollbar_thumbs().0.unwrap();
    assert!(t1.y > t0 && t1.bottom() <= g.layout().body.bottom() + 1e-9);
    // Dragging the thumb scrolls.
    g.scroll_to(0.0, 0.0);
    let t = g.scrollbar_thumbs().0.unwrap();
    g.pointer_down(t.x + 2.0, t.y + 2.0, Mods::NONE, 1);
    g.pointer_move(t.x + 2.0, t.y + 102.0);
    g.pointer_up();
    assert!(g.view().scroll_y > 0.0);
}

fn editing(text: &str) -> GridController {
    let mut g = grid_with(vec![Column::text("t", vec!["x"])]);
    g.begin_edit(Some(text), false);
    g
}

fn buf(g: &GridController) -> String {
    g.editor().unwrap().buffer.clone()
}

#[test]
fn editor_selection_word_moves_and_replacement() {
    let mut g = editing("alpha beta_2 gamma");
    g.key_down(Key::Home, Mods::NONE);
    g.key_down(Key::Right, Mods::SHIFT);
    g.key_down(Key::Right, Mods::SHIFT);
    assert_eq!(g.editor().unwrap().selected_text().as_deref(), Some("al"));
    g.key_down(Key::Left, Mods::SHIFT);
    assert_eq!(g.editor().unwrap().selected_text().as_deref(), Some("a"));
    g.key_down(Key::Right, Mods::NONE);
    assert!(g.editor().unwrap().selection().is_none(), "an arrow collapses the selection");
    assert_eq!(g.editor().unwrap().cursor, 1, "collapse goes to the right edge");
    g.key_down(Key::Right, Mods::CTRL);
    assert_eq!(g.editor().unwrap().cursor, 5, "to the end of the word");
    g.key_down(Key::Right, Mods::CTRL);
    assert_eq!(g.editor().unwrap().cursor, 12, "underscore and digits belong to the word");
    g.key_down(Key::Left, Mods::CTRL | Mods::SHIFT);
    assert_eq!(g.editor().unwrap().selected_text().as_deref(), Some("beta_2"));
    g.key_down(Key::Char('Z'), Mods::SHIFT);
    assert_eq!(buf(&g), "alpha Z gamma", "typing replaces the selection");
    g.key_down(Key::Char('a'), Mods::CTRL);
    assert_eq!(g.editor().unwrap().selected_text().as_deref(), Some("alpha Z gamma"));
    g.key_down(Key::Backspace, Mods::NONE);
    assert_eq!(buf(&g), "");
    assert!(g.editor().is_some(), "still editing");
}

#[test]
fn editor_copy_cut_paste_and_multibyte_text() {
    let mut g = editing("héllo wörld ✓");
    g.key_down(Key::Home, Mods::NONE);
    g.key_down(Key::Right, Mods::CTRL | Mods::SHIFT);
    assert_eq!(g.editor_copy().as_deref(), Some("héllo"));
    assert_eq!(g.editor_cut().as_deref(), Some("héllo"));
    assert_eq!(buf(&g), " wörld ✓");
    g.key_down(Key::End, Mods::NONE);
    g.paste_text("a\tb\r\nc\n").unwrap();
    assert_eq!(buf(&g), " wörld ✓a b c", "pasting while editing is one line");
    assert_eq!(g.editor_copy(), None, "nothing selected, nothing to copy");
    g.key_down(Key::Left, Mods::CTRL);
    g.key_down(Key::Delete, Mods::NONE);
    assert_eq!(buf(&g), " wörld ✓a b ", "no panic or split characters on multibyte boundaries");
    let sel_before = g.editor().unwrap().cursor;
    g.key_down(Key::Home, Mods::SHIFT);
    g.key_down(Key::Delete, Mods::NONE);
    assert_eq!(buf(&g), "", "{sel_before}");
    assert_eq!(g.worksheet().columns[0].display(0), "x", "nothing committed until Enter");
}

#[test]
fn input_method_composition_commit_and_cancel() {
    let mut g = grid_with(vec![Column::text("t", vec!["x"])]);
    // The first committed text starts an edit, as typing would.
    g.ime_commit("日本");
    assert_eq!(buf(&g), "日本");
    assert_eq!(g.editor().unwrap().cursor, 2);
    g.ime_update("ご", 1);
    assert_eq!(g.formula_bar_text(), "日本ご", "the preedit is shown at the caret");
    assert_eq!(buf(&g), "日本", "but is not part of the text yet");
    assert_eq!(g.editor().unwrap().display_cursor(), 3);
    let scene = g.render();
    let all: Vec<String> = scene
        .items
        .iter()
        .filter_map(|i| if let Shape::Text { text, .. } = &i.shape { Some(text.clone()) } else { None })
        .collect();
    assert!(all.iter().any(|t| t == "日本ご"));
    g.ime_cancel();
    assert_eq!(g.formula_bar_text(), "日本");
    g.ime_update("ご", 1);
    g.ime_commit("語");
    assert_eq!(buf(&g), "日本語");
    assert!(g.editor().unwrap().composition.is_none());
    // a commit replaces a selection
    g.key_down(Key::Char('a'), Mods::CTRL);
    g.ime_commit("字");
    assert_eq!(buf(&g), "字");
    let (x, y, h) = g.ime_anchor().expect("candidate window position while editing");
    assert!(x > 0.0 && y >= 0.0 && h > 0.0);
    g.key_down(Key::Enter, Mods::NONE);
    assert_eq!(g.worksheet().columns[0].display(0), "字");
    assert!(g.ime_anchor().is_none());
}

#[test]
fn clicking_and_dragging_in_the_editor_moves_the_caret_and_selects() {
    let mut g = grid_with(vec![Column::text("t", vec!["hello world"])]);
    g.select_cell(CellRef { row: 0, col: 0 });
    g.begin_edit(None, false);
    let r = g.cell_rect(CellRef { row: 0, col: 0 });
    let (ox, oy) = (0.0, 0.0);
    let size = g.metrics().font_size;
    let wx = |n: usize| {
        openstats_charts::core::font::text_width(&"hello world".chars().take(n).collect::<String>(), size, false)
    };
    let y = r.y + oy + r.h / 2.0;
    g.pointer_down(r.x + ox + 4.0 + wx(5), y, Mods::NONE, 1);
    assert_eq!(g.editor().unwrap().cursor, 5);
    g.pointer_move(r.x + ox + 4.0 + wx(11), y);
    g.pointer_up();
    assert_eq!(g.editor().unwrap().selected_text().as_deref(), Some(" world"));
    g.pointer_down(r.x + ox + 4.0 + wx(2), y, Mods::NONE, 2);
    g.pointer_up();
    assert_eq!(g.editor().unwrap().selected_text().as_deref(), Some("hello"), "double click selects the word");
    g.pointer_down(r.x + ox + 4.0 + wx(8), y, Mods::SHIFT, 1);
    g.pointer_up();
    assert_eq!(g.editor().unwrap().selected_text().as_deref(), Some("hello wo"), "shift-click extends from the anchor");
}
