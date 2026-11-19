//! Phase 6a: graphs that follow their data, brushing between graphs and the worksheet, graph
//! formatting and notes, and persistence of all of it.

use openstats_charts::core::{Scene, Shape};
use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_ui::dialogs::{Command, Value};
use openstats_ui::grid::{CellRef, Mods};
use openstats_ui::workspace::{Outcome, Target, Top, Workspace};
use std::collections::BTreeSet;

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

fn pick(w: &mut Workspace, field: &str, cols: &[usize]) {
    let d = w.dialog_mut().expect("dialog open");
    let mut v = d.values().clone();
    v.set(field, Value::Vars(cols.to_vec()));
    d.set_values(v);
}

fn set(w: &mut Workspace, field: &str, val: Value) {
    let d = w.dialog_mut().expect("dialog open");
    let mut v = d.values().clone();
    v.set(field, val);
    d.set_values(v);
}

fn scatter(w: &mut Workspace) {
    w.execute(Command::OpenDialog("graph.scatter".into()));
    pick(w, "y", &[1]);
    pick(w, "x", &[0]);
    assert_eq!(w.run_dialog(), Outcome::None);
}

fn texts(s: &Scene) -> Vec<String> {
    s.items
        .iter()
        .filter_map(|i| if let Shape::Text { text, .. } = &i.shape { Some(text.clone()) } else { None })
        .collect()
}

#[test]
fn a_graph_remembers_its_recipe_and_rows() {
    let mut w = loaded();
    scatter(&mut w);
    let g = &w.graphs()[0];
    let r = g.recipe.as_ref().expect("scatter is linkable");
    assert_eq!(r.dialog, "graph.scatter");
    assert_eq!(r.sheet, 0);
    assert_eq!(g.rows, (0..30).collect::<Vec<_>>());
    assert!(g.auto);
}

#[test]
fn graphs_follow_changes_of_the_data() {
    let mut w = loaded();
    scatter(&mut w);
    let before = w.graphs()[0].spec.series[0].y.clone();
    // Typing into the worksheet goes through the workspace, which redraws the graph.
    w.grid_mut().select_cell(CellRef { row: 4, col: 1 });
    w.paste_text("500");
    let after = &w.graphs()[0].spec.series[0].y;
    assert_eq!(after[4], 500.0, "the point moved");
    assert_ne!(&before, after);
    // The copy in the Session window moved too.
    let sess = w.sessions()[0].report.render_text();
    assert!(sess.contains("Scatterplot"));
    let openstats_stats::report::Block::Chart(c) = w.sessions()[0]
        .report
        .sections
        .iter()
        .flat_map(|s| s.blocks.iter())
        .find(|b| matches!(b, openstats_stats::report::Block::Chart(_)))
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(c.series[0].y[4], 500.0);
}

#[test]
fn frozen_graphs_keep_their_data_until_updated() {
    let mut w = loaded();
    scatter(&mut w);
    w.execute(Command::OpenDialog("graph.update".into()));
    set(&mut w, "action", Value::Choice(1));
    assert_eq!(w.run_dialog(), Outcome::None);
    assert!(!w.graphs()[0].auto);
    w.grid_mut().enter_cell(CellRef { row: 0, col: 1 }, "777").unwrap();
    assert!(w.refresh_graphs().is_empty(), "a frozen graph is not redrawn");
    assert_ne!(w.graphs()[0].spec.series[0].y[0], 777.0);
    w.execute(Command::OpenDialog("graph.update".into()));
    set(&mut w, "action", Value::Choice(0));
    assert_eq!(w.run_dialog(), Outcome::None);
    assert_eq!(w.graphs()[0].spec.series[0].y[0], 777.0, "Update now redraws a frozen graph");
    assert!(!w.graphs()[0].auto);
    w.execute(Command::OpenDialog("graph.update".into()));
    set(&mut w, "action", Value::Choice(2));
    w.run_dialog();
    assert!(w.graphs()[0].auto);
}

#[test]
fn missing_values_change_the_row_map() {
    let mut w = loaded();
    scatter(&mut w);
    w.grid_mut().enter_cell(CellRef { row: 3, col: 0 }, "*").unwrap();
    let names = w.refresh_graphs();
    assert_eq!(names.len(), 1);
    let g = &w.graphs()[0];
    assert_eq!(g.spec.series[0].x.len(), 29);
    assert_eq!(g.rows.len(), 29);
    assert!(!g.rows.contains(&3));
}

#[test]
fn rubber_band_brushing_highlights_worksheet_rows_and_other_graphs() {
    let mut w = loaded();
    scatter(&mut w);
    // A second graph of the same data.
    w.execute(Command::OpenDialog("graph.scatter".into()));
    pick(&mut w, "y", &[0]);
    pick(&mut w, "x", &[1]);
    w.run_dialog();
    let id = w.graphs()[0].id;
    w.open_item(Target::Graph(id));
    assert!(matches!(w.top(), Top::Graph(g) if g == id));
    // Window coordinates of the points come from the rendered scene.
    let scene = w.render();
    let pts: Vec<_> = scene.targets.iter().filter(|t| t.series == 0).collect();
    assert!(pts.len() >= 30);
    let pick_point = |i: usize| {
        let t = pts.iter().find(|t| t.index == i).unwrap();
        match &t.region {
            openstats_charts::core::hit::Region::Point(p, _) => *p,
            openstats_charts::core::hit::Region::Rect(r) => r.center(),
        }
    };
    // Drag a box around points 10..=12 (they are on a rising line).
    let (a, b) = (pick_point(10), pick_point(12));
    let top = w.upper_pane();
    assert!(top.contains(a) && top.contains(b));
    w.pointer_down(a.x - 6.0, b.y - 6.0, openstats_ui::workspace::Button::Left, Mods::NONE, 1);
    w.pointer_move(b.x + 6.0, a.y + 6.0);
    w.pointer_up();
    let (sheet, rows) = w.brushed();
    assert_eq!(sheet, 0);
    assert!(rows.contains(&10) && rows.contains(&11) && rows.contains(&12), "{rows:?}");
    assert!(rows.len() <= 5, "{rows:?}");
    assert_eq!(w.grid().highlight(), rows, "the worksheet highlights the brushed rows");
    assert!(w.status().contains("brushed"));
    // Shift-click adds a point; a plain click on empty space clears.
    let p20 = pick_point(20);
    w.pointer_down(
        p20.x,
        p20.y,
        openstats_ui::workspace::Button::Left,
        Mods { shift: true, ctrl: false, alt: false },
        1,
    );
    w.pointer_up();
    assert!(w.brushed().1.contains(&20) && w.brushed().1.contains(&11));
    w.pointer_down(top.x + 2.0, top.y + 2.0, openstats_ui::workspace::Button::Left, Mods::NONE, 1);
    w.pointer_up();
    assert!(w.brushed().1.is_empty());
    assert!(w.grid().highlight().is_empty());
}

#[test]
fn brushed_rows_can_be_stored_and_cleared() {
    let mut w = loaded();
    scatter(&mut w);
    w.set_brush(0, BTreeSet::from([2, 5]));
    w.execute(Command::OpenDialog("graph.brush".into()));
    set(&mut w, "action", Value::Choice(1));
    assert_eq!(w.run_dialog(), Outcome::None);
    let ws = w.grid().worksheet();
    assert_eq!(ws.n_cols(), 3);
    let col = ws.columns[2].data.to_f64_vec();
    assert_eq!(col.iter().filter(|v| **v == 1.0).count(), 2);
    assert_eq!((col[2], col[5], col[0]), (1.0, 1.0, 0.0));
    w.execute(Command::OpenDialog("graph.brush".into()));
    set(&mut w, "action", Value::Choice(0));
    w.run_dialog();
    assert!(w.brushed().1.is_empty());
    // Nothing brushed: storing is refused with a message.
    w.execute(Command::OpenDialog("graph.brush".into()));
    set(&mut w, "action", Value::Choice(1));
    assert!(matches!(w.run_dialog(), Outcome::Error(_)));
}

#[test]
fn format_and_notes_change_the_drawn_graph_and_survive_saving() {
    let mut w = loaded();
    scatter(&mut w);
    let plain = w.render();
    w.execute(Command::OpenDialog("graph.format".into()));
    set(&mut w, "font", Value::Num(1.5));
    set(&mut w, "grid", Value::Choice(2));
    set(&mut w, "xmin", Value::Num(0.0));
    set(&mut w, "xmax", Value::Num(10.0));
    set(&mut w, "colors", Value::Text("#ff0000".into()));
    assert_eq!(w.run_dialog(), Outcome::None);
    w.execute(Command::OpenDialog("graph.annotate".into()));
    set(&mut w, "x", Value::Num(5.0));
    set(&mut w, "y", Value::Num(10.0));
    set(&mut w, "note", Value::Text("look here".into()));
    assert_eq!(w.run_dialog(), Outcome::None);
    let id = w.graphs()[0].id;
    w.open_item(Target::Graph(id));
    let styled = w.render();
    assert!(texts(&styled).iter().any(|t| t == "look here"), "the note is drawn");
    assert_ne!(plain.items.len(), styled.items.len());
    let st = &w.graphs()[0].style;
    assert_eq!(st.x_range, Some((0.0, 10.0)));
    assert_eq!(st.colors, vec![Some(0xff0000)]);
    // Save and reopen.
    let bytes = w.save_project();
    let mut w2 = Workspace::new(1200.0, 800.0);
    assert_eq!(w2.open_project(&bytes), Outcome::None);
    let g = &w2.graphs()[0];
    assert_eq!(g.style, w.graphs()[0].style);
    assert_eq!(g.recipe, w.graphs()[0].recipe);
    assert_eq!(g.rows.len(), 30, "the row map is rebuilt on open");
    // The reopened graph still follows the data.
    w2.grid_mut().enter_cell(CellRef { row: 1, col: 1 }, "-42").unwrap();
    assert_eq!(w2.refresh_graphs().len(), 1);
    assert_eq!(w2.graphs()[0].spec.series[0].y[1], -42.0);
}

#[test]
fn format_rejects_bad_input() {
    let mut w = loaded();
    scatter(&mut w);
    w.execute(Command::OpenDialog("graph.format".into()));
    set(&mut w, "xmin", Value::Num(5.0));
    assert!(matches!(w.run_dialog(), Outcome::Error(_)), "one limit alone is refused");
    set(&mut w, "xmax", Value::Num(1.0));
    assert!(matches!(w.run_dialog(), Outcome::Error(_)), "min above max is refused");
    set(&mut w, "xmax", Value::Num(9.0));
    set(&mut w, "colors", Value::Text("red".into()));
    assert!(matches!(w.run_dialog(), Outcome::Error(_)), "colour names are refused");
}

#[test]
fn user_titles_survive_a_redraw() {
    let mut w = loaded();
    scatter(&mut w);
    w.execute(Command::OpenDialog("graph.edit".into()));
    set(&mut w, "title", Value::Text("My title".into()));
    w.run_dialog();
    w.grid_mut().enter_cell(CellRef { row: 0, col: 1 }, "9").unwrap();
    w.refresh_graphs();
    assert_eq!(w.graphs()[0].spec.title, "My title");
}

#[test]
fn analyses_that_store_results_are_not_linked() {
    let mut w = loaded();
    w.execute(Command::OpenDialog("stat.regression.fit".into()));
    pick(&mut w, "response", &[1]);
    pick(&mut w, "preds", &[0]);
    w.run_dialog();
    assert!(w.graphs().iter().all(|g| g.recipe.is_none()), "only plain graph dialogs are rebuilt");
}

#[test]
fn brushing_also_marks_the_points_of_other_graphs() {
    let mut w = loaded();
    scatter(&mut w);
    w.execute(Command::OpenDialog("graph.scatter".into()));
    pick(&mut w, "y", &[0]);
    pick(&mut w, "x", &[1]);
    w.run_dialog();
    let second = w.graphs()[1].id;
    w.open_item(Target::Graph(second));
    let circles =
        |s: &Scene| s.items.iter().filter(|i| matches!(i.shape, Shape::Marker { size, .. } if size == 9.0)).count();
    let plain = circles(&w.render());
    w.set_brush(0, BTreeSet::from([1, 2, 3]));
    assert_eq!(circles(&w.render()), plain + 3, "one ring per brushed point");
    w.set_brush(0, BTreeSet::new());
    assert_eq!(circles(&w.render()), plain);
}

#[test]
fn point_to_row_maps_match_the_charts_for_every_brushable_graph() {
    use openstats_ui::dialogs::{Registry, Values};
    let mut wb = Workbook::new();
    let n = 20;
    let mut x: Vec<f64> = (0..n).map(f64::from).collect();
    let y: Vec<f64> = (0..n).map(|i| f64::from(i * i % 7)).collect();
    let mut size: Vec<f64> = (0..n).map(|i| 1.0 + f64::from(i % 4)).collect();
    x[2] = f64::NAN;
    size[5] = 0.0;
    let mut y2 = y.clone();
    y2[9] = f64::NAN;
    for (name, d) in [("x", x), ("y", y), ("size", size), ("y2", y2)] {
        wb.worksheets[0].add_column(Column::numeric(name, d)).unwrap();
    }
    let ws = &wb.worksheets[0];
    let reg = Registry::standard();
    let defaults = |id: &str| -> Values { reg.get(id).unwrap().spec.defaults() };
    let check = |id: &str, v: &Values| {
        let charts = reg.charts(id, ws, v).expect("linkable").expect("runs");
        let rows = reg.brush_rows(id, ws, v).expect("brushable");
        assert_eq!(charts.len(), rows.len(), "{id}");
        for (c, r) in charts.iter().zip(&rows) {
            assert_eq!(c.series[0].y.len(), r.len(), "{id}: one row per point");
        }
        rows
    };
    let mut v = defaults("graph.scatter");
    v.set("y", Value::Vars(vec![1]));
    v.set("x", Value::Vars(vec![0]));
    let rows = check("graph.scatter", &v);
    assert!(!rows[0].contains(&2), "the row with a missing x is not drawn");
    check("graph.marginal", &v);
    let mut v = defaults("graph.bubble");
    v.set("y", Value::Vars(vec![1]));
    v.set("x", Value::Vars(vec![0]));
    v.set("size", Value::Vars(vec![2]));
    let rows = check("graph.bubble", &v);
    assert!(!rows[0].contains(&5) && !rows[0].contains(&2));
    let mut v = defaults("graph.timeseries");
    v.set("vars", Value::Vars(vec![1, 3]));
    let rows = check("graph.timeseries", &v);
    assert_eq!(rows.len(), 2);
    // Not linkable: dialogs the workspace owns and analyses that store results.
    assert!(!reg.is_linkable("graph.edit") && !reg.is_linkable("graph.format"));
    assert!(!reg.is_linkable("stat.regression.fit"));
    assert!(reg.brush_rows("graph.histogram", ws, &defaults("graph.histogram")).is_none());
}
