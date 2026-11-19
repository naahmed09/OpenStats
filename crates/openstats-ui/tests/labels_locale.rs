//! Phase 6a: value labels, regional settings and the date/time edge cases of typing into the worksheet.

use openstats_charts::core::{Scene, Shape};
use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_ui::dialogs::{Command, Value};
use openstats_ui::grid::CellRef;
use openstats_ui::workspace::{NavAction, Outcome, Target, Workspace};

fn loaded() -> Workspace {
    let mut w = Workspace::new(1200.0, 800.0);
    let mut wb = Workbook::new();
    wb.worksheets[0].add_column(Column::numeric("level", vec![1.0, 2.0, 3.0, 2.0, 1.0])).unwrap();
    wb.worksheets[0].add_column(Column::text("grp", vec!["a", "b", "a", "b", "a"])).unwrap();
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

fn texts(s: &Scene) -> Vec<String> {
    s.items
        .iter()
        .filter_map(|i| if let Shape::Text { text, .. } = &i.shape { Some(text.clone()) } else { None })
        .collect()
}

fn labels(w: &mut Workspace, col: usize, text: &str) -> Outcome {
    w.execute(Command::OpenDialog("data.valuelabels".into()));
    set(w, "col", Value::Vars(vec![col]));
    set(w, "labels", Value::Text(text.into()));
    w.run_dialog()
}

fn options(w: &mut Workspace, locale: usize) {
    w.execute(Command::OpenDialog("tools.options".into()));
    set(w, "locale", Value::Choice(locale));
    assert_eq!(w.run_dialog(), Outcome::None);
}

#[test]
fn value_labels_show_in_the_worksheet_and_undo() {
    let mut w = loaded();
    assert_eq!(labels(&mut w, 0, "1 = Low, 2.0 = Medium; 3=High"), Outcome::None);
    let c = &w.grid().worksheet().columns[0];
    assert_eq!((c.display(0).as_str(), c.display(1).as_str(), c.display(2).as_str()), ("Low", "Medium", "High"));
    assert_eq!(c.data.to_f64_vec(), vec![1.0, 2.0, 3.0, 2.0, 1.0], "the values are untouched");
    let t = texts(&w.render());
    assert!(t.iter().any(|x| x == "Medium") && t.iter().any(|x| x == "High"), "{t:?}");
    // Editing shows the value, not the label.
    w.grid_mut().select_cell(CellRef { row: 0, col: 0 });
    w.grid_mut().begin_edit(None, false);
    assert_eq!(w.grid().editor().unwrap().buffer, "1");
    w.grid_mut().cancel_edit();
    // Text columns can be labelled too; clearing removes the labels; one undo step each.
    assert_eq!(labels(&mut w, 1, "a=Alpha"), Outcome::None);
    assert_eq!(w.grid().worksheet().columns[1].display(0), "Alpha");
    w.execute(Command::Undo);
    assert_eq!(w.grid().worksheet().columns[1].display(0), "a");
    w.execute(Command::OpenDialog("data.valuelabels".into()));
    set(&mut w, "col", Value::Vars(vec![0]));
    set(&mut w, "clear", Value::Bool(true));
    assert_eq!(w.run_dialog(), Outcome::None);
    assert_eq!(w.grid().worksheet().columns[0].display(0), "1");
}

#[test]
fn value_labels_reject_bad_input_and_are_saved() {
    let mut w = loaded();
    for bad in ["", "1", "x=Low", "1=", "1=A, 1.0=B"] {
        assert!(matches!(labels(&mut w, 0, bad), Outcome::Error(_)), "{bad:?} is refused");
        w.cancel_dialog();
    }
    assert_eq!(labels(&mut w, 0, "1=Low"), Outcome::None);
    let bytes = w.save_project();
    let mut w2 = Workspace::new(1200.0, 800.0);
    assert_eq!(w2.open_project(&bytes), Outcome::None);
    assert_eq!(w2.grid().worksheet().columns[0].display(0), "Low");
}

#[test]
fn regional_settings_change_what_is_shown_and_typed() {
    let mut w = loaded();
    options(&mut w, 2); // continental Europe
    w.grid_mut().enter_cell(CellRef { row: 0, col: 0 }, "1,5").unwrap();
    assert_eq!(w.grid().worksheet().columns[0].data.to_f64_vec()[0], 1.5);
    assert!(texts(&w.render()).iter().any(|t| t == "1,5"), "shown with a decimal comma");
    w.grid_mut().enter_cell(CellRef { row: 1, col: 0 }, "1.234,5").unwrap();
    assert_eq!(w.grid().worksheet().columns[0].data.to_f64_vec()[1], 1234.5);
    w.grid_mut().enter_cell(CellRef { row: 2, col: 0 }, "1.234").unwrap();
    assert_eq!(
        w.grid().worksheet().columns[0].data.to_f64_vec()[2],
        1234.0,
        "a dot before three digits is a thousands mark"
    );
    // A dot that cannot be a thousands mark (1.5) is still read as a decimal point.
    w.grid_mut().enter_cell(CellRef { row: 3, col: 0 }, "1.5").unwrap();
    assert_eq!(w.grid().worksheet().columns[0].data.to_f64_vec()[3], 1.5);
    // Pasting reads the same way.
    w.grid_mut().select_cell(CellRef { row: 3, col: 0 });
    w.paste_text("7,25\n8,5");
    assert_eq!(&w.grid().worksheet().columns[0].data.to_f64_vec()[3..5], &[7.25, 8.5]);
    // CSV export follows.
    match w.navigator_action(Target::Worksheet(0), NavAction::Export) {
        Outcome::File(f) => {
            let t = String::from_utf8(f.bytes).unwrap();
            assert!(t.contains(";") && t.contains("1,5"), "{t}");
        }
        o => panic!("{o:?}"),
    }
    // And the setting is part of the project.
    let bytes = w.save_project();
    let mut w2 = Workspace::new(1200.0, 800.0);
    w2.open_project(&bytes);
    assert_eq!(w2.grid().locale(), openstats_core::locale::Locale::CONTINENTAL);
    // Importing a delimited file reads with the setting.
    assert_eq!(w2.import_data("eu.csv", b"v;w\n1,5;2,5\n"), Outcome::None);
    assert_eq!(w2.grid().worksheet().columns[0].data.to_f64_vec(), vec![1.5]);
}

#[test]
fn typed_dates_follow_the_date_order() {
    let mut w = loaded();
    w.grid_mut().enter_cell(CellRef { row: 0, col: 2 }, "18/11/2026").unwrap();
    assert_eq!(w.grid().worksheet().columns[2].kind(), openstats_core::ColumnKind::Text, "month 18 does not exist");
    let mut w = loaded();
    options(&mut w, 1); // United Kingdom: day/month/year
    w.grid_mut().enter_cell(CellRef { row: 0, col: 2 }, "18/11/2026").unwrap();
    w.grid_mut().enter_cell(CellRef { row: 1, col: 2 }, "01/02/2026").unwrap();
    let c = &w.grid().worksheet().columns[2];
    assert_eq!(c.kind(), openstats_core::ColumnKind::DateTime);
    let d = c.data.as_datetime().unwrap();
    assert_eq!(d[1], openstats_core::datetime::parse("2026-02-01").unwrap(), "1 February, not 2 January");
    assert_eq!(c.display_with(1, &w.grid().locale()), "1/2/2026");
}

#[test]
fn date_and_time_entry_edge_cases() {
    // What a person can type into a date column, whatever the order: these must read the same value.
    let want = openstats_core::datetime::parse("2026-11-18").unwrap();
    for s in [
        "2026-11-18",
        "11/18/2026",
        "Nov 18, 2026",
        "18-Nov-2026",
        "November 18 2026",
        "2026-11-18T00:00:00",
        "2026-11-18T00:00:00Z",
    ] {
        assert_eq!(openstats_core::datetime::parse(s).unwrap(), want, "{s}");
    }
    // Times and fractions.
    let t = openstats_core::datetime::parse("2026-11-18 13:45:30.25").unwrap();
    assert_eq!(openstats_core::datetime::format(t, "yyyy-MM-dd HH:mm:ss.ff"), "2026-11-18 13:45:30.25");
    let pm = openstats_core::datetime::parse("11/18/2026 1:45 PM").unwrap();
    assert_eq!(openstats_core::datetime::format(pm, "yyyy-MM-dd HH:mm"), "2026-11-18 13:45");
    let noon = openstats_core::datetime::parse("11/18/2026 12:00 AM").unwrap();
    assert_eq!(noon, want, "12 AM is midnight");
    // Leap days and impossible dates.
    assert!(openstats_core::datetime::parse("2024-02-29").is_ok());
    assert!(openstats_core::datetime::parse("2026-02-29").is_err());
    assert!(openstats_core::datetime::parse("2026-13-01").is_err());
    assert!(openstats_core::datetime::parse("2026-04-31").is_err());
    // Two-digit years use the pivot (30): 29 -> 2029, 31 -> 1931.
    assert_eq!(openstats_core::datetime::format(openstats_core::datetime::parse("1/2/29").unwrap(), "yyyy"), "2029");
    assert_eq!(openstats_core::datetime::format(openstats_core::datetime::parse("1/2/31").unwrap(), "yyyy"), "1931");
    // Before 1970 and far in the future round-trip.
    for s in ["1900-01-01", "1969-12-31 23:59:59", "2100-12-31", "0001-01-01", "9999-12-31"] {
        let ts = openstats_core::datetime::parse(s).unwrap();
        let back = openstats_core::datetime::format(ts, "yyyy-MM-dd HH:mm:ss");
        assert_eq!(openstats_core::datetime::parse(&back).unwrap(), ts, "{s} -> {back}");
    }
    // Missing tokens.
    assert_eq!(openstats_core::datetime::parse("*").unwrap(), openstats_core::missing::DATETIME_MISSING);
}
