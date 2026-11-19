//! Phase 6a: value labels and regional settings in the data engine.

use openstats_core::column::Column;
use openstats_core::io::{json, load_workbook_in, otb};
use openstats_core::locale::Locale;
use openstats_core::Workbook;

fn labelled() -> Workbook {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    let mut a = Column::numeric("level", vec![1.0, 2.0, 3.0, 2.5, f64::NAN]);
    a.props.value_labels =
        vec![("1".into(), "Low".into()), ("2".into(), "Medium".into()), ("2.5".into(), "Mid-high".into())];
    let mut t = Column::text("grp", vec!["a", "b", "a", "c", ""]);
    t.props.value_labels = vec![("a".into(), "Alpha".into())];
    ws.add_column(a).unwrap();
    ws.add_column(t).unwrap();
    wb
}

#[test]
fn labels_replace_values_only_in_display() {
    let wb = labelled();
    let c = &wb.worksheets[0].columns[0];
    assert_eq!(c.display(0), "Low");
    assert_eq!(c.display(1), "Medium");
    assert_eq!(c.display(2), "3", "unlabelled values show as they are");
    assert_eq!(c.display(3), "Mid-high");
    assert_eq!(c.label_for(4), None, "missing values have no label");
    assert_eq!(c.get(0), openstats_core::Value::Numeric(1.0), "the data keep their values");
    let t = &wb.worksheets[0].columns[1];
    assert_eq!((t.display(0).as_str(), t.display(1).as_str()), ("Alpha", "b"));
}

#[test]
fn labels_survive_otb_and_json() {
    let wb = labelled();
    let back = otb::decode(&otb::encode(&wb)).unwrap();
    assert_eq!(back.worksheets[0].columns[0].props.value_labels, wb.worksheets[0].columns[0].props.value_labels);
    assert_eq!(back.worksheets[0].columns[1].props.value_labels, wb.worksheets[0].columns[1].props.value_labels);
    let j = json::workbook_to_json(&wb, false);
    assert!(j.contains("value_labels"));
    let back = json::workbook_from_json(&j).unwrap();
    assert_eq!(back.worksheets[0].columns[0].props.value_labels, wb.worksheets[0].columns[0].props.value_labels);
    // Columns without labels write no extra JSON.
    let plain = json::workbook_to_json(&Workbook::new(), false);
    assert!(!plain.contains("value_labels"));
}

#[test]
fn locales_format_numbers_and_dates() {
    let mut wb = Workbook::new();
    let ws = &mut wb.worksheets[0];
    ws.add_column(Column::numeric("x", vec![1.5, 1234.25])).unwrap();
    let ts = openstats_core::datetime::parse("2026-11-18").unwrap();
    ws.add_column(Column::datetime("d", vec![ts])).unwrap();
    let (x, d) = (&ws.columns[0], &ws.columns[1]);
    assert_eq!(x.display_with(0, &Locale::CONTINENTAL), "1,5");
    assert_eq!(x.display_with(0, &Locale::UNITED_STATES), "1.5");
    assert_eq!(d.display_with(0, &Locale::UNITED_STATES), "11/18/2026");
    assert_eq!(d.display_with(0, &Locale::CONTINENTAL), "18/11/2026");
    assert_eq!(d.display_with(0, &Locale::UNITED_KINGDOM), "18/11/2026");
    assert_eq!(d.display_with(0, &Locale::ISO), "2026-11-18");
    for (i, (_, l)) in Locale::PRESETS.iter().enumerate() {
        assert_eq!(l.preset_index(), i);
    }
}

#[test]
fn delimited_files_are_read_with_the_locale() {
    let csv = "x;when\n1,5;18/11/2026\n2,25;01/02/2026\n";
    let wb = load_workbook_in(Some("a.csv"), csv.as_bytes(), &Locale::CONTINENTAL).unwrap();
    let ws = &wb.worksheets[0];
    assert_eq!(ws.columns[0].data.to_f64_vec(), vec![1.5, 2.25]);
    let d = ws.columns[1].data.as_datetime().expect("dates were recognised");
    assert_eq!(d[0], openstats_core::datetime::parse("2026-11-18").unwrap());
    assert_eq!(d[1], openstats_core::datetime::parse("2026-02-01").unwrap(), "day first");
    let us = load_workbook_in(Some("a.csv"), "x,when\n1.5,11/18/2026\n".as_bytes(), &Locale::UNITED_STATES).unwrap();
    assert_eq!(us.worksheets[0].columns[0].data.to_f64_vec(), vec![1.5]);
}
