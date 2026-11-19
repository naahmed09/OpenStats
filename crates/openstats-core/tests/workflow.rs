//! End-to-end tests of `openstats-core`: import → transform (undoable) →
//! persist → reload, plus a randomised undo/redo stress test.

use openstats_core::io::{csv, json, otb};
use openstats_core::ops::{self, recode, sort, stack, subset, unstack, Target};
use openstats_core::{Column, ColumnData, ColumnKind, Edit, History, Value, Workbook};

const CSV: &str = "\
Part,Operator,Measurement,Date
P1,Ann,10.02,2026-11-02
P1,Bob,10.05,2026-11-02
P2,Ann,9.98,2026-11-03
P2,Bob,*,2026-11-03
P3,Ann,10.11,2026-11-04
P3,Bob,10.07,
";

fn load() -> Workbook {
    let ws = csv::read_csv(CSV, &csv::CsvReadOptions { sheet_name: "Gage".into(), ..Default::default() }).unwrap();
    let mut wb = Workbook::empty();
    wb.worksheets.push(ws);
    wb
}

#[test]
fn import_transform_persist_undo() {
    let mut wb = load();
    let mut h = History::default();
    let original = wb.clone();
    {
        let ws = wb.active_sheet().unwrap();
        assert_eq!(ws.columns[2].kind(), ColumnKind::Numeric);
        assert_eq!(ws.columns[3].kind(), ColumnKind::DateTime);
        assert_eq!(ws.columns[2].data.missing_count(), 1);
    }

    // Unstack measurements by operator into new columns.
    let ws = wb.active_sheet().unwrap().clone();
    let cols = unstack::unstack(&ws, &[2], &[1], Default::default()).unwrap();
    let e = ops::store_columns(&wb, 0, cols, Target::Append).unwrap();
    h.execute(&mut wb, "Unstack", e).unwrap();
    assert_eq!(wb.active_sheet().unwrap().columns[4].name(), "Measurement_Ann");

    // Stack them back with subscripts.
    let ws = wb.active_sheet().unwrap().clone();
    let st = stack::stack_columns(&ws, &[4, 5], stack::Subscripts::Names, "Stacked").unwrap();
    let e = ops::store_columns(&wb, 0, st, Target::Append).unwrap();
    h.execute(&mut wb, "Stack", e).unwrap();
    assert_eq!(wb.active_sheet().unwrap().columns[6].len(), 6);

    // Sort everything by measurement descending inside one transaction.
    let ws = wb.active_sheet().unwrap().clone();
    let all: Vec<usize> = (0..4).collect();
    let sorted = sort::sort_columns(&ws, &all, &[sort::SortKey::desc(2)]).unwrap();
    h.begin_group("Sort");
    let e = ops::store_columns(&wb, 0, sorted, Target::At(0)).unwrap();
    h.execute(&mut wb, "", e).unwrap();
    h.end_group().unwrap();
    assert_eq!(wb.active_sheet().unwrap().columns[2].get(0), Value::Numeric(10.11));
    assert!(wb.active_sheet().unwrap().columns[2].data.is_missing(5), "missing sorts last");

    // Recode operators to codes.
    let ws = wb.active_sheet().unwrap().clone();
    let coded = recode::recode(
        &ws.columns[1].data,
        &[recode::RecodeRule::Value { from: "Ann".into(), to: Value::Numeric(1.0) }],
        ColumnKind::Numeric,
        &recode::Unmatched::Value(Value::Numeric(2.0)),
        true,
    )
    .unwrap();
    let e = ops::store_columns(&wb, 0, vec![Column::new("OpCode", coded)], Target::Append).unwrap();
    h.execute(&mut wb, "Recode", e).unwrap();

    // Subset to a new worksheet.
    let ws = wb.active_sheet().unwrap().clone();
    let sub = subset::subset_worksheet(&ws, &subset::RowSelector::NotMissing(2), false, "Complete").unwrap();
    let e = ops::add_worksheets(&wb, vec![sub]).unwrap();
    h.execute(&mut wb, "Subset", e).unwrap();
    assert_eq!(wb.worksheets[1].columns[2].len(), 5);

    // Persist in both project formats and reload.
    let bin = otb::encode(&wb);
    assert_eq!(otb::decode(&bin).unwrap(), wb);
    let text = json::workbook_to_json(&wb, true);
    assert_eq!(json::workbook_from_json(&text).unwrap(), wb);

    // Undo everything back to the import state, then redo it all.
    let final_state = wb.clone();
    let mut labels = Vec::new();
    while h.can_undo() {
        labels.push(h.undo(&mut wb).unwrap());
    }
    assert_eq!(labels, ["Subset", "Recode", "Sort", "Stack", "Unstack"]);
    assert_eq!(wb, original);
    while h.can_redo() {
        h.redo(&mut wb).unwrap();
    }
    assert_eq!(wb, final_state);
}

/// Deterministic 64-bit LCG (Knuth MMIX constants) so the stress test needs no RNG crate.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn random_edit(rng: &mut Lcg, wb: &Workbook) -> Edit {
    let ws = wb.sheet(0).unwrap();
    let ncols = ws.n_cols();
    let nrows = ws.n_rows().max(1);
    match rng.below(8) {
        0 if ncols > 0 => {
            let col = rng.below(ncols);
            let value = match ws.columns[col].kind() {
                ColumnKind::Numeric => Value::Numeric(rng.below(1000) as f64 / 10.0),
                ColumnKind::Text => Value::Text(format!("t{}", rng.below(50))),
                ColumnKind::DateTime => Value::DateTime(rng.below(1_000_000) as i64 * 1_000_000),
            };
            Edit::SetCell { sheet: 0, col, row: rng.below(nrows + 3), value }
        }
        1 => Edit::insert_blank_rows(wb, 0, rng.below(nrows + 1), 1 + rng.below(3)).unwrap(),
        2 => Edit::remove_rows_all(wb, 0, rng.below(nrows), 1 + rng.below(3)).unwrap(),
        3 => {
            let at = rng.below(ncols + 1);
            let kind = rng.below(3);
            let data = match kind {
                0 => ColumnData::Numeric((0..rng.below(6)).map(|i| i as f64).collect()),
                1 => ColumnData::Text((0..rng.below(6)).map(|i| format!("s{i}")).collect()),
                _ => ColumnData::DateTime((0..rng.below(6)).map(|i| i as i64 * 86_400_000_000).collect()),
            };
            Edit::InsertColumn { sheet: 0, at, column: Column::new("", data) }
        }
        4 if ncols > 1 => Edit::RemoveColumn { sheet: 0, at: rng.below(ncols) },
        5 if ncols > 0 => Edit::Resize { sheet: 0, col: rng.below(ncols), len: rng.below(nrows + 4) },
        6 => Edit::SetConstant { sheet: 0, k: 1 + rng.below(3), value: Some(Value::Numeric(rng.below(9) as f64)) },
        _ => Edit::Batch(vec![
            Edit::insert_blank_rows(wb, 0, 0, 1).unwrap(),
            Edit::SetConstant { sheet: 0, k: 9, value: None },
        ]),
    }
}

#[test]
fn randomized_undo_redo_matches_snapshots() {
    let mut rng = Lcg(0x5EED_0F0F_E7AB);
    let mut wb = load();
    let mut h = History::new(10_000, usize::MAX);
    let mut snapshots = vec![wb.clone()];
    for _ in 0..600 {
        let e = random_edit(&mut rng, &wb);
        if h.execute(&mut wb, "edit", e).is_ok() {
            snapshots.push(wb.clone());
        }
        // Occasionally undo a few steps and branch from there.
        if rng.below(10) == 0 {
            for _ in 0..rng.below(4) {
                if h.can_undo() {
                    h.undo(&mut wb).unwrap();
                    snapshots.pop();
                    assert_eq!(&wb, snapshots.last().unwrap());
                }
            }
        }
    }
    let depth = h.undo_len();
    assert_eq!(depth + 1, snapshots.len());
    for i in (0..depth).rev() {
        h.undo(&mut wb).unwrap();
        assert_eq!(wb, snapshots[i], "undo mismatch at step {i}");
    }
    for snap in snapshots.iter().skip(1) {
        h.redo(&mut wb).unwrap();
        assert_eq!(&wb, snap);
    }
}

#[test]
fn large_worksheet_roundtrip() {
    let n = 200_000;
    let mut wb = Workbook::new();
    let ws = wb.sheet_mut(0).unwrap();
    ws.add_column(Column::numeric("x", (0..n).map(|i| (i as f64).sin()).collect())).unwrap();
    ws.add_column(Column::text("g", (0..n).map(|i| format!("g{}", i % 17)).collect::<Vec<_>>())).unwrap();
    let bytes = otb::encode(&wb);
    assert_eq!(otb::decode(&bytes).unwrap(), wb);
    let text = csv::write_csv(wb.sheet(0).unwrap(), &Default::default());
    let back = csv::read_csv(&text, &Default::default()).unwrap();
    assert_eq!(back.columns[0].data, wb.sheet(0).unwrap().columns[0].data);
    assert_eq!(back.columns[1].levels().len(), 17);
}

#[test]
fn calculator_and_formula_columns_are_undoable_and_persisted() {
    use openstats_core::formula::{self, recalc, FunctionRegistry};
    let mut wb = load();
    let mut h = History::default();
    let reg = FunctionRegistry::standard();
    let original = wb.clone();

    // Calc > Calculator: store a computed column.
    let ws = wb.active_sheet().unwrap().clone();
    let r =
        formula::evaluate("IF(ISMISSING(Measurement), *, (Measurement - MEAN(Measurement)) / STDEV(Measurement))", &ws)
            .unwrap();
    let e = ops::store_columns(&wb, 0, vec![Column::new("Z", r.data)], Target::Append).unwrap();
    h.execute(&mut wb, "Calculator", e).unwrap();
    let z = wb.active_sheet().unwrap().columns[4].data.to_f64_vec();
    assert!(z[3].is_nan());
    let present: Vec<f64> = z.iter().copied().filter(|v| !v.is_nan()).collect();
    assert!(present.iter().sum::<f64>().abs() < 1e-12, "standardized values sum to zero");

    // A formula column that depends on it, then edit an input and recalculate.
    let e = ops::store_columns(&wb, 0, vec![Column::numeric("Flag", vec![])], Target::Append).unwrap();
    h.execute(&mut wb, "New column", e).unwrap();
    let e = recalc::set_formula_edit(&wb, 0, 5, Some("ABS(Z) > 1"), &reg, None).unwrap();
    h.execute(&mut wb, "Formula", e).unwrap();
    let flags = wb.active_sheet().unwrap().columns[5].data.to_f64_vec();
    assert_eq!(flags.len(), 6);

    // Persist with the formula and reload.
    let back = openstats_core::io::otb::decode(&openstats_core::io::otb::encode(&wb)).unwrap();
    assert_eq!(back.worksheets[0].columns[5].props.formula.as_deref(), Some("ABS(Z) > 1"));
    let back =
        openstats_core::io::json::workbook_from_json(&openstats_core::io::json::workbook_to_json(&wb, false)).unwrap();
    assert_eq!(back, wb);

    while h.can_undo() {
        h.undo(&mut wb).unwrap();
    }
    assert_eq!(wb, original);
}

#[test]
fn xlsx_import_to_project_roundtrip() {
    let bytes = include_bytes!("fixtures/openpyxl_sample.xlsx");
    let wb = openstats_core::io::load_workbook(Some("sample.xlsx"), bytes).unwrap();
    let bin = openstats_core::io::otb::encode(&wb);
    assert_eq!(openstats_core::io::otb::decode(&bin).unwrap(), wb);
    let ws = &wb.worksheets[0];
    let r = openstats_core::formula::evaluate("MEAN(Diameter) + YEAR(MAX(Inspected)) * 0", ws).unwrap();
    assert!((r.data.to_f64_vec()[0] - 10.036_666_666_666_667).abs() < 1e-12);
}
