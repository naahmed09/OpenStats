//! Column formulas: dependency ordering, cycle detection and recalculation.
//!
//! A column whose [`ColumnProps::formula`](crate::column::ColumnProps) is set
//! is recomputed from its formula whenever its inputs change. Formulas may
//! reference other formula columns; evaluation follows a topological order and
//! circular references are rejected.

use super::ast::{ColumnRef, Expr};
use super::eval::EvalCtx;
use super::functions::FunctionRegistry;
use super::parser::parse;
use super::Arr;
use crate::column::Column;
use crate::edit::Edit;
use crate::error::{CoreError, Result};
use crate::workbook::Workbook;
use crate::worksheet::Worksheet;
use std::collections::{BTreeSet, VecDeque};

/// Parsed formula columns of a worksheet: (column index, expression, deps).
fn formula_columns(ws: &Worksheet) -> Result<Vec<(usize, Expr, BTreeSet<usize>)>> {
    let mut out = Vec::new();
    for (i, c) in ws.columns.iter().enumerate() {
        let Some(f) = &c.props.formula else { continue };
        let e = parse(f).map_err(|err| CoreError::invalid(format!("formula of {}: {err}", ws.display_name(i))))?;
        let mut deps = BTreeSet::new();
        for r in e.column_refs() {
            let j = match &r {
                ColumnRef::Number(n) if *n <= ws.n_cols() => n - 1,
                ColumnRef::Name(name) if ws.find(name).is_some() => ws.find(name).expect("checked"),
                _ => {
                    let label = match r {
                        ColumnRef::Number(n) => format!("C{n}"),
                        ColumnRef::Name(s) => s,
                    };
                    return Err(CoreError::NotFound(format!(
                        "column {label} used in the formula of {}",
                        ws.display_name(i)
                    )));
                }
            };
            deps.insert(j);
        }
        out.push((i, e, deps));
    }
    Ok(out)
}

/// Formula columns in evaluation order (Kahn's algorithm). Errors on cycles.
pub fn dependency_order(ws: &Worksheet) -> Result<Vec<usize>> {
    Ok(ordered(ws)?.into_iter().map(|(i, _)| i).collect())
}

fn ordered(ws: &Worksheet) -> Result<Vec<(usize, Expr)>> {
    let fcols = formula_columns(ws)?;
    let is_formula: BTreeSet<usize> = fcols.iter().map(|f| f.0).collect();
    let mut indeg: Vec<usize> =
        fcols.iter().map(|(_, _, d)| d.iter().filter(|j| is_formula.contains(j)).count()).collect();
    let pos_of = |col: usize| fcols.iter().position(|f| f.0 == col);
    let mut queue: VecDeque<usize> = (0..fcols.len()).filter(|&k| indeg[k] == 0).collect();
    let mut order = Vec::with_capacity(fcols.len());
    while let Some(k) = queue.pop_front() {
        order.push(k);
        let col = fcols[k].0;
        for (m, (_, _, deps)) in fcols.iter().enumerate() {
            if deps.contains(&col) {
                indeg[m] -= 1;
                if indeg[m] == 0 {
                    queue.push_back(m);
                }
            }
        }
    }
    if order.len() != fcols.len() {
        let stuck: Vec<String> =
            (0..fcols.len()).filter(|k| !order.contains(k)).map(|k| ws.display_name(fcols[k].0)).collect();
        return Err(CoreError::invalid(format!("circular reference among columns: {}", stuck.join(", "))));
    }
    let _ = pos_of;
    let mut fcols: Vec<Option<(usize, Expr, BTreeSet<usize>)>> = fcols.into_iter().map(Some).collect();
    Ok(order.into_iter().map(|k| fcols[k].take().map(|(i, e, _)| (i, e)).expect("each once")).collect())
}

/// Formula columns that (transitively) depend on column `col`.
pub fn dependents(ws: &Worksheet, col: usize) -> Result<Vec<usize>> {
    let fcols = formula_columns(ws)?;
    let mut hit: BTreeSet<usize> = BTreeSet::new();
    let mut frontier = vec![col];
    while let Some(c) = frontier.pop() {
        for (i, _, deps) in &fcols {
            if deps.contains(&c) && hit.insert(*i) {
                frontier.push(*i);
            }
        }
    }
    Ok(hit.into_iter().collect())
}

fn result_data(a: Arr) -> crate::column::ColumnData {
    a.data
}

/// Evaluate every formula column in dependency order on a working copy, so
/// that later formulas see earlier results. Returns new data per column.
pub fn recalculate(
    ws: &Worksheet,
    registry: &FunctionRegistry,
    now: Option<i64>,
) -> Result<Vec<(usize, crate::column::ColumnData)>> {
    let order = ordered(ws)?;
    let mut work = ws.clone();
    let mut out = Vec::with_capacity(order.len());
    for (i, e) in order {
        let ctx = EvalCtx { ws: &work, registry, now };
        let r =
            ctx.eval(&e).map_err(|err| CoreError::invalid(format!("formula of {}: {err}", work.display_name(i))))?;
        let data = result_data(r);
        work.columns[i].data = data.clone();
        out.push((i, data));
    }
    Ok(out)
}

/// An undoable edit that refreshes every formula column of a sheet whose
/// value changed.
pub fn recalc_edit(wb: &Workbook, sheet: usize, registry: &FunctionRegistry, now: Option<i64>) -> Result<Edit> {
    let ws = wb.sheet(sheet)?;
    let mut edits = Vec::new();
    for (i, data) in recalculate(ws, registry, now)? {
        if ws.columns[i].data != data {
            let mut col = ws.columns[i].clone();
            col.data = data;
            edits.push(Edit::ReplaceColumn { sheet, col: i, column: col });
        }
    }
    Ok(Edit::Batch(edits))
}

/// An undoable edit that sets (or clears, with `None`) the formula of column
/// `col` and recalculates it and its dependents. Fails without changing
/// anything if the formula does not parse, references unknown columns,
/// creates a cycle, or fails to evaluate.
pub fn set_formula_edit(
    wb: &Workbook,
    sheet: usize,
    col: usize,
    formula: Option<&str>,
    registry: &FunctionRegistry,
    now: Option<i64>,
) -> Result<Edit> {
    let ws = wb.sheet(sheet)?;
    ws.column(col)?;
    let mut trial = ws.clone();
    trial.columns[col].props.formula = formula.map(str::to_string);
    let results = recalculate(&trial, registry, now)?;
    let mut edits = Vec::new();
    // The target column always gets its new props; data only if it has a formula.
    let mut target: Column = trial.columns[col].clone();
    for (i, data) in results {
        if i == col {
            target.data = data;
        } else if ws.columns[i].data != data {
            let mut c = ws.columns[i].clone();
            c.data = data;
            edits.push(Edit::ReplaceColumn { sheet, col: i, column: c });
        }
    }
    edits.insert(0, Edit::ReplaceColumn { sheet, col, column: target });
    Ok(Edit::Batch(edits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;
    use crate::History;

    fn wb() -> Workbook {
        let mut wb = Workbook::new();
        let ws = wb.sheet_mut(0).unwrap();
        ws.add_column(Column::numeric("x", vec![1.0, 2.0, 3.0])).unwrap();
        ws.add_column(Column::numeric("y", vec![])).unwrap();
        ws.add_column(Column::numeric("z", vec![])).unwrap();
        wb
    }

    #[test]
    fn chained_formulas_and_undo() {
        let reg = FunctionRegistry::standard();
        let mut w = wb();
        let mut h = History::default();
        let e = set_formula_edit(&w, 0, 2, Some("y * 10"), &reg, None);
        assert!(e.is_ok(), "y has no formula yet but exists");
        h.execute(&mut w, "z", e.unwrap()).unwrap();
        // y is empty, so z = y*10 is empty
        assert_eq!(w.sheet(0).unwrap().columns[2].len(), 0);
        let e = set_formula_edit(&w, 0, 1, Some("x + 1"), &reg, None).unwrap();
        h.execute(&mut w, "y", e).unwrap();
        let ws = w.sheet(0).unwrap();
        assert_eq!(ws.columns[1].data, ColumnData::Numeric(vec![2.0, 3.0, 4.0]));
        assert_eq!(ws.columns[2].data, ColumnData::Numeric(vec![20.0, 30.0, 40.0]));
        assert_eq!(dependency_order(ws).unwrap(), vec![1, 2]);
        assert_eq!(dependents(ws, 0).unwrap(), vec![1, 2]);

        // Edit x, then recalc.
        h.execute(&mut w, "cell", Edit::SetCell { sheet: 0, col: 0, row: 0, value: 10.0.into() }).unwrap();
        let e = recalc_edit(&w, 0, &reg, None).unwrap();
        h.execute(&mut w, "recalc", e).unwrap();
        assert_eq!(w.sheet(0).unwrap().columns[2].get(0).as_f64(), 110.0);

        h.undo(&mut w).unwrap();
        h.undo(&mut w).unwrap();
        h.undo(&mut w).unwrap();
        assert!(w.sheet(0).unwrap().columns[1].props.formula.is_none());
    }

    #[test]
    fn cycles_and_bad_refs_rejected() {
        let reg = FunctionRegistry::standard();
        let mut w = wb();
        let e = set_formula_edit(&w, 0, 1, Some("z + 1"), &reg, None).unwrap();
        e.apply(&mut w).unwrap();
        let err = set_formula_edit(&w, 0, 2, Some("y * 2"), &reg, None).unwrap_err();
        assert!(err.to_string().contains("circular"), "{err}");
        assert!(set_formula_edit(&w, 0, 2, Some("C2 + C9"), &reg, None).is_err());
        assert!(set_formula_edit(&w, 0, 2, Some("x +"), &reg, None).is_err());
        assert!(set_formula_edit(&w, 0, 0, Some("x + 1"), &reg, None).is_err(), "self reference");
    }
}
