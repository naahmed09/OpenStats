//! Write a real frame and a summary of it for the JavaScript decoder test:
//! `cargo run -p openstats-host --example dump_frame -- openstats-desktop/web/test/fixtures`.

use openstats_charts::render::canvas::{from_binary, Command};
use openstats_core::column::Column;
use openstats_core::Workbook;
use openstats_host::Host;
use std::collections::BTreeMap;

fn op(c: &Command) -> &'static str {
    match c {
        Command::Clear { .. } => "Clear",
        Command::Save => "Save",
        Command::Restore => "Restore",
        Command::Clip { .. } => "Clip",
        Command::FillStyle(_) => "FillStyle",
        Command::StrokeStyle { .. } => "StrokeStyle",
        Command::BeginPath => "BeginPath",
        Command::MoveTo(..) => "MoveTo",
        Command::LineTo(..) => "LineTo",
        Command::Arc { .. } => "Arc",
        Command::Rect { .. } => "Rect",
        Command::ClosePath => "ClosePath",
        Command::Fill => "Fill",
        Command::Stroke => "Stroke",
        Command::Text { .. } => "Text",
    }
}

fn esc(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let mut h = Host::new(1000.0, 650.0);
    let mut wb = Workbook::new();
    let x: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.5 + f64::from((i * 7) % 5)).collect();
    wb.worksheets[0].add_column(Column::numeric("measure", x)).unwrap();
    *h.workspace_mut().grid_mut() = openstats_ui::grid::GridController::new(wb, 800.0, 300.0);
    h.workspace_mut().set_size(1000.0, 650.0);
    for id in ["stat.basic.descriptive", "graph.histogram"] {
        h.workspace_mut().open_dialog(id).unwrap();
        let d = h.workspace_mut().dialog_mut().unwrap();
        d.select_column(0);
        d.focus_field("vars");
        d.add_selected();
        h.workspace_mut().run_dialog();
    }
    let frame = h.render().to_vec();
    let cmds = from_binary(&frame).expect("valid frame");
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for c in &cmds {
        *counts.entry(op(c)).or_default() += 1;
    }
    let texts: Vec<String> = cmds
        .iter()
        .filter_map(|c| if let Command::Text { text, .. } = c { Some(format!("\"{}\"", esc(text))) } else { None })
        .collect();
    let counts_json: Vec<String> = counts.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
    // First text command in full, to check every field the decoder reads.
    let first = cmds
        .iter()
        .find_map(|c| match c {
            Command::Text { x, y, text, size, color, bold, anchor, rotate, overbars } => Some(format!(
                "{{\"x\":{x},\"y\":{y},\"text\":\"{}\",\"size\":{size},\"color\":[{},{},{},{}],\"bold\":{bold},\"anchor\":{anchor},\"rotate\":{rotate},\"overbars\":{}}}",
                esc(text), color.r, color.g, color.b, color.a, overbars.len()
            )),
            _ => None,
        })
        .unwrap_or_default();
    let stroke = cmds
        .iter()
        .find_map(|c| match c {
            Command::StrokeStyle { color, width, dash } if !dash.is_empty() => Some(format!(
                "{{\"width\":{width},\"dash\":{},\"color\":[{},{},{},{}]}}",
                dash.len(),
                color.r,
                color.g,
                color.b,
                color.a
            )),
            _ => None,
        })
        .unwrap_or_else(|| "null".into());
    let summary = format!(
        "{{\"total\":{},\"counts\":{{{}}},\"texts\":[{}],\"firstText\":{},\"dashedStroke\":{},\"width\":1000,\"height\":650}}",
        cmds.len(),
        counts_json.join(","),
        texts.join(","),
        first,
        stroke
    );
    std::fs::write(format!("{dir}/frame.bin"), &frame).expect("write frame");
    std::fs::write(format!("{dir}/frame.json"), summary).expect("write summary");
    println!("{} commands, {} bytes", cmds.len(), frame.len());
}
