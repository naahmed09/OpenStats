//! Graphs that know their data (Phase 6a): the dialog that drew a graph is remembered so the graph can be
//! redrawn when the worksheet changes, points can be brushed back to worksheet rows, and the look of a graph
//! (colours, fonts, axis ranges, notes) is kept with it.

use crate::dialogs::spec::{Value, Values};
use openstats_charts::core::{Color, Marker, Point, Scene, Stroke, TextStyle, Theme, Viewport};
use openstats_core::{ColumnData, Worksheet};
use std::collections::BTreeSet;

/// How a graph was made: the dialog, its values and the worksheet it read.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphRecipe {
    /// Id of the graph dialog.
    pub dialog: String,
    /// The dialog values used.
    pub values: Values,
    /// Index of the worksheet the graph reads.
    pub sheet: usize,
    /// Which of the dialog's charts this graph is (0-based).
    pub chart: usize,
}

/// A text note placed at data coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    /// x position in data units.
    pub x: f64,
    /// y position in data units.
    pub y: f64,
    /// The note.
    pub text: String,
}

/// The look of a graph, beyond its title and axis labels.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphStyle {
    /// Multiplier for the title, label and tick font sizes.
    pub font_scale: f64,
    /// Series colours (`0xRRGGBB`) overriding the palette, by series position.
    pub colors: Vec<Option<u32>>,
    /// Draw grid lines (`None` keeps the theme's setting).
    pub grid: Option<bool>,
    /// Fixed x range.
    pub x_range: Option<(f64, f64)>,
    /// Fixed y range.
    pub y_range: Option<(f64, f64)>,
    /// Notes.
    pub notes: Vec<Annotation>,
}

impl Default for GraphStyle {
    fn default() -> GraphStyle {
        GraphStyle { font_scale: 1.0, colors: Vec::new(), grid: None, x_range: None, y_range: None, notes: Vec::new() }
    }
}

impl GraphStyle {
    /// Whether nothing is customised.
    pub fn is_default(&self) -> bool {
        *self == GraphStyle::default()
    }

    /// The theme and viewport to draw with.
    pub fn apply(&self, base: &Theme) -> (Theme, Viewport) {
        let mut t = base.clone();
        t.title_size *= self.font_scale;
        t.label_size *= self.font_scale;
        t.tick_size *= self.font_scale;
        if let Some(g) = self.grid {
            t.show_grid = g;
        }
        for (i, c) in self.colors.iter().enumerate() {
            if let (Some(hex), Some(slot)) = (c, t.palette.series.get_mut(i)) {
                *slot = Color::hex(*hex);
            }
        }
        (t, Viewport { x: self.x_range, y: self.y_range })
    }

    /// Draw the notes on `scene` (first panel's axes).
    pub fn draw_notes(&self, scene: &mut Scene, ink: Color) {
        let Some(ax) = scene.axes.first().cloned() else { return };
        for n in &self.notes {
            let p = ax.to_pixel(Point::new(n.x, n.y));
            if ax.plot.contains(p) {
                scene.marker(p, Marker::Cross, 5.0, ink, None);
                scene.text(Point::new(p.x + 6.0, p.y - 6.0), n.text.clone(), TextStyle::new(12.0, ink));
            }
        }
    }
}

/// Mark the points of `scene` (series 0) whose worksheet row is in `brushed` with a ring.
pub fn draw_brush(scene: &mut Scene, rows: &[usize], brushed: &BTreeSet<usize>, color: Color) {
    if brushed.is_empty() || rows.is_empty() {
        return;
    }
    let centers: Vec<Point> = scene
        .targets
        .iter()
        .filter(|t| t.series == 0)
        .filter(|t| rows.get(t.index).is_some_and(|r| brushed.contains(r)))
        .map(|t| match &t.region {
            openstats_charts::core::hit::Region::Point(p, _) => *p,
            openstats_charts::core::hit::Region::Rect(r) => r.center(),
        })
        .collect();
    for c in centers {
        scene.marker(c, Marker::Circle, 9.0, color.with_alpha(0.25), Some(Stroke::new(color, 2.0)));
    }
}

/// FNV-1a fingerprint of every column the recipe's dialog values select, so a change of the data
/// behind a graph is noticed cheaply.
pub fn fingerprint(ws: &Worksheet, values: &Values) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: u64| {
        h ^= b;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    eat(ws.n_cols() as u64);
    for v in values.0.values() {
        let Value::Vars(cols) = v else { continue };
        for &c in cols {
            eat(c as u64 + 0x9e37);
            match ws.columns.get(c).map(|c| &c.data) {
                Some(ColumnData::Numeric(d)) => {
                    eat(d.len() as u64);
                    for x in d {
                        eat(if x.is_nan() { 0x7ff8 } else { x.to_bits() });
                    }
                }
                Some(ColumnData::DateTime(d)) => {
                    eat(d.len() as u64);
                    for &x in d {
                        eat(x as u64);
                    }
                }
                Some(ColumnData::Text(d)) => {
                    eat(d.len() as u64);
                    for s in d {
                        for b in s.bytes() {
                            eat(u64::from(b));
                        }
                        eat(0xff);
                    }
                }
                None => eat(0xdead),
            }
        }
    }
    h
}

// ------------------------------------------------------------------------------ persistence

/// Everything about a graph that is kept in the project besides its chart.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphMeta {
    /// How the graph was made.
    pub recipe: Option<GraphRecipe>,
    /// Redraw automatically when the data change.
    pub auto: bool,
    /// Look.
    pub style: GraphStyle,
    /// Title, x label, y label chosen by the user (kept across redraws).
    pub labels: [Option<String>; 3],
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\t', "\\t").replace('\n', "\\n")
}

fn unesc(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some(o) => out.push(o),
                None => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Text form of `m` (line based, tab separated).
pub fn encode(m: &GraphMeta) -> String {
    let mut o = String::new();
    if let Some(r) = &m.recipe {
        o.push_str(&format!("recipe\t{}\t{}\t{}\t{}\n", esc(&r.dialog), r.sheet, r.chart, u8::from(m.auto)));
        for (k, v) in &r.values.0 {
            let (t, s) = match v {
                Value::Vars(c) => ('V', c.iter().map(usize::to_string).collect::<Vec<_>>().join(",")),
                Value::Num(x) => ('N', x.to_bits().to_string()),
                Value::Text(s) => ('T', esc(s)),
                Value::Choice(i) => ('C', i.to_string()),
                Value::Bool(b) => ('B', u8::from(*b).to_string()),
            };
            o.push_str(&format!("v\t{}\t{t}\t{s}\n", esc(k)));
        }
    }
    let s = &m.style;
    if s.font_scale != 1.0 {
        o.push_str(&format!("font\t{}\n", s.font_scale));
    }
    if let Some(g) = s.grid {
        o.push_str(&format!("grid\t{}\n", u8::from(g)));
    }
    if let Some((a, b)) = s.x_range {
        o.push_str(&format!("xrange\t{a}\t{b}\n"));
    }
    if let Some((a, b)) = s.y_range {
        o.push_str(&format!("yrange\t{a}\t{b}\n"));
    }
    for (i, c) in s.colors.iter().enumerate() {
        if let Some(c) = c {
            o.push_str(&format!("color\t{i}\t{c}\n"));
        }
    }
    for n in &s.notes {
        o.push_str(&format!("note\t{}\t{}\t{}\n", n.x, n.y, esc(&n.text)));
    }
    for (i, l) in m.labels.iter().enumerate() {
        if let Some(l) = l {
            o.push_str(&format!("label\t{i}\t{}\n", esc(l)));
        }
    }
    o
}

/// Inverse of [`encode`]; unknown lines are ignored.
pub fn decode(text: &str) -> GraphMeta {
    let mut m = GraphMeta { recipe: None, auto: true, style: GraphStyle::default(), labels: [None, None, None] };
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f.as_slice() {
            ["recipe", d, sheet, chart, auto] => {
                m.recipe = Some(GraphRecipe {
                    dialog: unesc(d),
                    values: Values::default(),
                    sheet: sheet.parse().unwrap_or(0),
                    chart: chart.parse().unwrap_or(0),
                });
                m.auto = *auto != "0";
            }
            ["v", k, t, s] => {
                if let Some(r) = m.recipe.as_mut() {
                    let v = match *t {
                        "V" => Value::Vars(s.split(',').filter_map(|x| x.parse().ok()).collect()),
                        "N" => Value::Num(f64::from_bits(s.parse().unwrap_or(f64::NAN.to_bits()))),
                        "T" => Value::Text(unesc(s)),
                        "C" => Value::Choice(s.parse().unwrap_or(0)),
                        _ => Value::Bool(*s == "1"),
                    };
                    r.values.0.insert(unesc(k), v);
                }
            }
            ["font", x] => m.style.font_scale = x.parse().unwrap_or(1.0),
            ["grid", x] => m.style.grid = Some(*x == "1"),
            ["xrange", a, b] => m.style.x_range = a.parse().ok().zip(b.parse().ok()),
            ["yrange", a, b] => m.style.y_range = a.parse().ok().zip(b.parse().ok()),
            ["color", i, c] => {
                if let (Ok(i), Ok(c)) = (i.parse::<usize>(), c.parse::<u32>()) {
                    if m.style.colors.len() <= i {
                        m.style.colors.resize(i + 1, None);
                    }
                    m.style.colors[i] = Some(c);
                }
            }
            ["note", x, y, t] => {
                if let (Ok(x), Ok(y)) = (x.parse(), y.parse()) {
                    m.style.notes.push(Annotation { x, y, text: unesc(t) });
                }
            }
            ["label", i, t] => {
                if let Ok(i) = i.parse::<usize>() {
                    if i < 3 {
                        m.labels[i] = Some(unesc(t));
                    }
                }
            }
            _ => {}
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_round_trips() {
        let mut v = Values::default();
        v.0.insert("y".into(), Value::Vars(vec![1]));
        v.0.insert("x".into(), Value::Vars(vec![0, 2]));
        v.0.insert("n".into(), Value::Num(2.5));
        v.0.insert("t".into(), Value::Text("a\tb\nc".into()));
        v.0.insert("c".into(), Value::Choice(3));
        v.0.insert("b".into(), Value::Bool(true));
        let m = GraphMeta {
            recipe: Some(GraphRecipe { dialog: "graph.scatter".into(), values: v, sheet: 1, chart: 2 }),
            auto: false,
            style: GraphStyle {
                font_scale: 1.25,
                colors: vec![None, Some(0xff0000)],
                grid: Some(false),
                x_range: Some((0.0, 10.0)),
                y_range: Some((-1.5, 3.0)),
                notes: vec![Annotation { x: 1.0, y: 2.0, text: "peak\there".into() }],
            },
            labels: [Some("T".into()), None, Some("Y\nlabel".into())],
        };
        assert_eq!(decode(&encode(&m)), m);
    }
}
