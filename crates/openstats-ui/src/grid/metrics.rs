//! Geometry of the grid: row/column sizes, scroll position and virtualization.

use openstats_charts::core::font::text_width;
use std::collections::BTreeMap;

/// Maximum number of rows of a worksheet (as in Minitab).
pub const MAX_ROWS: usize = 10_000_000;
/// Maximum number of columns of a worksheet.
pub const MAX_COLS: usize = 4_000;

/// Sizes in pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct Metrics {
    /// Height of a data row.
    pub row_height: f64,
    /// Height of the column header (label row plus name row).
    pub header_height: f64,
    /// Width of a column without an explicit width.
    pub default_col_width: f64,
    /// Smallest column width.
    pub min_col_width: f64,
    /// Largest column width.
    pub max_col_width: f64,
    /// Cell font size.
    pub font_size: f64,
    /// Scroll bar thickness.
    pub scrollbar: f64,
    /// Rows always shown (blank rows below the data to type into).
    pub min_display_rows: usize,
    /// Columns always shown.
    pub min_display_cols: usize,
    /// Extra blank rows shown below the last data row.
    pub spare_rows: usize,
    /// Extra blank columns shown right of the last data column.
    pub spare_cols: usize,
    /// Pointer distance from a column border that grabs it for resizing.
    pub resize_grip: f64,
}

impl Default for Metrics {
    fn default() -> Metrics {
        Metrics {
            row_height: 22.0,
            header_height: 40.0,
            default_col_width: 84.0,
            min_col_width: 28.0,
            max_col_width: 600.0,
            font_size: 12.0,
            scrollbar: 12.0,
            min_display_rows: 1000,
            min_display_cols: 20,
            spare_rows: 100,
            spare_cols: 5,
            resize_grip: 4.0,
        }
    }
}

/// Column widths: a default plus sparse overrides, so even 4000 columns cost nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnWidths {
    default: f64,
    overrides: BTreeMap<usize, f64>,
}

impl ColumnWidths {
    /// All columns `default` wide.
    pub fn new(default: f64) -> ColumnWidths {
        ColumnWidths { default, overrides: BTreeMap::new() }
    }

    /// Width of column `c`.
    pub fn width(&self, c: usize) -> f64 {
        self.overrides.get(&c).copied().unwrap_or(self.default)
    }

    /// Set the width of column `c` (reverting to the default removes the override).
    pub fn set(&mut self, c: usize, w: f64) {
        if (w - self.default).abs() < 1e-9 {
            self.overrides.remove(&c);
        } else {
            self.overrides.insert(c, w);
        }
    }

    /// Left edge of column `c` relative to the first column.
    pub fn x_of(&self, c: usize) -> f64 {
        let extra: f64 = self.overrides.range(..c).map(|(_, w)| w - self.default).sum();
        c as f64 * self.default + extra
    }

    /// Index of the column containing content x position `x` (clamped to `0 .. n`).
    pub fn col_at(&self, x: f64, n: usize) -> usize {
        if n == 0 || x <= 0.0 {
            return 0;
        }
        let (mut lo, mut hi) = (0usize, n - 1);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if self.x_of(mid) <= x {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }
}

/// The visible window: size in pixels and scroll offsets of the *body* (headers are frozen).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct View {
    /// Total width available.
    pub width: f64,
    /// Total height available.
    pub height: f64,
    /// Horizontal scroll offset in content pixels.
    pub scroll_x: f64,
    /// Vertical scroll offset in content pixels.
    pub scroll_y: f64,
}

/// Width of the row-number gutter for `rows` rows.
pub fn gutter_width(rows: usize, font_size: f64) -> f64 {
    let digits = rows.max(1).to_string().len() as f64;
    (digits * text_width("0", font_size, false) + 18.0).max(38.0)
}

/// Maps a scroll position of a size-capped proxy element (browsers limit element heights to
/// roughly 16–33 million pixels) onto the logical scroll offset of a huge worksheet, and back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScrollProxy {
    /// Largest scroll range of the proxy element.
    pub proxy_range: f64,
    /// Largest logical scroll offset.
    pub logical_range: f64,
}

impl ScrollProxy {
    /// Proxy for a logical range, capping the proxy at `cap` pixels.
    pub fn new(logical_range: f64, cap: f64) -> ScrollProxy {
        ScrollProxy { proxy_range: logical_range.clamp(0.0, cap), logical_range: logical_range.max(0.0) }
    }

    /// Logical offset for a proxy offset.
    pub fn to_logical(&self, proxy: f64) -> f64 {
        if self.proxy_range <= 0.0 {
            return 0.0;
        }
        (proxy / self.proxy_range).clamp(0.0, 1.0) * self.logical_range
    }

    /// Proxy offset for a logical offset.
    pub fn to_proxy(&self, logical: f64) -> f64 {
        if self.logical_range <= 0.0 {
            return 0.0;
        }
        (logical / self.logical_range).clamp(0.0, 1.0) * self.proxy_range
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_prefix_and_lookup() {
        let mut w = ColumnWidths::new(80.0);
        assert_eq!(w.x_of(3), 240.0);
        w.set(1, 120.0);
        w.set(5, 40.0);
        assert_eq!(w.x_of(1), 80.0);
        assert_eq!(w.x_of(2), 200.0);
        assert_eq!(w.x_of(6), 80.0 * 6.0 + 40.0 - 40.0);
        for c in 0..30 {
            // Every pixel of a column maps back to it.
            let (a, b) = (w.x_of(c), w.x_of(c) + w.width(c));
            assert_eq!(w.col_at(a, 30), c);
            assert_eq!(w.col_at(b - 0.01, 30), c);
        }
        assert_eq!(w.col_at(-5.0, 30), 0);
        assert_eq!(w.col_at(1e9, 30), 29);
        assert_eq!(w.col_at(10.0, 0), 0);
        w.set(1, 80.0);
        assert_eq!(w.x_of(2), 160.0);
    }

    #[test]
    fn gutter_grows_with_digits() {
        assert!(gutter_width(10_000_000, 12.0) > gutter_width(99, 12.0));
        assert!(gutter_width(0, 12.0) >= 38.0);
    }

    #[test]
    fn proxy_round_trips() {
        let p = ScrollProxy::new(220_000_000.0, 16_000_000.0);
        assert_eq!(p.proxy_range, 16_000_000.0);
        assert_eq!(p.to_logical(0.0), 0.0);
        assert_eq!(p.to_logical(16_000_000.0), 220_000_000.0);
        for l in [0.0, 1.0e6, 123_456_789.0, 220_000_000.0] {
            assert!((p.to_logical(p.to_proxy(l)) - l).abs() < 1e-3);
        }
        let small = ScrollProxy::new(5_000.0, 16_000_000.0);
        assert_eq!(small.to_proxy(1234.0), 1234.0);
        assert_eq!(ScrollProxy::new(0.0, 10.0).to_logical(5.0), 0.0);
        assert_eq!(ScrollProxy::new(0.0, 10.0).to_proxy(5.0), 0.0);
    }
}
