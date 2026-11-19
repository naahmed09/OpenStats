//! OpenStats 2D chart engine (Phases 3a + 3b), std only.
//!
//! * [`core`] — geometry, scales (linear, log, probability, band, date/time),
//!   Wilkinson-extended tick generation, font metrics and bitmap glyphs,
//!   palettes and styles, the scene graph, layout, hit-testing and viewports.
//! * [`render`] — three back ends for a [`core::Scene`]: standalone SVG, a
//!   canvas draw-command stream (JSON and compact binary) and a software
//!   anti-aliased PNG rasteriser.
//! * [`plots`] — builders that turn every `ChartSpec` of `openstats-stats`
//!   (and richer typed inputs) into scenes: histograms, box plots, interval,
//!   individual-value, bar, pie, ECDF and probability plots, scatter plots with
//!   fits and bands, LOWESS, main-effects / interaction / cube / contour /
//!   surface plots, control charts with zones, stages and rule labels, Pareto,
//!   capability and multi-panel (sixpack, Gage R&R, four-in-one) layouts.
//!
//! Everything is deterministic: the same input gives the same bytes.

pub mod core;
pub mod plots;
pub mod render;
pub mod render_3d;

pub use core::{Scene, Theme};
pub use plots::{render_spec, PlotOptions};
