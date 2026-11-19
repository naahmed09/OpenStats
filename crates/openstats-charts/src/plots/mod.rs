//! Plot builders: `ChartSpec` → [`Scene`], and typed builders for the richer plots.

pub mod basic;
pub mod common;
pub mod extra;
pub mod quality;
pub mod relation;

use crate::core::hit::Viewport;
use crate::core::scene::Scene;
use crate::core::style::Theme;
use openstats_stats::report::{ChartKind, ChartSpec};

pub use basic::{
    bar_chart, boxplot_groups, ecdf_plot, histogram_from_data, histogram_with_curves, individual_value_plot,
    interval_plot, pie_chart, probability_plot, BoxGroup, HistogramOverlay, IntervalGroup,
};
pub use quality::{
    capability_histogram, control_chart_panel, grid_scene, panels_scene, pareto_chart, ControlChartOptions,
};
pub use relation::{lowess, poly_fit, scatter_plot, Fit, FitBands, ScatterData};

/// Size, theme and viewport of a rendered plot.
#[derive(Debug, Clone)]
pub struct PlotOptions {
    /// Width in pixels.
    pub width: f64,
    /// Height in pixels.
    pub height: f64,
    /// Colours and fonts.
    pub theme: Theme,
    /// Zoom / pan window (data coordinates).
    pub viewport: Viewport,
    /// Draw ±1σ and ±2σ zone lines on control charts.
    pub zones: bool,
    /// Limit width in σ used to place the zone lines (3 for Shewhart limits).
    pub limit_sigma: f64,
}

impl Default for PlotOptions {
    fn default() -> PlotOptions {
        PlotOptions {
            width: 640.0,
            height: 400.0,
            theme: Theme::light(),
            viewport: Viewport::default(),
            zones: false,
            limit_sigma: 3.0,
        }
    }
}

impl PlotOptions {
    /// Same options with another size.
    pub fn sized(mut self, w: f64, h: f64) -> PlotOptions {
        self.width = w;
        self.height = h;
        self
    }
}

/// Render any chart specification of `openstats-stats`.
pub fn render_spec(spec: &ChartSpec, opts: &PlotOptions) -> Scene {
    match spec.kind {
        ChartKind::Histogram => basic::histogram_spec(spec, opts),
        ChartKind::Boxplot => basic::boxplot_spec(spec, opts),
        ChartKind::ProbabilityPlot => basic::probability_spec(spec, opts),
        ChartKind::Bar => basic::bar_spec(spec, opts),
        ChartKind::IntervalPlot => basic::interval_spec(spec, opts),
        ChartKind::Scatter => relation::scatter_spec(spec, opts),
        ChartKind::Line => relation::line_spec(spec, opts),
        ChartKind::Contour => relation::contour_spec(spec, opts),
        ChartKind::Surface => relation::surface_spec(spec, opts),
        ChartKind::Pie => extra::pie_spec(spec, opts),
        ChartKind::Ecdf => extra::ecdf_spec(spec, opts),
        ChartKind::IndividualValue => extra::individual_spec(spec, opts),
        ChartKind::Area => extra::area_spec(spec, opts),
        ChartKind::Bubble => extra::bubble_spec(spec, opts),
        ChartKind::Heatmap => extra::heatmap_spec(spec, opts),
        ChartKind::Dotplot => extra::dotplot_spec(spec, opts),
        ChartKind::Matrix => extra::matrix_spec(spec, opts),
        ChartKind::Marginal => extra::marginal_spec(spec, opts),
        ChartKind::StackedBar => extra::stacked_bar_spec(spec, opts),
        ChartKind::DataProbability => extra::data_probability_spec(spec, opts),
        ChartKind::Scatter3D => extra::scatter3d_spec(spec, opts),
        ChartKind::TimeLine => relation::timeline_spec(spec, opts),
        ChartKind::Pareto => {
            let v = spec.series.first().map(|s| s.y.clone()).unwrap_or_default();
            pareto_chart(&spec.categories, &v, &spec.title, &spec.x_label, &spec.y_label, opts)
        }
    }
}

/// Render several chart specifications as one multi-panel scene (`cols` columns).
pub fn render_panels(specs: &[ChartSpec], cols: usize, title: &str, opts: &PlotOptions) -> Scene {
    quality::panels_scene(specs, cols, title, opts)
}
