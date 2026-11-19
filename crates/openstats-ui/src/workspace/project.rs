//! Items of a project besides worksheets: graphs and Session windows.

use openstats_stats::report::{ChartSpec, SessionReport};

/// Identifier of a graph or session item (unique within a project).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ItemId(pub u64);

/// A graph in the Navigator (`Histogram 1`, `Scatterplot 2`, …).
#[derive(Debug, Clone, PartialEq)]
pub struct GraphItem {
    /// Identifier.
    pub id: ItemId,
    /// Name shown in the Navigator.
    pub name: String,
    /// What to draw.
    pub spec: ChartSpec,
    /// The session item and `(section, block)` the graph came from, when it came from one.
    pub source: Option<(ItemId, usize, usize)>,
    /// How the graph was made, so it can follow changes of the data.
    pub recipe: Option<super::linked::GraphRecipe>,
    /// Redraw automatically when the data behind the graph change.
    pub auto: bool,
    /// Colours, fonts, axis ranges and notes.
    pub style: super::linked::GraphStyle,
    /// Title and axis labels the user set; kept when the graph is redrawn.
    pub labels: [Option<String>; 3],
    /// The worksheet row behind each point of the first series (empty when points cannot be brushed).
    pub rows: Vec<usize>,
    /// Fingerprint of the data the graph was last drawn from.
    pub fingerprint: u64,
}

/// A Session window (`Session 1`, …).
#[derive(Debug, Clone, PartialEq)]
pub struct SessionItem {
    /// Identifier.
    pub id: ItemId,
    /// Name shown in the Navigator.
    pub name: String,
    /// The accumulated output.
    pub report: SessionReport,
}

/// A macro script (`Macro 1`, …) stored in the project.
#[derive(Debug, Clone, PartialEq)]
pub struct MacroItem {
    /// Identifier.
    pub id: ItemId,
    /// Name shown in the Navigator; `%name` calls it from other scripts.
    pub name: String,
    /// The script text.
    pub source: String,
}

/// Default name stem of a graph for a chart kind.
pub fn graph_stem(spec: &ChartSpec) -> &'static str {
    use openstats_stats::report::ChartKind::*;
    match spec.kind {
        Histogram => "Histogram",
        Boxplot => "Boxplot",
        ProbabilityPlot => "Probability Plot",
        Scatter => "Scatterplot",
        TimeLine => "Time Series Plot",
        Pareto => "Pareto Chart",
        Line => {
            if spec.title.starts_with("Time Series") {
                "Time Series Plot"
            } else if spec.title.contains("Chart") {
                "Control Chart"
            } else {
                "Line Plot"
            }
        }
        Bar => "Bar Chart",
        IntervalPlot => "Interval Plot",
        Contour => "Contour Plot",
        Surface => "Surface Plot",
        Pie => "Pie Chart",
        Ecdf => "Empirical CDF",
        IndividualValue => "Individual Value Plot",
        Area => "Area Graph",
        Bubble => "Bubble Plot",
        Heatmap => "Heatmap",
        Dotplot => "Dotplot",
        Matrix => "Matrix Plot",
        Marginal => "Marginal Plot",
        StackedBar => "Bar Chart",
        DataProbability => "Probability Plot",
        Scatter3D => "3D Scatterplot",
    }
}
