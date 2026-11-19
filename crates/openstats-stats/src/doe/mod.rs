//! Design of experiments.

pub mod design;
pub mod factorial;
pub mod hadamard;

pub use design::{Defining, Design, DesignKind, FactorSpec, PointType};
pub use factorial::{
    default_generators, fractional_factorial, full_factorial, plackett_burman, AliasStructure, FactorialOptions,
};
pub mod analysis;
pub mod model;

pub use analysis::{
    analyze_factorial, lenth_pse, AnovaGroup, EffectRow, FactorialAnalysis, FactorialAnalysisOptions, Lenth,
};
pub use model::{PolyTerm, PolynomialModel};
pub mod plots;
pub mod report_impl;

pub use plots::{
    contour_chart, cube_chart, interaction_charts, main_effects_charts, normal_effects_chart, pareto_chart,
    surface_chart, SurfacePlot,
};
pub mod rsm;

pub use rsm::{
    analyze_surface, box_behnken, canonical_analysis, central_composite, AlphaRule, Canonical, CcdKind, CcdOptions,
    StationaryKind, SurfaceAnalysis, SurfaceAnovaRow, SurfaceModel, SurfaceOptions,
};
pub mod taguchi;

pub use taguchi::{
    analyze_taguchi, available_arrays, orthogonal_array, OrthogonalArray, ResponseTable, SignalToNoise,
    TaguchiAnalysis, TaguchiAnovaRow, TaguchiDesign,
};

pub mod optimize;
pub mod response_optimizer;

pub use optimize::{
    composite_desirability, optimize_responses, Goal, OptimizationResult, OptimizeOptions, ResponseGoal, Solution,
};

pub use response_optimizer::{
    design_from_columns, fit_responses, model_summary_section, parse_goal, FittedResponse, ModelOrder,
};

pub mod extra;

pub use extra::{
    extreme_vertices, fit_mixture, optimal_design, simplex_centroid, simplex_lattice, split_plot, Criterion,
    MixtureAnovaRow, MixtureDesign, MixtureFit, MixtureModel, OptModel, OptimalDesign, OptimalOptions, SplitPlot,
    SplitPlotRun,
};

pub mod extra2;

pub use extra2::{
    analyze_variability, conference_matrix, conference_orders, definitive_screening, overlaid_contour, steepest_path,
    DsdDesign, Overlay, OverlayResponse, PathPoint, VariabilityAnalysis, VariabilityTerm,
};
