//! Scene graph, scales, ticks, fonts, styles, layout and interaction metadata.

pub mod font;
pub mod geom;
pub mod hit;
pub mod layout;
pub mod scale;
pub mod scene;
pub mod style;
mod tables;
pub mod ticks;

pub use font::{text_width, FontMetrics};
pub use geom::{Point, Rect};
pub use hit::{HitIndex, HitResult, HitTarget, Viewport};
pub use layout::{Frame, FrameSpec, LegendEntry};
pub use scale::{ProbabilityKind, Scale};
pub use scene::{Anchor, Item, Scene, Shape, TextStyle};
pub use style::{Color, Dash, Marker, Palette, Stroke, Theme};
pub use ticks::{extended_ticks, log_ticks, time_ticks, Tick};
