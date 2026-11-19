//! Colours, palettes, line styles and marker shapes.

/// 8-bit RGBA colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    /// Red.
    pub r: u8,
    /// Green.
    pub g: u8,
    /// Blue.
    pub b: u8,
    /// Alpha (255 = opaque).
    pub a: u8,
}

impl Color {
    /// Opaque colour from 0xRRGGBB.
    pub const fn hex(v: u32) -> Color {
        Color { r: (v >> 16) as u8, g: (v >> 8) as u8, b: v as u8, a: 255 }
    }

    /// Same colour with alpha `a` (0.0–1.0).
    pub fn with_alpha(self, a: f64) -> Color {
        Color { a: (a.clamp(0.0, 1.0) * 255.0).round() as u8, ..self }
    }

    /// Linear blend towards white by `t` (0 = unchanged, 1 = white).
    pub fn lighten(self, t: f64) -> Color {
        let m = |c: u8| (f64::from(c) + (255.0 - f64::from(c)) * t.clamp(0.0, 1.0)).round() as u8;
        Color { r: m(self.r), g: m(self.g), b: m(self.b), a: self.a }
    }

    /// `#rrggbb`.
    pub fn css(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }

    /// Opacity 0.0–1.0.
    pub fn opacity(&self) -> f64 {
        f64::from(self.a) / 255.0
    }

    /// White.
    pub const WHITE: Color = Color::hex(0xFFFFFF);
    /// Black.
    pub const BLACK: Color = Color::hex(0x000000);
    /// Transparent.
    pub const NONE: Color = Color { r: 0, g: 0, b: 0, a: 0 };
}

/// Dash pattern of a stroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dash {
    /// Continuous.
    Solid,
    /// Long dashes.
    Dashed,
    /// Dots.
    Dotted,
    /// Dash-dot.
    DashDot,
}

impl Dash {
    /// On/off lengths in multiples of the line width (empty = solid).
    pub fn pattern(self, width: f64) -> Vec<f64> {
        let w = width.max(1.0);
        match self {
            Dash::Solid => Vec::new(),
            Dash::Dashed => vec![6.0 * w, 3.0 * w],
            Dash::Dotted => vec![w, 2.0 * w],
            Dash::DashDot => vec![6.0 * w, 2.5 * w, w, 2.5 * w],
        }
    }
}

/// Stroke style.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stroke {
    /// Colour.
    pub color: Color,
    /// Width in pixels.
    pub width: f64,
    /// Dash pattern.
    pub dash: Dash,
}

impl Stroke {
    /// Solid stroke.
    pub fn new(color: Color, width: f64) -> Stroke {
        Stroke { color, width, dash: Dash::Solid }
    }

    /// With a dash pattern.
    pub fn dashed(mut self, d: Dash) -> Stroke {
        self.dash = d;
        self
    }
}

/// Marker shapes (distinct in shape as well as colour, for accessibility).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Marker {
    /// Filled circle.
    Circle,
    /// Filled square.
    Square,
    /// Filled diamond.
    Diamond,
    /// Upward triangle.
    TriangleUp,
    /// Downward triangle.
    TriangleDown,
    /// Diagonal cross.
    Cross,
    /// Plus sign.
    Plus,
    /// Hollow circle.
    OpenCircle,
}

/// Colour-blind-safe palette (Okabe–Ito order, tuned for a Minitab-like look:
/// blue first, then vermilion, green, purple, orange, sky blue, grey, black).
#[derive(Debug, Clone, PartialEq)]
pub struct Palette {
    /// Series colours, cycled.
    pub series: Vec<Color>,
    /// Marker shapes, cycled in step with the colours.
    pub markers: Vec<Marker>,
    /// Dash patterns, cycled.
    pub dashes: Vec<Dash>,
}

impl Palette {
    /// The default palette.
    pub fn standard() -> Palette {
        Palette {
            series: vec![
                Color::hex(0x0072B2),
                Color::hex(0xD55E00),
                Color::hex(0x009E73),
                Color::hex(0xCC79A7),
                Color::hex(0xE69F00),
                Color::hex(0x56B4E9),
                Color::hex(0x7F7F7F),
                Color::hex(0x000000),
            ],
            markers: vec![
                Marker::Circle,
                Marker::Square,
                Marker::TriangleUp,
                Marker::Diamond,
                Marker::Cross,
                Marker::TriangleDown,
                Marker::Plus,
                Marker::OpenCircle,
            ],
            dashes: vec![Dash::Solid, Dash::Dashed, Dash::DashDot, Dash::Dotted],
        }
    }

    /// Bright colours for a black background (the high-contrast theme).
    pub fn high_contrast() -> Palette {
        Palette {
            series: vec![
                Color::hex(0x56B4E9),
                Color::hex(0xFFA040),
                Color::hex(0x00E08C),
                Color::hex(0xFF7FC0),
                Color::hex(0xFFE000),
                Color::hex(0x9FD0FF),
                Color::hex(0xFFFFFF),
                Color::hex(0xC0C0C0),
            ],
            ..Palette::standard()
        }
    }

    /// Colour of series `i`.
    pub fn color(&self, i: usize) -> Color {
        self.series[i % self.series.len()]
    }

    /// Marker of series `i`.
    pub fn marker(&self, i: usize) -> Marker {
        self.markers[i % self.markers.len()]
    }

    /// Dash of series `i`.
    pub fn dash(&self, i: usize) -> Dash {
        self.dashes[i % self.dashes.len()]
    }
}

/// Everything about the look of a chart.
#[derive(Debug, Clone, PartialEq)]
pub struct Theme {
    /// Series palette.
    pub palette: Palette,
    /// Background.
    pub background: Color,
    /// Axis, tick and text colour.
    pub ink: Color,
    /// Grid line colour.
    pub grid: Color,
    /// Plot-area background.
    pub plot_background: Color,
    /// Control-limit colour (UCL/LCL, spec limits).
    pub limit: Color,
    /// Centre-line colour.
    pub center: Color,
    /// Special-cause / out-of-control colour.
    pub alert: Color,
    /// Title font size.
    pub title_size: f64,
    /// Axis-label font size.
    pub label_size: f64,
    /// Tick-label font size.
    pub tick_size: f64,
    /// Draw grid lines.
    pub show_grid: bool,
}

impl Theme {
    /// Default light theme.
    pub fn light() -> Theme {
        Theme {
            palette: Palette::standard(),
            background: Color::WHITE,
            ink: Color::hex(0x222222),
            grid: Color::hex(0xDDDDDD),
            plot_background: Color::WHITE,
            limit: Color::hex(0xD62728),
            center: Color::hex(0x2CA02C),
            alert: Color::hex(0xD62728),
            title_size: 15.0,
            label_size: 12.0,
            tick_size: 10.5,
            show_grid: true,
        }
    }

    /// Dark theme.
    pub fn dark() -> Theme {
        Theme {
            background: Color::hex(0x1E1E1E),
            ink: Color::hex(0xE8E8E8),
            grid: Color::hex(0x444444),
            plot_background: Color::hex(0x1E1E1E),
            limit: Color::hex(0xFF6B6B),
            center: Color::hex(0x6BCB77),
            alert: Color::hex(0xFF6B6B),
            ..Theme::light()
        }
    }
}

impl Theme {
    /// High contrast: pure white on pure black, bright series colours, heavier grid.
    pub fn high_contrast() -> Theme {
        Theme {
            palette: Palette::high_contrast(),
            background: Color::BLACK,
            ink: Color::WHITE,
            grid: Color::hex(0x7A7A7A),
            plot_background: Color::BLACK,
            limit: Color::hex(0xFF8080),
            center: Color::hex(0x00FF7F),
            alert: Color::hex(0xFF6060),
            ..Theme::light()
        }
    }
}

impl Default for Theme {
    fn default() -> Theme {
        Theme::light()
    }
}
