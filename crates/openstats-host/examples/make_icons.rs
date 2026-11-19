//! Draws the OpenStats application icon and writes every size a Tauri bundle needs.
//!
//! ```text
//! cargo run -p openstats-host --example make_icons -- openstats-desktop/tauri/src-tauri/icons
//! ```
//!
//! The icon is a rounded blue tile with a white histogram and an orange normal curve, drawn with
//! the project's own scene graph and PNG rasteriser (no image crates). It also writes
//! `icon.ico` (PNG-compressed entries) and `icon.icns` (PNG payloads), both plain container formats.

use openstats_charts::core::{Color, Point, Rect, Scene, Stroke};
use openstats_charts::render::raster::render_png;
use std::f64::consts::PI;
use std::path::Path;

const DESIGN: f64 = 1024.0;

/// The icon on a 1024 × 1024 canvas.
fn design() -> Scene {
    let mut s = Scene::new(DESIGN, DESIGN, Color::WHITE.with_alpha(0.0));
    s.title = "OpenStats".into();
    // Rounded square tile (a polygon with quarter-circle corners).
    // macOS icon grid: the tile fills 824 of 1024 units, leaving a transparent margin.
    let (m, r) = (100.0, 185.0);
    let mut tile = Vec::new();
    for (cx, cy, a0) in [
        (DESIGN - m - r, m + r, -90.0_f64),
        (DESIGN - m - r, DESIGN - m - r, 0.0),
        (m + r, DESIGN - m - r, 90.0),
        (m + r, m + r, 180.0),
    ] {
        for k in 0..=12 {
            let a = (a0 + 90.0 * f64::from(k) / 12.0).to_radians();
            tile.push(Point::new(cx + r * a.cos(), cy + r * a.sin()));
        }
    }
    s.polygon(tile, Some(Color::hex(0x0072B2)), None);
    // Histogram bars: a bell shape sampled at seven bins.
    let (left, right, base) = (200.0, DESIGN - 200.0, DESIGN - 230.0);
    let n = 7;
    let w = (right - left) / n as f64;
    for i in 0..n {
        let z = (i as f64 + 0.5 - n as f64 / 2.0) / 1.9;
        let h = 480.0 * (-0.5 * z * z).exp() + 40.0;
        let bar = Rect::new(left + i as f64 * w + 8.0, base - h, w - 16.0, h);
        s.rect(bar, Some(Color::WHITE), None);
    }
    // The fitted normal curve over the bars.
    let curve: Vec<Point> = (0..=64)
        .map(|k| {
            let t = f64::from(k) / 64.0;
            let z = (t - 0.5) * 2.0 * 2.4;
            let y = base - 40.0 - 500.0 * (-0.5 * z * z).exp() / (2.0 * PI).sqrt() * 2.5;
            Point::new(left + t * (right - left), y)
        })
        .collect();
    s.polyline_gaps(&curve, Stroke::new(Color::hex(0xE69F00), 34.0));
    s
}

fn png(size: u32) -> Vec<u8> {
    render_png(&design(), f64::from(size) / DESIGN)
}

fn ico(sizes: &[u32]) -> Vec<u8> {
    let images: Vec<(u32, Vec<u8>)> = sizes.iter().map(|&s| (s, png(s))).collect();
    let mut out = vec![0, 0, 1, 0];
    out.extend_from_slice(&(images.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * images.len();
    for (s, data) in &images {
        let dim = if *s >= 256 { 0 } else { *s as u8 };
        out.extend_from_slice(&[dim, dim, 0, 0]);
        out.extend_from_slice(&1u16.to_le_bytes()); // colour planes
        out.extend_from_slice(&32u16.to_le_bytes()); // bits per pixel
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += data.len();
    }
    for (_, data) in images {
        out.extend_from_slice(&data);
    }
    out
}

fn icns() -> Vec<u8> {
    // ic07 = 128, ic08 = 256, ic09 = 512, ic10 = 1024 (PNG payloads).
    let parts: Vec<(&[u8; 4], Vec<u8>)> =
        vec![(b"ic07", png(128)), (b"ic08", png(256)), (b"ic09", png(512)), (b"ic10", png(1024))];
    let total: usize = 8 + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = b"icns".to_vec();
    out.extend_from_slice(&(total as u32).to_be_bytes());
    for (kind, data) in parts {
        out.extend_from_slice(kind);
        out.extend_from_slice(&((8 + data.len()) as u32).to_be_bytes());
        out.extend_from_slice(&data);
    }
    out
}

fn main() -> std::io::Result<()> {
    let dir = std::env::args().nth(1).expect("usage: make_icons <output directory>");
    let dir = Path::new(&dir);
    std::fs::create_dir_all(dir)?;
    for (name, size) in [("32x32.png", 32), ("128x128.png", 128), ("128x128@2x.png", 256), ("icon.png", 512)] {
        std::fs::write(dir.join(name), png(size))?;
    }
    std::fs::write(dir.join("icon.ico"), ico(&[16, 24, 32, 48, 64, 128, 256]))?;
    std::fs::write(dir.join("icon.icns"), icns())?;
    println!("wrote icons to {}", dir.display());
    Ok(())
}
