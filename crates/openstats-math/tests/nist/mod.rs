//! NIST StRD support: `.dat` parser, model-expression evaluator with
//! forward-mode automatic differentiation, and fixture discovery.

pub mod expr;
pub mod parser;

use std::path::{Path, PathBuf};

/// `log10` relative error, NIST's "LRE": the number of correct significant
/// digits, capped at 15. When the certified value is zero the absolute error
/// is used instead.
pub fn lre(got: f64, cert: f64) -> f64 {
    if !got.is_finite() {
        return 0.0;
    }
    let err = if cert == 0.0 { got.abs() } else { ((got - cert) / cert).abs() };
    if err == 0.0 {
        15.0
    } else {
        (-err.log10()).clamp(0.0, 15.0)
    }
}

/// All `.dat` fixtures, as `(path, official)`; files in `tests/fixtures/nist/`
/// (the official downloads) take precedence over `embedded/` files of the
/// same name.
pub fn discover() -> Vec<(PathBuf, bool)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/nist");
    let mut found: Vec<(PathBuf, bool)> = Vec::new();
    for (dir, official) in [(root.clone(), true), (root.join("embedded"), false)] {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        let mut files: Vec<PathBuf> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("dat")))
            .collect();
        files.sort();
        for f in files {
            let name = f.file_name().map(|n| n.to_ascii_lowercase());
            if found.iter().any(|(p, _)| p.file_name().map(|n| n.to_ascii_lowercase()) == name) {
                continue;
            }
            found.push((f, official));
        }
    }
    found.sort_by(|a, b| a.0.file_name().cmp(&b.0.file_name()));
    found
}
