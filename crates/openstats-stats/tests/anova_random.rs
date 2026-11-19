//! Random-factor ANOVA, nested ANOVA and mixed models against NumPy / SciPy
//! references (`tests/gen/anova_ref.py`) and textbook closed forms.

mod common;
use common::anova2_data::*;
use openstats_stats::anova::{balanced_anova, glm_random, nested_anova, RandomOptions};
use openstats_stats::mixed::{mixed_model, MixedMethod, MixedOptions, RandomSpec};
use openstats_stats::model::Data;
use openstats_stats::report::Reportable;

fn names(prefix: &str, ids: &[usize]) -> Vec<String> {
    ids.iter().map(|i| format!("{prefix}{i}")).collect()
}

fn close(a: f64, b: f64, rel: f64, what: &str) {
    assert!((a - b).abs() <= rel * (1.0 + b.abs()), "{what}: {a} vs {b}");
}

#[test]
fn unbalanced_random_ems_matches_dense_projections() {
    let d = Data::new()
        .with_factor("A", &names("a", UB_A))
        .unwrap()
        .with_factor("B", &names("b", UB_B))
        .unwrap()
        .with_numeric("y", UB_Y.to_vec())
        .unwrap();
    let r = glm_random("y ~ A*B", &["B".into()], &d, &RandomOptions::default()).unwrap();
    assert!(!r.balanced);
    for i in 0..3 {
        assert_eq!(r.rows[i].df as f64, UB_DF[i]);
        close(r.rows[i].ss, UB_SS[i], 1e-9, "ss");
        // components: B, A*B, Error
        close(r.rows[i].coefficients[0], UB_COEF_B[i], 1e-8, "coef B");
        close(r.rows[i].coefficients[1], UB_COEF_AB[i], 1e-8, "coef AB");
        assert_eq!(r.rows[i].coefficients[2], 1.0);
    }
    close(r.ss_error, UB_SSE[0], 1e-8, "sse");
    assert!(balanced_anova("y ~ A*B", &["B".into()], &d, &RandomOptions::default()).is_err());
    // Report renders and round-trips.
    let rep = openstats_stats::report::SessionReport::new("t");
    let mut rep = rep;
    rep.push(r.report_section());
    rep.validate().unwrap();
    assert!(openstats_stats::report::SessionReport::from_json(&rep.to_json()).is_ok());
}

#[test]
fn unbalanced_nested_matches_dense_projections() {
    let top = names("t", NS_TOP);
    let mid = names("m", NS_MID);
    let low = names("l", NS_LOW);
    let r = nested_anova(NS_Y, &[("Top", &top), ("Mid", &mid), ("Low", &low)]).unwrap();
    assert!(!r.balanced);
    for i in 0..3 {
        assert_eq!(r.rows[i].df as f64, NS_DF[i]);
        close(r.rows[i].ss, NS_SS[i], 1e-9, "ss");
        for j in 0..3 {
            close(r.rows[i].coefficients[j], NS_COEF[i * 3 + j], 1e-8, "coef");
        }
    }
    close(r.ss_error, NS_SSE[0], 1e-8, "sse");
}

#[test]
fn balanced_nested_coefficients_are_level_counts() {
    // 3 tops, 2 mids per top, 3 lows per mid, 2 replicates: EMS(top) = s2 + 2 s2_low + 6 s2_mid + 12 s2_top.
    let (mut top, mut mid, mut low, mut y) = (vec![], vec![], vec![], vec![]);
    let mut k = 0u32;
    for t in 0..3 {
        for m in 0..2 {
            for l in 0..3 {
                for _ in 0..2 {
                    top.push(format!("t{t}"));
                    mid.push(format!("m{}", t * 2 + m));
                    low.push(format!("l{}", (t * 2 + m) * 3 + l));
                    k += 1;
                    y.push(10.0 + ((k * 37) % 17) as f64 * 0.31 + t as f64);
                }
            }
        }
    }
    let r = nested_anova(&y, &[("Top", &top), ("Mid", &mid), ("Low", &low)]).unwrap();
    assert!(r.balanced);
    let c = &r.rows[0].coefficients;
    assert!((c[0] - 12.0).abs() < 1e-9 && (c[1] - 6.0).abs() < 1e-9 && (c[2] - 2.0).abs() < 1e-9, "{c:?}");
    // Each level is tested against the next one down; the lowest against Error.
    assert_eq!(r.rows[0].denominator, vec![(1, 1.0)]);
    assert_eq!(r.rows[2].denominator, vec![(3, 1.0)]);
    // Variance components solve MS = EMS exactly.
    let ms: Vec<f64> = r.rows.iter().map(|x| x.ms).collect();
    let s2 = r.ms_error;
    let s_low = (ms[2] - s2) / 2.0;
    let s_mid = (ms[1] - s2 - 2.0 * s_low) / 6.0;
    close(r.components[2].estimate, s_low, 1e-9, "low");
    close(r.components[1].estimate, s_mid, 1e-9, "mid");
}

fn two_way(a: usize, b: usize, n: usize) -> Data {
    let (mut fa, mut fb, mut y) = (vec![], vec![], vec![]);
    let mut k = 0u32;
    for i in 0..a {
        for j in 0..b {
            for _ in 0..n {
                k += 1;
                fa.push(format!("a{i}"));
                fb.push(format!("b{j}"));
                y.push(20.0 + 1.5 * i as f64 + 0.8 * j as f64 + ((k * 29) % 13) as f64 * 0.37 + ((i * j) % 3) as f64);
            }
        }
    }
    Data::new().with_factor("A", &fa).unwrap().with_factor("B", &fb).unwrap().with_numeric("y", y).unwrap()
}

#[test]
fn two_way_mixed_unrestricted_and_restricted() {
    let d = two_way(3, 4, 3);
    // A fixed, B random.
    let un = glm_random("y ~ A*B", &["B".into()], &d, &RandomOptions::default()).unwrap();
    let re =
        glm_random("y ~ A*B", &["B".into()], &d, &RandomOptions { restricted: true, ..Default::default() }).unwrap();
    // Unrestricted: F_B = MS_B / MS_AB; restricted: F_B = MS_B / MSE. F_A = MS_A / MS_AB in both.
    assert_eq!(un.rows[1].denominator, vec![(2, 1.0)]);
    assert_eq!(re.rows[1].denominator, vec![(3, 1.0)]);
    assert_eq!(un.rows[0].denominator, vec![(2, 1.0)]);
    assert_eq!(re.rows[0].denominator, vec![(2, 1.0)]);
    close(un.rows[1].f.unwrap(), un.rows[1].ms / un.rows[2].ms, 1e-12, "F_B unrestricted");
    close(re.rows[1].f.unwrap(), re.rows[1].ms / re.ms_error, 1e-12, "F_B restricted");
    // EMS coefficients of the balanced design: A: s2 + 3 s2_AB + 12 Q; B: s2 + (3 s2_AB) + 9 s2_B.
    close(un.rows[1].coefficients[0], 9.0, 1e-9, "B in EMS(B)");
    close(un.rows[1].coefficients[1], 3.0, 1e-9, "AB in EMS(B)");
    assert_eq!(re.rows[1].coefficients[1], 0.0);
    close(un.rows[0].coefficients[1], 3.0, 1e-9, "AB in EMS(A)");
    // Variance components (balanced closed forms, unrestricted).
    let (ms_b, ms_ab, mse) = (un.rows[1].ms, un.rows[2].ms, un.ms_error);
    close(un.components[1].estimate, (ms_ab - mse) / 3.0, 1e-9, "AB");
    close(un.components[0].estimate, (ms_b - ms_ab) / 9.0, 1e-9, "B");
    close(re.components[0].estimate, (ms_b - mse) / 9.0, 1e-9, "B restricted");
    assert!(balanced_anova("y ~ A*B", &["B".into()], &d, &RandomOptions::default()).is_ok());
}

#[test]
fn three_way_synthesized_denominator() {
    // A fixed (2), B and C random (3 each), 2 replicates.
    let (mut fa, mut fb, mut fc, mut y) = (vec![], vec![], vec![], vec![]);
    let mut k = 0u32;
    for i in 0..2 {
        for j in 0..3 {
            for l in 0..3 {
                for _ in 0..2 {
                    k += 1;
                    fa.push(format!("a{i}"));
                    fb.push(format!("b{j}"));
                    fc.push(format!("c{l}"));
                    y.push(5.0 + i as f64 * 0.9 + ((k * 31 + j as u32 * 7 + l as u32 * 3) % 23) as f64 * 0.21);
                }
            }
        }
    }
    let d = Data::new()
        .with_factor("A", &fa)
        .unwrap()
        .with_factor("B", &fb)
        .unwrap()
        .with_factor("C", &fc)
        .unwrap()
        .with_numeric("y", y)
        .unwrap();
    let r = glm_random("y ~ A*B*C", &["B".into(), "C".into()], &d, &RandomOptions::default()).unwrap();
    // Terms: A, B, C, A*B, A*C, B*C, A*B*C -> A is tested against AB + AC - ABC.
    let den = &r.rows[0].denominator;
    assert_eq!(den.len(), 3);
    let w = |row: usize| den.iter().find(|x| x.0 == row).map(|x| x.1).unwrap();
    assert!((w(3) - 1.0).abs() < 1e-9 && (w(4) - 1.0).abs() < 1e-9 && (w(6) + 1.0).abs() < 1e-9, "{den:?}");
    let value = r.rows[3].ms + r.rows[4].ms - r.rows[6].ms;
    close(r.rows[0].f.unwrap(), r.rows[0].ms / value, 1e-10, "F_A");
    let nu = value * value
        / (r.rows[3].ms.powi(2) / r.rows[3].df as f64
            + r.rows[4].ms.powi(2) / r.rows[4].df as f64
            + r.rows[6].ms.powi(2) / r.rows[6].df as f64);
    close(r.rows[0].den_df.unwrap(), nu, 1e-9, "Satterthwaite df");
    assert!(r.rows[0].synthesized);
    assert!(r.denominator_label(0).contains('-'));
    assert!(r.ems_text(0).contains("Q[1]"));
}

fn crossed_data() -> Data {
    Data::new()
        .with_factor("subj", &names("s", MX_S))
        .unwrap()
        .with_factor("item", &names("i", MX_I))
        .unwrap()
        .with_factor("T", &MX_T.iter().map(|&t| if t == 0 { "a" } else { "b" }.to_string()).collect::<Vec<_>>())
        .unwrap()
        .with_numeric("x", MX_X.to_vec())
        .unwrap()
        .with_numeric("y", MX_Y.to_vec())
        .unwrap()
}

#[test]
fn crossed_mixed_model_matches_dense_reml() {
    let d = crossed_data();
    let rs = [RandomSpec::new(&["subj"]), RandomSpec::new(&["item"])];
    let f = mixed_model("y ~ x + T", &rs, &d, &MixedOptions::default()).unwrap();
    assert!(f.converged);
    for k in 0..3 {
        close(f.variances[k].variance, MX_REML_PHI[k], 2e-4, "variance");
    }
    close(-2.0 * f.log_likelihood, MX_REML_M2LL[0], 1e-9, "-2 REML loglik");
    for k in 0..3 {
        close(f.fixed[k].estimate, MX_REML_BETA[k], 1e-6, "beta");
        close(f.fixed[k].se, MX_REML_SE[k], 1e-4, "se");
        close(f.fixed[k].df, MX_REML_DF[k], 2e-2, "Satterthwaite df");
    }
    for k in 0..3 {
        close(f.variances[k].se.unwrap(), MX_REML_SEVAR[k], 2e-2, "variance se");
    }
    // Term tests agree with the squared t statistics for one-df terms.
    for t in &f.tests {
        let c = f.fixed.iter().find(|c| c.name.starts_with(&t.term)).unwrap();
        close(t.f, c.t * c.t, 1e-8, "F = t^2");
        close(t.den_df, c.df, 1e-6, "den df");
    }
    // Conditional fit decomposes the response.
    for (i, r) in f.residuals.iter().enumerate() {
        close(f.fitted[i] + r, MX_Y[i], 1e-9, "fit + resid");
    }
    // ML.
    let ml =
        mixed_model("y ~ x + T", &rs, &d, &MixedOptions { method: MixedMethod::Ml, ..Default::default() }).unwrap();
    for k in 0..3 {
        close(ml.variances[k].variance, MX_ML_PHI[k], 2e-4, "ml variance");
    }
    close(-2.0 * ml.log_likelihood, MX_ML_M2LL[0], 1e-9, "-2 ml loglik");
    for k in 0..3 {
        close(ml.fixed[k].estimate, MX_ML_BETA[k], 1e-6, "ml beta");
    }
    let mut rep = openstats_stats::report::SessionReport::new("m");
    rep.push(f.report_section());
    rep.validate().unwrap();
}

#[test]
fn balanced_one_way_satterthwaite_df_is_groups_minus_one() {
    let g: Vec<String> = (0..20).map(|i| format!("g{}", i / 5)).collect();
    let y: Vec<f64> = (0..20).map(|i| 10.0 + (i / 5) as f64 * 1.3 + ((i * 7) % 11) as f64 * 0.4).collect();
    let d = Data::new().with_factor("g", &g).unwrap().with_numeric("y", y).unwrap();
    let f = mixed_model("y ~ 1", &[RandomSpec::new(&["g"])], &d, &MixedOptions::default()).unwrap();
    close(f.fixed[0].df, 3.0, 5e-3, "df of the grand mean");
}

#[test]
fn errors() {
    let d = two_way(3, 3, 2);
    assert!(glm_random("y ~ A*B", &[], &d, &RandomOptions::default()).is_err());
    assert!(glm_random("y ~ A", &["B".into()], &d, &RandomOptions::default()).is_err());
    assert!(glm_random("y ~ A*B", &["y".into()], &d, &RandomOptions::default()).is_err());
    assert!(mixed_model("y ~ A", &[], &d, &MixedOptions::default()).is_err());
    assert!(mixed_model("y ~ A", &[RandomSpec::new(&["zz"])], &d, &MixedOptions::default()).is_err());
    assert!(nested_anova(&[1.0, 2.0], &[]).is_err());
}

#[test]
fn nested_subjects_keep_the_between_effect() {
    // 2 groups x 4 subjects (unique ids) x 3 times: closed-form split-plot sums of squares.
    let (mut sub, mut grp, mut time, mut y) = (vec![], vec![], vec![], vec![]);
    let mut k = 0.0f64;
    for s in 0..8usize {
        for t in 0..3usize {
            k += 1.0;
            sub.push(format!("p{s}"));
            grp.push(if s < 4 { "ctl" } else { "trt" }.to_string());
            time.push(format!("t{t}"));
            y.push(
                10.0 + (s as f64 * 0.7).sin() * 3.0
                    + t as f64 * 1.5
                    + (k * 1.3).sin()
                    + if s >= 4 { 0.5 * t as f64 } else { 0.0 },
            );
        }
    }
    let d = Data::new()
        .with_factor("Group", &grp)
        .unwrap()
        .with_factor("Time", &time)
        .unwrap()
        .with_factor("Subject", &sub)
        .unwrap()
        .with_numeric("y", y.clone())
        .unwrap();
    let r = glm_random("y ~ Group*Time + Subject(Group)", &["Subject".into()], &d, &RandomOptions::default()).unwrap();
    let mean = |idx: &[usize]| idx.iter().map(|&i| y[i]).sum::<f64>() / idx.len() as f64;
    let all: Vec<usize> = (0..24).collect();
    let gm = mean(&all);
    let g_idx = |g: usize| -> Vec<usize> { (0..24).filter(|&i| (i / 12) == g).collect() };
    let ss_group: f64 = (0..2).map(|g| 12.0 * (mean(&g_idx(g)) - gm).powi(2)).sum();
    let subj_mean = |s: usize| mean(&(3 * s..3 * s + 3).collect::<Vec<_>>());
    let ss_subj: f64 = (0..8).map(|s| 3.0 * (subj_mean(s) - mean(&g_idx(s / 4))).powi(2)).sum();
    let t_idx = |t: usize| -> Vec<usize> { (0..24).filter(|&i| i % 3 == t).collect() };
    let ss_time: f64 = (0..3).map(|t| 8.0 * (mean(&t_idx(t)) - gm).powi(2)).sum();
    let by = |g: usize, t: usize| -> Vec<usize> { (0..24).filter(|&i| i / 12 == g && i % 3 == t).collect() };
    let ss_gt: f64 = (0..2)
        .flat_map(|g| (0..3).map(move |t| (g, t)))
        .map(|(g, t)| 4.0 * (mean(&by(g, t)) - mean(&g_idx(g)) - mean(&t_idx(t)) + gm).powi(2))
        .sum();
    // Terms: Group, Time, Subject(Group), Group*Time.
    assert_eq!(r.rows.iter().map(|x| x.df).collect::<Vec<_>>(), vec![1, 2, 6, 2]);
    close(r.rows[0].ss, ss_group, 1e-9, "Group");
    close(r.rows[1].ss, ss_time, 1e-9, "Time");
    close(r.rows[2].ss, ss_subj, 1e-9, "Subject(Group)");
    close(r.rows[3].ss, ss_gt, 1e-9, "Group*Time");
    assert_eq!(r.rows[0].denominator, vec![(2, 1.0)]);
    assert_eq!(r.rows[1].denominator, vec![(4, 1.0)]);
    close(r.rows[0].coefficients[0], 3.0, 1e-9, "EMS(Group) subjects term");
}

#[test]
fn gage_fixed_operator_scales_the_operator_component() {
    use openstats_stats::msa::{CrossedStudy, GageOptions, Interaction, Source};
    let (mut m, mut p, mut o) = (vec![], vec![], vec![]);
    let mut k = 0u32;
    for part in 0..6 {
        for op in 0..3 {
            for _ in 0..3 {
                k += 1;
                p.push(format!("p{part}"));
                o.push(format!("o{op}"));
                m.push(20.0 + part as f64 * 1.7 + op as f64 * 0.6 + ((k * 41) % 19) as f64 * 0.05);
            }
        }
    }
    let s = CrossedStudy::from_stacked(&m, &p, &o).unwrap();
    let base = GageOptions { interaction: Interaction::Keep, ..GageOptions::default() };
    let random = s.anova(&base).unwrap();
    let fixed = s.anova(&GageOptions { operator_fixed: true, ..base }).unwrap();
    let (vr, vf) = (random.component(Source::Operator).unwrap().var, fixed.component(Source::Operator).unwrap().var);
    assert!(vr > 0.0);
    close(vf, vr * 2.0 / 3.0, 1e-12, "fixed operator component");
    // Tests and the other components are unchanged.
    assert_eq!(random.anova, fixed.anova);
    close(
        random.component(Source::PartToPart).unwrap().var,
        fixed.component(Source::PartToPart).unwrap().var,
        1e-12,
        "part",
    );
    assert!(fixed.operator_fixed && !random.operator_fixed);
    let mut rep = openstats_stats::report::SessionReport::new("t");
    rep.push(fixed.report_section());
    let text = rep.render_text();
    assert!(text.contains("Operators are a fixed factor"));
}
