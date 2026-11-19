//! Quality completion (Phase 5m): acceptance sampling, run chart, multi-vari, attribute capability,
//! Box-Cox interval. References come from `tests/gen/qual_ref.py` (SciPy brute force and explicit loops).

mod common;
use common::qual_data::*;
use openstats_stats::acceptance::*;
use openstats_stats::attrcap::{binomial_capability, poisson_capability};
use openstats_stats::runchart::{run_chart, Center};
use openstats_stats::spc::transform::{box_cox_interval, box_cox_profile, BoxCox, LambdaChoice};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

type PlanCase = (&'static [f64], LotModel, Option<u64>, u64, u64, f64, f64);

#[test]
fn attribute_plan_search_matches_brute_force() {
    let cases: [PlanCase; 4] = [
        (PLAN_B1_IN, LotModel::Binomial, None, PLAN_B1_N, PLAN_B1_C, PLAN_B1_PA_AQL, PLAN_B1_PA_RQL),
        (PLAN_B2_IN, LotModel::Binomial, None, PLAN_B2_N, PLAN_B2_C, PLAN_B2_PA_AQL, PLAN_B2_PA_RQL),
        (PLAN_P1_IN, LotModel::Poisson, None, PLAN_P1_N, PLAN_P1_C, PLAN_P1_PA_AQL, PLAN_P1_PA_RQL),
        (PLAN_H1_IN, LotModel::Hypergeometric, Some(1000), PLAN_H1_N, PLAN_H1_C, PLAN_H1_PA_AQL, PLAN_H1_PA_RQL),
    ];
    for (inp, model, lot, n, c, pa_a, pa_r) in cases {
        let s = find_attribute_plan(inp[0], inp[1], inp[2], inp[3], model, lot).unwrap();
        assert_eq!((s.plan.n, s.plan.c), (n, c), "{model:?}");
        close(s.pa_aql, pa_a, 1e-10);
        close(s.pa_rql, pa_r, 1e-10);
        assert!(s.pa_aql >= 1.0 - inp[2] && s.pa_rql <= inp[3]);
    }
}

#[test]
fn oc_aoq_ati_match_scipy() {
    let plan = AttributePlan { n: 80, c: 2 };
    for (i, &p) in OC_P.iter().enumerate() {
        close(plan.accept_probability(LotModel::Binomial, None, p).unwrap(), OC_PA_BIN[i], 1e-11);
        close(plan.accept_probability(LotModel::Poisson, None, p).unwrap(), OC_PA_POIS[i], 1e-11);
        close(plan.accept_probability(LotModel::Hypergeometric, Some(1000), p).unwrap(), OC_PA_HYP[i], 1e-10);
        close(plan.aoq(LotModel::Hypergeometric, Some(1000), p).unwrap(), OC_AOQ_HYP[i], 1e-10);
        close(plan.ati(LotModel::Hypergeometric, Some(1000), p).unwrap().unwrap(), OC_ATI_HYP[i], 1e-10);
    }
    assert!(plan.ati(LotModel::Binomial, None, 0.01).unwrap().is_none());
    let (v, at) = aoql(&plan, LotModel::Binomial, Some(1000)).unwrap();
    close(v, AOQL_BIN, 1e-6);
    close(at, AOQL_BIN_AT, 1e-3);
}

#[test]
fn double_plan_matches_nested_sums() {
    let plan = DoublePlan { n1: 50, c1: 1, r1: 4, n2: 100, c2: 4 };
    for (i, &p) in DBL_P.iter().enumerate() {
        let o = plan.outcome(LotModel::Binomial, p).unwrap();
        close(o.accept_first, DBL_PA1[i], 1e-11);
        close(o.accept_second, DBL_PA2[i], 1e-11);
        close(o.second_sample, DBL_SECOND[i], 1e-11);
        close(o.asn, DBL_ASN[i], 1e-11);
        close(plan.aoq(LotModel::Binomial, Some(2000), p).unwrap(), DBL_AOQ[i], 1e-11);
        close(plan.ati(LotModel::Binomial, Some(2000), p).unwrap().unwrap(), DBL_ATI[i], 1e-11);
    }
}

#[test]
fn variables_plans_match_scipy() {
    let k = find_variables_plan(VAR_IN[0], VAR_IN[1], VAR_IN[2], VAR_IN[3], true).unwrap();
    assert_eq!(k.plan.n, VAR_KNOWN_N);
    close(k.plan.k, VAR_KNOWN_K, 1e-12);
    close(k.pa_rql, VAR_KNOWN_PA_RQL, 1e-10);
    let u = find_variables_plan(VAR_IN[0], VAR_IN[1], VAR_IN[2], VAR_IN[3], false).unwrap();
    assert_eq!(u.plan.n, VAR_UNK_N);
    close(u.plan.k, VAR_UNK_K, 1e-8);
    close(u.pa_rql, VAR_UNK_PA_RQL, 1e-8);
    assert!(u.plan.n > k.plan.n);
    let known = VariablesPlan { n: 12, k: 1.9, sigma_known: true };
    let unk = VariablesPlan { n: 20, k: 1.8, sigma_known: false };
    for (i, &p) in VAR_P.iter().enumerate() {
        close(known.accept_probability(p).unwrap(), VAR_OC_KNOWN[i], 1e-11);
        close(unk.accept_probability(p).unwrap(), VAR_OC_UNK[i], 1e-8);
    }
}

#[test]
fn variables_decision() {
    let plan = VariablesPlan { n: 5, k: 1.5, sigma_known: false };
    let data = [10.0, 10.2, 9.9, 10.1, 9.8];
    let d = decide_variables(&plan, &data, 11.0, LimitSide::Upper, None).unwrap();
    close(d.mean, 10.0, 1e-12);
    assert!(d.accept && d.index > 1.5);
    let d2 = decide_variables(&plan, &data, 10.2, LimitSide::Upper, None).unwrap();
    assert!(!d2.accept);
    let lower = decide_variables(&plan, &data, 9.0, LimitSide::Lower, None).unwrap();
    assert!(lower.accept);
    let known = VariablesPlan { n: 5, k: 1.5, sigma_known: true };
    assert!(decide_variables(&known, &data, 11.0, LimitSide::Upper, None).is_err());
    assert!(decide_variables(&known, &data, 11.0, LimitSide::Upper, Some(0.1)).unwrap().accept);
}

#[test]
fn run_chart_matches_loops() {
    let r = run_chart(RUN_X, Center::Median).unwrap();
    close(r.center, RUN_CENTER, 1e-12);
    assert_eq!(r.runs_about as u64, RUN_RUNS);
    assert_eq!(r.longest_about as u64, RUN_LONGEST);
    close(r.expected_about, RUN_EXP, 1e-12);
    close(r.p_clustering, RUN_P_CLUSTER, 1e-10);
    close(r.p_mixtures, RUN_P_MIX, 1e-10);
    assert_eq!(r.runs_updown as u64, RUN_UD);
    assert_eq!(r.longest_updown as u64, RUN_UD_LONGEST);
    close(r.expected_updown, RUN_UD_EXP, 1e-12);
    close(r.p_trends, RUN_P_TREND, 1e-10);
    close(r.p_oscillation, RUN_P_OSC, 1e-10);
    assert!(r.on >= 1);
    let sec = r.section("X", "Median");
    assert_eq!(sec.blocks.len(), 4);
}

#[test]
fn binomial_capability_matches_scipy() {
    let c = binomial_capability(BIN_D, BIN_N, 0.95, Some(0.02)).unwrap();
    close(c.p_bar, BIN_P, 1e-12);
    close(c.p_interval.0, BIN_LO, 1e-9);
    close(c.p_interval.1, BIN_HI, 1e-9);
    close(c.z, BIN_Z, 1e-9);
    close(c.z_interval.0, BIN_Z_LO, 1e-8);
    close(c.z_interval.1, BIN_Z_HI, 1e-8);
    let (chi, df, p) = c.chi_square.unwrap();
    close(chi, BIN_CHI, 1e-10);
    assert_eq!(df, 14.0);
    close(p, BIN_CHI_P, 1e-9);
    assert!(c.section("Defects").blocks.len() >= 5);
}

#[test]
fn poisson_capability_matches_scipy() {
    let c = poisson_capability(POI_D, POI_U, 0.95, None).unwrap();
    close(c.dpu, POI_DPU, 1e-12);
    close(c.dpu_interval.0, POI_LO, 1e-9);
    close(c.dpu_interval.1, POI_HI, 1e-9);
    close(c.min_dpu, POI_MIN, 1e-12);
    close(c.max_dpu, POI_MAX, 1e-12);
    close(c.z, POI_Z, 1e-9);
    let (chi, df, p) = c.chi_square.unwrap();
    close(chi, POI_CHI, 1e-10);
    assert_eq!(df, 11.0);
    close(p, POI_CHI_P, 1e-9);
}

#[test]
fn box_cox_interval_matches_scipy() {
    let fit = BoxCox::fit(BC_X, LambdaChoice::Optimal).unwrap();
    close(fit.lambda, BC_LAMBDA, 1e-5);
    let (lo, hi) = box_cox_interval(BC_X, 0.95).unwrap();
    close(lo, BC_LO, 1e-5);
    close(hi, BC_HI, 1e-5);
    let ll = box_cox_profile(BC_X, BC_GRID).unwrap();
    for (a, b) in ll.iter().zip(BC_LLF) {
        close(*a, *b, 1e-10);
    }
    assert!(box_cox_interval(BC_X, 1.5).is_err());
    assert!(box_cox_profile(&[1.0, -2.0, 3.0], &[0.0]).is_err());
}
