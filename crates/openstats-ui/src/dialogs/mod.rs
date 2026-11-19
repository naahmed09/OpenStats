//! Menus and dialogs. Every statistical dialog is generated from a [`DialogSpec`]; the
//! [`Registry`] binds each one to a routine of `openstats-stats`.

pub mod analyses;
pub(crate) mod analyses_adv;
pub(crate) mod analyses_anova;
pub(crate) mod analyses_basic;
pub(crate) mod analyses_data;
pub(crate) mod analyses_data2;
pub(crate) mod analyses_doe;
pub(crate) mod analyses_doe2;
pub(crate) mod analyses_graph;
pub(crate) mod analyses_ml;
pub(crate) mod analyses_ml2;
pub(crate) mod analyses_mv;
pub(crate) mod analyses_np;
pub(crate) mod analyses_power;
pub(crate) mod analyses_qual;
pub(crate) mod analyses_qual2;
pub(crate) mod analyses_reg;
pub(crate) mod analyses_regfam;
pub(crate) mod analyses_rel;
pub(crate) mod analyses_rel2;
pub(crate) mod analyses_tables;
pub(crate) mod analyses_ts;
pub(crate) mod analyses_workspace;
pub mod analysis;
pub mod assistant;
pub mod dialog;
pub(crate) mod linked;
pub mod menu;
pub mod spec;

pub use analysis::{Analysis, DialogError, Output, Registry, Res, Runner};
pub use dialog::{DialogEvent, DialogView};
pub use menu::{Command, Entry, Menu, MenuBar, MenuResult, Topic, BAR_HEIGHT};
pub use spec::{validate, validate_info, ColInfo, DialogSpec, Field, FieldKind, Page, Problem, Value, Values, VarKind};
