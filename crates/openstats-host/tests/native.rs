//! The native-shell layer: frames, actions, menus and text painters, with no window system.

use openstats_charts::render::raster::{TextPainter, TextRun};
use openstats_host::keys::SPECIAL;
use openstats_host::native::{parse_menu, to_xrgb, Action, MenuNode, NativeApp, NativeGraph};

const CTRL: u32 = 2;

/// Counts the runs it is asked for and draws every one as a solid 2 × 2 block.
#[derive(Default)]
struct Counting {
    runs: Vec<String>,
}

impl TextPainter for Counting {
    fn render(&mut self, text: &str, _size: f32, _bold: bool) -> Option<TextRun> {
        self.runs.push(text.to_string());
        Some(TextRun {
            advance: 6.0 * text.chars().count() as f32,
            left: 0,
            top: 2,
            width: 2,
            height: 2,
            coverage: vec![255; 4],
        })
    }
}

fn type_text(app: &mut NativeApp, t: &str) {
    for c in t.chars() {
        app.key(c as u32, 0);
    }
}

#[test]
fn frames_follow_size_and_scale_and_use_the_painter() {
    let mut app = NativeApp::new(800.0, 600.0, 2.0);
    let img = app.frame(None);
    assert_eq!((img.width, img.height), (1600, 1200));
    let mut p = Counting::default();
    let _ = app.frame(Some(&mut p));
    for expected in ["File", "Stat", "Project"] {
        assert!(
            p.runs.iter().any(|r| r == expected),
            "no text run for {expected}: {:?}",
            &p.runs[..p.runs.len().min(12)]
        );
    }
    app.resize(500.0, 400.0, 1.0);
    let img = app.frame(None);
    assert_eq!((img.width, img.height), (500, 400));
    // Nonsense scale factors fall back to 1.
    app.resize(500.0, 400.0, f64::NAN);
    assert_eq!(app.scale(), 1.0);
}

#[test]
fn xrgb_conversion_packs_and_crops() {
    let app = NativeApp::new(40.0, 30.0, 1.0);
    let img = app.frame(None);
    let mut buf = vec![0xDEAD_BEEF_u32; 50 * 35];
    to_xrgb(&img, &mut buf, 50, 35);
    let i = 4 * (5 * img.width + 7);
    let want = (u32::from(img.rgba[i]) << 16) | (u32::from(img.rgba[i + 1]) << 8) | u32::from(img.rgba[i + 2]);
    assert_eq!(buf[5 * 50 + 7], want);
    assert_eq!(buf[34 * 50 + 49], 0, "outside the image is black");
    assert!(buf.iter().all(|&p| p >> 24 == 0), "the top byte stays clear");
}

#[test]
fn menus_parse_and_report_changes_once() {
    let mut app = NativeApp::new(800.0, 600.0, 1.0);
    let menu = app.menu_if_changed().expect("first call returns the menu");
    let labels: Vec<_> = menu
        .iter()
        .map(|m| match m {
            MenuNode::Submenu { label, .. } => label.as_str(),
            _ => "?",
        })
        .collect();
    assert_eq!(&labels[..2], ["File", "Edit"]);
    assert!(labels.contains(&"View") && labels.contains(&"Stat"));
    assert!(app.menu_if_changed().is_none(), "unchanged menu is not re-sent");
    // Undo becomes enabled after an edit, which is a change.
    type_text(&mut app, "5");
    app.key(SPECIAL + 8, 0);
    assert!(app.menu_if_changed().is_some());

    let find = |nodes: &[MenuNode], id: &str| -> Option<bool> {
        fn walk(nodes: &[MenuNode], id: &str) -> Option<bool> {
            nodes.iter().find_map(|n| match n {
                MenuNode::Item { id: i, enabled, .. } if i == id => Some(*enabled),
                MenuNode::Submenu { entries, .. } => walk(entries, id),
                _ => None,
            })
        }
        walk(nodes, id)
    };
    let m = parse_menu(&app.host().menu_json()).unwrap();
    assert_eq!(find(&m, "edit.undo"), Some(true));
    assert_eq!(find(&m, "file.save"), Some(true));
    assert!(parse_menu("{}").is_err() && parse_menu("[{\"entries\":[]}]").is_err());
}

#[test]
fn titles_are_reported_when_they_change() {
    let mut app = NativeApp::new(800.0, 600.0, 1.0);
    assert_eq!(app.title_if_changed().as_deref(), Some("Untitled - OpenStats"));
    assert!(app.title_if_changed().is_none());
}

#[test]
fn commands_become_actions() {
    let mut app = NativeApp::new(800.0, 600.0, 1.0);
    type_text(&mut app, "42");
    app.key(SPECIAL + 8, 0);
    // Save: a project file to write.
    match &app.command("file.save")[..] {
        [Action::SaveFile { name, bytes }] => {
            assert!(name.ends_with(".otb"), "{name}");
            assert!(bytes.len() > 20);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(app.command("file.open"), vec![Action::PickProject]);
    assert_eq!(app.command("file.import"), vec![Action::PickData]);
    // Ctrl+C on a selected cell puts text on the clipboard.
    app.key(SPECIAL, 0); // Up: back onto the cell holding 42
    let acts = app.key('c' as u32, CTRL);
    assert!(matches!(&acts[..], [Action::Clipboard(t)] if t.trim() == "42"), "{acts:?}");
    // Print produces PNG pages.
    match &app.command("file.print")[..] {
        [Action::SavePages { pages, .. }] => {
            assert!(!pages.is_empty());
            assert_eq!(&pages[0][1..4], b"PNG");
        }
        other => panic!("{other:?}"),
    }
    assert!(app.command("no.such.command").is_empty());
}

#[test]
fn import_open_and_tear_off_round_trip() {
    let mut app = NativeApp::new(1000.0, 700.0, 1.0);
    assert!(app.import_data("data.csv", b"x,y\n1,2\n2,4.1\n3,5.9\n4,8.2\n5,9.9\n6,12.1\n").is_empty());
    let saved = match app.command("file.save").pop() {
        Some(Action::SaveFile { bytes, .. }) => bytes,
        other => panic!("{other:?}"),
    };
    let mut again = NativeApp::new(1000.0, 700.0, 1.0);
    assert!(again.open_project(&saved).is_empty());
    let mut p = Counting::default();
    let _ = again.frame(Some(&mut p));
    assert!(p.runs.iter().any(|r| r == "x") && p.runs.iter().any(|r| r == "y"), "headers drawn after reopening");
    // A bad project file is an error action, not a panic.
    assert!(matches!(&again.open_project(b"not a project")[..], [Action::Error(_)]));
}

#[test]
fn graph_windows_render_from_tear_off_json() {
    let mut app = NativeApp::new(1200.0, 800.0, 1.0);
    let h = app.host();
    let mut wb = openstats_core::Workbook::new();
    wb.worksheets[0]
        .add_column(openstats_core::column::Column::numeric("x", (0..20).map(f64::from).collect()))
        .unwrap();
    *h.workspace_mut().grid_mut() = openstats_ui::grid::GridController::new(wb, 800.0, 300.0);
    h.workspace_mut().open_dialog("graph.histogram").unwrap();
    h.workspace_mut().dialog_mut().unwrap().select_column(0);
    h.workspace_mut().dialog_mut().unwrap().focus_field("vars");
    h.workspace_mut().dialog_mut().unwrap().add_selected();
    h.workspace_mut().run_dialog();
    let id = h.workspace().graphs()[0].id;
    let json = h.workspace().graph_json(id).unwrap();
    let g = NativeGraph::from_json(&json).unwrap();
    assert_eq!(g.name(), "Histogram 1");
    let mut p = Counting::default();
    let img = g.frame(400.0, 300.0, 2.0, Some(&mut p));
    assert_eq!((img.width, img.height), (800, 600));
    assert!(img.rgba.chunks(4).any(|px| px[..3] != img.rgba[..3]), "not blank");
    assert!(!p.runs.is_empty(), "axis text goes through the painter");
    assert!(NativeGraph::from_json("{").is_err());
}

#[test]
fn themes_can_be_saved_and_restored_by_name() {
    let mut app = NativeApp::new(800.0, 600.0, 1.0);
    assert_eq!(app.theme_name(), "light");
    assert!(app.set_theme_by_name("high-contrast"));
    assert_eq!(app.theme_name(), "high-contrast");
    let img = app.frame(None);
    let gap = 4 * (100 * img.width + 212); // between the Navigator and the Session pane
    assert_eq!(&img.rgba[gap..gap + 3], &[0, 0, 0], "the desk is black");
    assert!(!app.set_theme_by_name("sepia"));
    assert_eq!(app.theme_name(), "high-contrast", "unknown names change nothing");
    assert!(app.set_theme_by_name(" dark\n"));
    assert_eq!(app.theme_name(), "dark");
    // Choosing a theme from the menu is reflected too.
    app.command("view.theme_light");
    assert_eq!(app.theme_name(), "light");
}

#[test]
fn identical_frames_are_not_redrawn() {
    let mut app = NativeApp::new(800.0, 600.0, 2.0);
    assert!(app.frame_if_changed(None).is_some(), "the first frame is drawn");
    assert!(app.frame_if_changed(None).is_none(), "nothing changed");
    // Pointer moves over empty space, a cancelled input method and repeated asks change nothing.
    app.pointer(1, 700.0, 500.0, 0, 0, 0);
    app.ime_cancel();
    assert!(app.frame_if_changed(None).is_none());
    // A real change, a resize and a new scale each draw once.
    app.key('7' as u32, 0);
    assert!(app.frame_if_changed(None).is_some());
    assert!(app.frame_if_changed(None).is_none());
    app.resize(700.0, 500.0, 2.0);
    assert!(app.frame_if_changed(None).is_some());
    app.resize(700.0, 500.0, 1.0);
    assert!(app.frame_if_changed(None).is_some());
    assert!(app.frame_if_changed(None).is_none());
    app.invalidate();
    assert!(app.frame_if_changed(None).is_some(), "invalidate forces a redraw");
}
