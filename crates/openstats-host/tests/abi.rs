//! The C ABI driven exactly as a shell drives it (through the memory stand-ins).

use openstats_charts::render::canvas::{from_binary, Command};
use openstats_host::abi::{self, memory};
use openstats_host::keys::{self, SPECIAL};
use openstats_host::{GraphHost, Host};
use openstats_ui::grid::Key;

fn put(text: &str) -> u32 {
    memory::write_input(text.as_bytes());
    text.len() as u32
}

fn run_command(id: &str) -> u32 {
    let n = put(id);
    abi::ot_command(n)
}

fn out_string() -> String {
    String::from_utf8(memory::read_out()).unwrap()
}

fn texts(frame: &[u8]) -> Vec<String> {
    from_binary(frame)
        .expect("valid frame")
        .into_iter()
        .filter_map(|c| if let Command::Text { text, .. } = c { Some(text) } else { None })
        .collect()
}

fn frame() -> Vec<u8> {
    assert!(abi::ot_render() > 0);
    memory::read_out()
}

fn drain() -> Vec<u32> {
    let mut v = vec![];
    loop {
        let k = abi::ot_outcome_next();
        if k == 0 {
            return v;
        }
        v.push(k);
    }
}

#[test]
fn key_codes_round_trip() {
    for (i, name) in keys::NAMES.iter().enumerate() {
        let k = keys::decode(SPECIAL + i as u32).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(keys::encode(k), SPECIAL + i as u32);
    }
    assert_eq!(keys::decode('é' as u32), Some(Key::Char('é')));
    assert_eq!(keys::decode(SPECIAL + 99), None);
    assert_eq!(keys::decode(0xD800), None, "surrogates are not scalar values");
    assert_eq!(keys::encode(Key::Char('✓')), '✓' as u32);
}

#[test]
fn a_full_session_through_the_abi() {
    assert_eq!(abi::ot_abi_version(), 1);
    assert_eq!(abi::ot_init(1200.0, 800.0), 1);
    // Frame: a valid canvas stream with the menu bar and the Navigator.
    let t = texts(&frame());
    for label in ["File", "Edit", "Data", "Calc", "Stat", "Graph", "Assistant", "Worksheet 1", "Session 1"] {
        assert!(t.iter().any(|x| x == label), "{label} missing from {t:?}");
    }
    // Import a CSV through the file path a shell would use.
    let name = "heights.csv";
    let data = "h,w\n1.5,60\n1.7,72\n1.6,65\n1.8,80\n1.75,70\n";
    memory::write_input(format!("{name}{data}").as_bytes());
    abi::ot_import(name.len() as u32, (name.len() + data.len()) as u32);
    assert!(drain().is_empty());
    assert!(texts(&frame()).iter().any(|x| x == "heights"), "the new worksheet is in the Navigator");

    // Open a dialog by command id, fill nothing, press Enter: the error stays in the dialog.
    assert_eq!(run_command("dialog:stat.basic.descriptive"), 1);
    assert_eq!(run_command("no.such.command"), 0);
    assert!(texts(&frame()).iter().any(|x| x.contains("Descriptive")));
    abi::ot_key(SPECIAL + 8, 0);
    assert!(texts(&frame()).iter().any(|x| x.contains("Descriptive")), "dialog still open");
    // Choose the first column with a click on the list, add with Space, press Enter.
    let (dw, dh) = (openstats_ui::dialogs::dialog::WIDTH, openstats_ui::dialogs::dialog::HEIGHT);
    let (ox, oy) = ((1200.0 - dw) / 2.0, (800.0 - dh) / 2.0);
    abi::ot_pointer(0, ox + 30.0, oy + 49.0, 0, 0, 1);
    abi::ot_pointer(2, ox + 30.0, oy + 49.0, 0, 0, 1);
    abi::ot_key(' ' as u32, 0);
    abi::ot_key(SPECIAL + 8, 0);
    let t = texts(&frame());
    assert!(!t.iter().any(|x| x == "Select >>"), "dialog closed after OK");
    assert!(t.iter().any(|x| x.contains("Descriptive Statistics")), "output in the Session window: {t:?}");

    // Edit a cell by typing, with an IME commit, and read the candidate anchor.
    abi::ot_pointer(0, 400.0, 520.0, 0, 0, 1);
    abi::ot_pointer(2, 400.0, 520.0, 0, 0, 1);
    abi::ot_key('7' as u32, 0);
    assert_eq!(abi::ot_ime_anchor(), 1);
    assert!(abi::ot_ime_x().is_finite() && abi::ot_ime_y().is_finite() && abi::ot_ime_h() > 0.0);
    let n = put("か");
    abi::ot_ime_update(n, 1);
    let n = put("か");
    abi::ot_ime_commit(n);
    abi::ot_key(SPECIAL + 10, 0);
    assert_eq!(abi::ot_ime_anchor(), 0);
    assert!(abi::ot_ime_x().is_nan());
    abi::ot_ime_cancel();

    // Save, then open into a fresh window; ctrl+S arrives as a file outcome.
    abi::ot_key('s' as u32, 2);
    let kinds = drain();
    assert_eq!(kinds, vec![1]);
    let n = abi::ot_save_project() as usize;
    let bytes = memory::read_out();
    assert_eq!(bytes.len(), n);
    assert!(bytes.starts_with(openstats_core::io::otb::MAGIC), "project magic");
    assert_eq!(abi::ot_init(900.0, 600.0), 1);
    memory::write_input(&bytes);
    abi::ot_open_project(bytes.len() as u32);
    assert!(drain().is_empty());
    let t = texts(&frame());
    assert!(t.iter().any(|x| x.contains("Descriptive Statistics")) && t.iter().any(|x| x == "heights"));
}

#[test]
fn outcomes_files_clipboard_print_errors_and_dialog_requests() {
    abi::ot_init(1000.0, 700.0);
    run_command("file.open");
    run_command("file.import");
    run_command("edit.select_all");
    assert_eq!(drain(), vec![4, 5]);
    run_command("file.export_worksheet");
    assert_eq!(abi::ot_outcome_next(), 1);
    assert_eq!(String::from_utf8(memory::read_slot(0)).unwrap(), "Worksheet 1.csv");
    assert_eq!(String::from_utf8(memory::read_slot(1)).unwrap(), "text/csv");
    assert_eq!(abi::ot_slot_len(0), 15);
    assert_eq!(abi::ot_slot_len(2) as usize, memory::read_slot(2).len());
    assert_eq!(abi::ot_slot_len(9), 0, "bad slot");
    assert_eq!(abi::ot_outcome_next(), 0);
    // A graph, then tear it off.
    run_command("dialog:graph.histogram");
    // Worksheet is empty: OK is refused with a message, no crash.
    abi::ot_key(SPECIAL + 8, 0);
    assert!(texts(&frame()).iter().any(|x| x == "Histogram"));
    abi::ot_key(SPECIAL + 10, 0);
    // Printing an empty Session window still yields at least one page.
    run_command("file.print");
    assert_eq!(abi::ot_outcome_next(), 3);
    assert!(abi::ot_print_pages() >= 1);
    assert!(abi::ot_print_frame(0) > 8);
    assert!(from_binary(&memory::read_out()).is_some());
    assert_eq!(abi::ot_print_frame(99), 0);
    // Bad project bytes give an error outcome carrying the message.
    memory::write_input(b"garbage");
    abi::ot_open_project(7);
    assert_eq!(abi::ot_outcome_next(), 6);
    assert!(String::from_utf8(memory::read_slot(0)).unwrap().contains("cannot open"));
    // Menu JSON and title.
    let n = abi::ot_menu_json() as usize;
    let j = out_string();
    assert_eq!(j.len(), n);
    assert!(j.contains("\"id\":\"file.save\"") && j.contains("Gage R&R"));
    assert!(abi::ot_title() > 0 && out_string().contains("OpenStats"));
    // The theme is reported by name and follows the theme commands.
    assert!(abi::ot_theme() > 0 && out_string() == "light");
    memory::write_input(b"view.theme_dark");
    assert_eq!(abi::ot_command(15), 1);
    assert!(abi::ot_theme() > 0 && out_string() == "dark");
    // Events before init and after resize never panic.
    abi::ot_resize(10.0, 10.0);
    abi::ot_wheel(5.0, 5.0, 0.0, 30.0);
    abi::ot_pointer(1, 5.0, 5.0, 0, 0, 0);
    abi::ot_key(0xFFFF_FFFF, 0);
    assert!(abi::ot_render() > 8);
}

#[test]
fn graph_windows_render_from_tear_off_json() {
    abi::ot_init(1200.0, 800.0);
    let mut h = Host::new(1200.0, 800.0);
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

    memory::write_input(json.as_bytes());
    let handle = abi::ot_graph_open(json.len() as u32);
    assert!(handle > 0);
    assert!(abi::ot_graph_title(handle) > 0);
    assert_eq!(out_string(), "Histogram 1");
    let n = abi::ot_graph_render(handle, 640.0, 400.0);
    assert!(n > 100);
    let cmds = from_binary(&memory::read_out()).unwrap();
    assert!(matches!(cmds[0], Command::Clear { w, h, .. } if (w - 640.0).abs() < 1e-3 && (h - 400.0).abs() < 1e-3));
    abi::ot_graph_close(handle);
    assert_eq!(abi::ot_graph_render(handle, 640.0, 400.0), 0, "closed handle draws nothing");
    assert_eq!(abi::ot_graph_render(0, 640.0, 400.0), 0);
    memory::write_input(b"{not json");
    assert_eq!(abi::ot_graph_open(9), 0);
    assert!(!out_string().is_empty(), "the error message is available");
    memory::write_input(br#"{"title":"t","sections":[]}"#);
    assert_eq!(abi::ot_graph_open(26), 0);
    assert!(GraphHost::from_json("nope").is_err());
}
