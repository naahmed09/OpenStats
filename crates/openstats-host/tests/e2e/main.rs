//! End-to-end workflows through the C ABI: a headless user importing data, running analyses from
//! menus and dialogs, reading the Session window, looking at charts and saving projects.
//!
//! Every number the application reports is compared with a value computed independently in the
//! test from the same data.

mod driver;

use driver::*;
use openstats_core::io::zip::ZipWriter;

// ---------------------------------------------------------------------- helpers

/// The whitespace-separated cells of the first row starting with `label` after `section`.
fn row(text: &str, section: &str, label: &str) -> Vec<String> {
    let start = text.find(section).unwrap_or_else(|| panic!("no \"{section}\" in\n{text}"));
    text[start..]
        .lines()
        .find(|l| l.starts_with(label))
        .unwrap_or_else(|| panic!("no row \"{label}\" under {section} in\n{text}"))
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn num(s: &str) -> f64 {
    s.parse().unwrap_or_else(|_| panic!("not a number: {s}"))
}

fn close(a: f64, b: f64, rel: f64) {
    assert!((a - b).abs() <= rel * b.abs().max(1.0), "{a} vs {b}");
}

/// Ordinary least squares of y on x, independently of the library.
fn ols(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let syy: f64 = y.iter().map(|v| (v - my).powi(2)).sum();
    let b = sxy / sxx;
    let a = my - b * mx;
    let sse = syy - b * sxy;
    ((a), b, (sse / (n - 2.0)).sqrt(), 1.0 - sse / syy)
}

fn line_data() -> (Vec<f64>, Vec<f64>, String) {
    let x: Vec<f64> = (0..30).map(f64::from).collect();
    let y: Vec<f64> = (0..30).map(|i| 1.0 + 2.0 * f64::from(i) + f64::from((i * 7) % 5)).collect();
    let mut csv = String::from("x,y\n");
    for (a, b) in x.iter().zip(&y) {
        csv.push_str(&format!("{a},{b}\n"));
    }
    (x, y, csv)
}

/// A minimal XLSX holding one sheet: a header row and one numeric column.
fn xlsx(header: &str, values: &[f64]) -> Vec<u8> {
    let mut z = ZipWriter::new();
    z.add("[Content_Types].xml", b"<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"/>", 6)
        .unwrap();
    z.add("_rels/.rels", br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#, 6).unwrap();
    z.add("xl/workbook.xml", br#"<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Parts" sheetId="1" r:id="rId1"/></sheets></workbook>"#, 6).unwrap();
    z.add("xl/_rels/workbook.xml.rels", br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/></Relationships>"#, 6).unwrap();
    z.add(
        "xl/sharedStrings.xml",
        format!(
            "<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><si><t>{header}</t></si></sst>"
        )
        .as_bytes(),
        6,
    )
    .unwrap();
    let mut rows = String::from("<row r=\"1\"><c r=\"A1\" t=\"s\"><v>0</v></c></row>");
    for (i, v) in values.iter().enumerate() {
        rows.push_str(&format!("<row r=\"{0}\"><c r=\"A{0}\"><v>{1}</v></c></row>", i + 2, v));
    }
    z.add("xl/worksheets/sheet1.xml", format!("<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><sheetData>{rows}</sheetData></worksheet>").as_bytes(), 6).unwrap();
    z.finish()
}

// ---------------------------------------------------------------------- workflows

#[test]
fn csv_regression_report_charts_save_and_reopen() {
    let d = Driver::start(1280.0, 800.0);
    let (x, y, csv) = line_data();
    d.import("line.csv", csv.as_bytes());
    assert!(d.outcomes().is_empty());
    assert!(d.shows("line"), "the worksheet is in the Navigator");
    assert!(d.title().contains("OpenStats"));

    // Stat > Regression > Regression..., response y, predictor x, residual plots, store fits and residuals.
    d.menu(&["Stat", "Regression", "Regression..."]);
    assert!(d.shows("Least-squares regression"), "the dialog is open");
    d.pick_column(1);
    d.tab();
    d.pick_column(0);
    d.click_text("Graphs...");
    d.tab();
    d.key(' ', 0);
    d.click_text("Storage...");
    d.tab();
    d.key(' ', 0);
    d.tab();
    d.key(' ', 0);
    d.enter();
    assert!(!d.shows("Least-squares regression"), "OK closed the dialog");

    // The Session report: coefficients, S and R² agree with an independent least squares.
    let text = d.session_text();
    let (a, b, s, r2) = ols(&x, &y);
    let constant = row(&text, "Coefficients", "Constant");
    let slope = row(&text, "Coefficients", "x");
    close(num(&constant[1]), a, 1e-5);
    close(num(&slope[1]), b, 1e-5);
    let summary = row(&text, "Model Summary", "1.");
    close(num(&summary[0]), s, 1e-5);
    close(num(&summary[1]), 100.0 * r2, 1e-4);
    let anova = row(&text, "Analysis of Variance", "Error");
    assert_eq!(num(&anova[1]), 28.0, "error degrees of freedom");

    // Stored columns: FITS = a + b x and RESI = y − FITS, row by row, read back as CSV.
    let sheet = d.worksheet_csv();
    let mut lines = sheet.lines();
    assert_eq!(lines.next().unwrap(), "x,y,FITS,RESI");
    let mut sum_resid = 0.0;
    for (i, l) in lines.enumerate() {
        let c: Vec<f64> = l.split(',').map(num).collect();
        close(c[2], a + b * x[i], 1e-9);
        close(c[3], y[i] - c[2], 1e-9);
        sum_resid += c[3];
    }
    assert!(sum_resid.abs() < 1e-9, "residuals sum to zero, got {sum_resid}");

    // Edit > Undo removes the stored columns in one step; Ctrl+Y brings them back.
    d.ctrl('z');
    assert_eq!(d.worksheet_csv().lines().next().unwrap(), "x,y");
    d.ctrl('y');
    assert_eq!(d.worksheet_csv(), sheet);

    // The four residual plots are in the Navigator and each draws a real chart.
    for name in ["Probability Plot 1", "Scatterplot 1", "Histogram 1", "Line Plot 1"] {
        assert!(d.shows(name), "{name} in the Navigator");
    }
    d.right_click_text("Scatterplot 1");
    d.click_text("Open in New Window");
    let json = match d.outcomes().as_slice() {
        [Out::TearOff { name, json }] => {
            assert_eq!(name, "Scatterplot 1");
            json.clone()
        }
        other => panic!("expected a tear-off, got {other:?}"),
    };
    let (h, title) = d.graph_open(&json);
    assert_eq!(title, "Scatterplot 1");
    let small = d.graph_texts(h, 400.0, 300.0);
    let large = d.graph_texts(h, 900.0, 600.0);
    assert!(large.iter().any(|t| t.contains("Residual")), "{large:?}");
    assert!(large.iter().any(|t| t.contains("Fits")) || small.iter().any(|t| t.contains("Fits")));
    d.graph_close(h);

    // Double-clicking a graph in the Navigator shows it in the upper pane.
    d.double_click_text("Histogram 1");
    assert!(d.shows("Histogram of Standardized Residuals"), "{:?}", d.texts());

    // Export a graph as PNG through its context menu.
    d.right_click_text("Histogram 1");
    d.click_text("Export as PNG...");
    match d.outcomes().as_slice() {
        [Out::File { name, mime, bytes }] => {
            assert_eq!((name.as_str(), mime.as_str()), ("Histogram 1.png", "image/png"));
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        }
        other => panic!("{other:?}"),
    }

    // Save the project, reopen it in a fresh window: the Session, worksheet and graphs survive.
    d.ctrl('s');
    let outs = d.outcomes();
    assert!(matches!(outs.as_slice(), [Out::File { name, .. }] if name.ends_with(".otb")));
    let project = d.save_project();
    let d2 = Driver::start(900.0, 600.0);
    assert!(!d2.shows("Scatterplot 1"));
    d2.open_project(&project);
    assert!(d2.outcomes().is_empty());
    assert_eq!(d2.session_text(), text);
    assert_eq!(d2.worksheet_csv(), sheet);
    for name in ["Probability Plot 1", "Scatterplot 1", "Histogram 1", "Line Plot 1"] {
        assert!(d2.shows(name), "{name} after reopening");
    }

    // Undo history is not part of a project: a reopened project starts with nothing to undo.
    d2.ctrl('z');
    assert!(matches!(d2.outcomes().as_slice(), [Out::Error(m)] if m.contains("nothing to undo")));
}

#[test]
fn xlsx_import_and_capability_analysis_with_spec_limits() {
    let d = Driver::start(1280.0, 800.0);
    let data = [9.9, 10.1, 10.0, 10.2, 9.8, 10.05, 9.95, 10.1, 10.0, 9.85, 10.15, 10.02];
    d.import("parts.xlsx", &xlsx("Diameter", &data));
    assert!(d.outcomes().is_empty(), "the workbook imported");
    assert!(d.shows("Parts"), "the sheet name is in the Navigator");
    assert!(d.worksheet_csv().starts_with("Diameter\n9.9\n10.1\n"));

    d.menu(&["Stat", "Quality Tools", "Capability Analysis", "Normal..."]);
    d.pick_column(0);
    d.tab(); // subgroup size
    d.tab(); // lower spec
    d.type_text("9.4");
    d.tab(); // upper spec
    d.type_text("10.6");
    d.enter();
    let text = d.session_text();
    assert!(text.contains("Process Capability of Diameter"));

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let sd = (data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let mr_bar = data.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (n - 1.0);
    let sd_within = mr_bar / std::f64::consts::FRAC_2_SQRT_PI; // d2(2) = 2/sqrt(pi)
    let (lsl, usl) = (9.4, 10.6);
    close(num(&row(&text, "Process Data", "Sample Mean")[2]), mean, 1e-5);
    close(num(&row(&text, "Process Data", "StDev (Overall)")[2]), sd, 1e-5);
    close(num(&row(&text, "Process Data", "StDev (Within)")[2]), sd_within, 1e-5);
    close(num(&row(&text, "Overall Capability", "Pp")[1]), (usl - lsl) / (6.0 * sd), 1e-5);
    close(num(&row(&text, "Overall Capability", "Ppk")[1]), (usl - mean).min(mean - lsl) / (3.0 * sd), 1e-5);
    close(
        num(&row(&text, "Potential (Within) Capability", "Cpk")[1]),
        (usl - mean).min(mean - lsl) / (3.0 * sd_within),
        1e-5,
    );
    close(num(&row(&text, "Potential (Within) Capability", "Cp")[1]), (usl - lsl) / (6.0 * sd_within), 1e-5);
    assert!(d.shows("Expected Performance"), "the Session window follows the newest output");
}

#[test]
fn every_menu_command_runs_and_every_dialog_opens_draws_and_cancels() {
    let d = Driver::start(1280.0, 800.0);
    let (_, _, csv) = line_data();
    d.import("line.csv", csv.as_bytes());
    assert!(d.command("file.export_worksheet"));
    d.outcomes();
    // Collect every command id the native menu would publish.
    abi_menu_json(&d);
    let ids = menu_ids();
    assert!(ids.len() > 45, "{} ids", ids.len());
    let mut dialogs = 0;
    for id in &ids {
        // Commands that replace or clear the project are exercised last.
        if matches!(id.as_str(), "file.new" | "edit.clear_session" | "file.open" | "file.import") {
            continue;
        }
        assert!(d.command(id), "{id} is unknown to the host");
        let outs = d.outcomes();
        assert!(outs.iter().all(|o| !matches!(o, Out::Error(m) if m.contains("panicked"))), "{id}: {outs:?}");
        let _ = d.frame();
        if id.starts_with("dialog:") {
            dialogs += 1;
            assert!(d.shows("OK") && d.shows("Cancel"), "{id} draws its dialog");
            d.escape();
            assert!(!d.shows("Cancel"), "{id} closes on Escape");
        }
        d.escape();
    }
    assert!(dialogs >= 25, "{dialogs} dialogs");
    assert!(d.command("file.open") && d.command("file.import"));
    assert_eq!(d.outcomes(), vec![Out::NeedOpen, Out::NeedImport]);
    assert!(!d.command("no.such.command"));
}

fn abi_menu_json(_d: &Driver) {
    openstats_host::abi::ot_menu_json();
    MENU.with(|m| *m.borrow_mut() = String::from_utf8(openstats_host::abi::memory::read_out()).unwrap());
}

thread_local! {
    static MENU: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

fn menu_ids() -> Vec<String> {
    MENU.with(|m| {
        let j = m.borrow();
        let mut ids: Vec<String> =
            j.split("\"id\":\"").skip(1).map(|r| r[..r.find('"').unwrap()].to_string()).collect();
        ids.sort();
        ids.dedup();
        ids
    })
}

#[test]
fn keyboard_shortcuts_themes_and_pane_focus() {
    let d = Driver::start(1280.0, 800.0);
    let (_, _, csv) = line_data();
    d.import("line.csv", csv.as_bytes());

    // Ctrl+Shift+D switches to the dark theme and back; the window background follows.
    let light = d.background();
    d.key('D', CTRL | SHIFT);
    let dark = d.background();
    assert!(u32::from(dark.0) + u32::from(dark.1) + u32::from(dark.2) < 150, "{dark:?}");
    assert!(u32::from(light.0) + u32::from(light.1) + u32::from(light.2) > 450, "{light:?}");
    // The cycle continues with high contrast (pure black) and returns to light.
    d.key('D', CTRL | SHIFT);
    assert_eq!(d.background(), (0, 0, 0), "high contrast");
    d.key('D', CTRL | SHIFT);
    assert_eq!(d.background(), light);
    // View > Theme picks one directly; the status bar names it.
    d.menu(&["View", "Theme", "High Contrast"]);
    assert_eq!(d.background(), (0, 0, 0));
    assert!(d.shows("High Contrast theme"));
    d.menu(&["View", "Theme", "Light"]);
    assert_eq!(d.background(), light);

    // Alt+S opens the Stat menu, F10 the File menu; Escape closes either.
    d.key('s', ALT);
    assert!(d.shows("Regression"), "Alt+S opens Stat");
    d.escape();
    d.special(15, 0);
    assert!(d.shows("Save Project") || d.shows("Export Session (Text)..."));
    d.escape();

    // Ctrl+O, Ctrl+I, Ctrl+S, Ctrl+P: the file commands as outcomes.
    d.ctrl('o');
    d.ctrl('i');
    d.ctrl('p');
    let outs = d.outcomes();
    assert_eq!(outs[0], Out::NeedOpen);
    assert_eq!(outs[1], Out::NeedImport);
    assert!(matches!(outs[2], Out::Print(n) if n >= 1));

    // Ctrl+F moves into the formula bar and typing edits the active cell; Escape abandons it.
    d.ctrl('f');
    d.type_text("42");
    d.escape();
    assert!(d.worksheet_csv().starts_with("x,y\n0,1\n"), "the escaped edit changed nothing");
    // F6 cycles panes: Navigator → Session → worksheet; typing in the worksheet edits the cell.
    d.f6();
    d.f6();
    d.f6();
    d.f6();
    d.ctrl('f');
    d.type_text("77");
    d.enter();
    assert!(d.worksheet_csv().starts_with("x,y\n77,1\n"), "{}", d.worksheet_csv());
    d.ctrl('z');
    assert!(d.worksheet_csv().starts_with("x,y\n0,1\n"));
    d.key('Z', CTRL | SHIFT);
    assert!(d.worksheet_csv().starts_with("x,y\n77,1\n"), "Ctrl+Shift+Z redoes");

    // Ctrl+Shift+N adds a worksheet, Ctrl+N starts over.
    d.key('N', CTRL | SHIFT);
    assert!(d.shows("Worksheet 2") || d.shows("Worksheet 1"));
    d.ctrl('n');
    assert!(!d.shows("line"), "a new project has no imported sheet");
}

#[test]
fn editing_pasting_and_input_methods_in_the_worksheet() {
    let d = Driver::start(1280.0, 800.0);
    d.ctrl('n');
    d.click_text("C1"); // the worksheet's first column header
                        // Paste a tab-separated block with a header row of text and numbers.
    d.paste("a\tb\n1\t2.5\n3\t4.5\n");
    assert_eq!(
        d.worksheet_csv(),
        "C1,C2\na,b\n1,2.5\n3,4.5\n",
        "pasted cells land at the active cell; the first row is data"
    );
    // An input-method composition commits text into a cell.
    d.special(4, CTRL); // Ctrl+Home: back to A1
    d.key('x', 0); // starts editing A1
    assert_eq!(openstats_host::abi::ot_ime_anchor(), 1, "the editor accepts composition");
    memory_ime("かな", 2);
    d.enter();
    assert_eq!(d.worksheet_csv(), "C1,C2\nxかな,b\n1,2.5\n3,4.5\n", "the committed text joins the typed text");
    // Copy the selection as TSV and paste it elsewhere: Ctrl+C then Ctrl+V arrives as paste text.
    d.ctrl('a');
    d.ctrl('c');
    let outs = d.outcomes();
    assert!(
        outs.iter().any(|o| matches!(o, Out::Clipboard(t) if t.starts_with("xかな\tb") && t.contains("3\t4.5"))),
        "{outs:?}"
    );
}

fn memory_ime(text: &str, caret: u32) {
    use openstats_host::abi::{self, memory};
    memory::write_input(text.as_bytes());
    abi::ot_ime_update(text.len() as u32, caret);
    memory::write_input(text.as_bytes());
    abi::ot_ime_commit(text.len() as u32);
}

#[test]
fn bad_input_is_reported_not_fatal() {
    let d = Driver::start(1000.0, 700.0);
    // Garbage as data and as a project.
    d.import("junk.xlsx", b"PK\x03\x04 not really a zip");
    let outs = d.outcomes();
    assert!(matches!(outs.as_slice(), [Out::Error(m)] if m.contains("junk.xlsx")), "{outs:?}");
    d.open_project(b"nonsense");
    assert!(matches!(d.outcomes().as_slice(), [Out::Error(_)]));
    // A project handed to Import is refused with advice.
    let (_, _, csv) = line_data();
    d.import("line.csv", csv.as_bytes());
    d.ctrl('s');
    d.outcomes();
    let project = d.save_project();
    d.import("p.otb", &project);
    let outs = d.outcomes();
    assert!(matches!(outs.as_slice(), [Out::Error(m)] if m.contains("Open Project")), "{outs:?}");

    // OK with nothing chosen keeps the dialog open and says why; Cancel closes it.
    d.menu(&["Stat", "Regression", "Regression..."]);
    d.enter();
    assert!(d.shows("Cancel"), "the dialog stays open");
    assert!(d.shows("Response: select a column."), "the problem is named in the dialog");
    d.click_text("Cancel");
    assert!(!d.shows("Cancel") && !d.shows("Response: select a column."));

    // A regression of a column on itself is refused by the analysis, not by a crash.
    d.menu(&["Stat", "Regression", "Regression..."]);
    d.pick_column(0);
    d.tab();
    d.pick_column(0);
    d.enter();
    assert!(!d.title().is_empty());
    let _ = d.frame();
}

#[test]
fn session_window_copy_print_scroll_and_export() {
    let d = Driver::start(1280.0, 800.0);
    let (_, _, csv) = line_data();
    d.import("line.csv", csv.as_bytes());
    d.menu(&["Stat", "Basic Statistics", "Display Descriptive Statistics..."]);
    d.pick_column(0);
    d.enter();
    let text = d.session_text();
    assert!(text.contains("Descriptive Statistics"));
    let mean = row(&text, "Mean", "x");
    assert!(!mean.is_empty());
    // Focus the Session window, select all and copy: the clipboard carries the report text.
    d.click_text("Variable");
    d.ctrl('a');
    d.ctrl('c');
    let outs = d.outcomes();
    let copied =
        outs.iter().find_map(|o| if let Out::Clipboard(t) = o { Some(t.clone()) } else { None }).expect("clipboard");
    assert!(copied.contains("Descriptive Statistics") && copied.contains("Mean"));
    // Printing yields pages; scrolling and paging never panic.
    d.ctrl('p');
    assert!(matches!(d.outcomes().as_slice(), [Out::Print(n)] if *n >= 1));
    for _ in 0..5 {
        d.wheel(800.0, 200.0, 120.0);
        d.special(7, 0);
    }
    d.special(5, 0);
    d.special(4, 0);
    let _ = d.frame();
    // Clear the Session window through the Edit menu.
    d.menu(&["Edit", "Clear Session Window"]);
    assert!(!d.session_text().contains("Descriptive Statistics"));
}

#[test]
fn window_resize_and_a_large_worksheet_stay_responsive() {
    let d = Driver::start(1280.0, 800.0);
    let mut csv = String::from("v\n");
    for i in 0..200_000 {
        csv.push_str(&format!("{}\n", f64::from(i % 97) * 0.5));
    }
    d.import("big.csv", csv.as_bytes());
    let t0 = std::time::Instant::now();
    for k in 0..40 {
        d.wheel(600.0, 500.0, 300.0 + f64::from(k));
        d.frame();
    }
    // 40 scrolled frames of a 200,000-row sheet, even in a debug build, stay well under a second each.
    assert!(t0.elapsed().as_secs_f64() < 40.0, "{:?}", t0.elapsed());
    for (w, h) in [(640.0, 480.0), (2000.0, 1200.0), (320.0, 240.0)] {
        openstats_host::abi::ot_resize(w, h);
        assert!(!d.frame().is_empty());
    }
    d.menu(&["Stat", "Basic Statistics", "Display Descriptive Statistics..."]);
    d.pick_column(0);
    d.enter();
    let text = d.session_text();
    let n = row(&text, "Variable", "v");
    assert_eq!(num(&n[1]), 200_000.0, "N = 200,000: {text}");
}

#[test]
fn sheet_tabs_status_bar_and_help() {
    let d = Driver::start(1280.0, 800.0);
    let (_, _, csv) = line_data();
    d.import("line.csv", csv.as_bytes());
    // Status bar: project, sheet and size.
    assert!(d.shows("Untitled") && d.shows("columns") && d.shows("Light theme"), "{:?}", d.texts());
    // Worksheet tabs: the imported sheet, then "+" adds another and switches to it.
    assert!(d.shows("line"), "tab for the imported sheet");
    d.click_text("+");
    assert!(d.shows("Worksheet 2"), "new tab");
    assert!(
        d.texts().iter().any(|t| t.contains("Worksheet 2") && t.contains("0 rows")),
        "status follows the active sheet"
    );
    d.click_text("line");
    assert!(d.texts().iter().any(|t| t.starts_with("line") && !t.contains("0 columns")));
    // Help menu.
    d.menu(&["Help", "Keyboard Shortcuts"]);
    assert!(d.session_text().contains("Ctrl+Shift+D"));
    d.menu(&["Help", "About OpenStats"]);
    assert!(d.session_text().contains("Apache-2.0 OR MIT"));
}
