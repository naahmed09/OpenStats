//! Session-window export to HTML and to Word (.docx).
//!
//! Both formats are built here from a [`SessionReport`]: tables keep their column alignment
//! (numbers right, text left) and footnotes, warnings are marked, and charts are drawn by the
//! chart engine (inline SVG in HTML, PNG pictures in Word). No third-party code is involved; the
//! Word package is a ZIP written with the project's own ZIP writer.

use openstats_charts::render::raster::render_png;
use openstats_charts::render::svg::to_svg;
use openstats_charts::{render_spec, PlotOptions};
use openstats_core::io::zip::ZipWriter;
use openstats_stats::report::{Block, ChartSpec, ColumnKind, Section, SessionReport, Table};
use std::fmt::Write;

const CHART_W: f64 = 640.0;
const CHART_H: f64 = 400.0;

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&#39;"),
            // Characters that XML 1.0 (and so Word) cannot hold are dropped.
            c if (c as u32) < 0x20 && c != '\n' && c != '\t' => {}
            c => o.push(c),
        }
    }
    o
}

fn chart_scene(spec: &ChartSpec) -> openstats_charts::core::Scene {
    let o = PlotOptions { width: CHART_W, height: CHART_H, ..Default::default() };
    render_spec(spec, &o)
}

fn right_aligned(k: ColumnKind) -> bool {
    !matches!(k, ColumnKind::Text)
}

// ------------------------------------------------------------------------------------ HTML

fn html_table(out: &mut String, t: &Table) {
    out.push_str("<table>\n");
    if !t.title.is_empty() {
        let _ = writeln!(out, "<caption>{}</caption>", esc(&t.title));
    }
    out.push_str("<thead><tr>");
    for c in &t.columns {
        let cls = if right_aligned(c.kind) { "num" } else { "txt" };
        let _ = write!(out, "<th scope=\"col\" class=\"{cls}\">{}</th>", esc(&c.header));
    }
    out.push_str("</tr></thead>\n<tbody>\n");
    let cols = t.formatted_columns();
    for r in 0..t.rows.len() {
        out.push_str("<tr>");
        for (ci, c) in t.columns.iter().enumerate() {
            let cls = if right_aligned(c.kind) { "num" } else { "txt" };
            let _ = write!(out, "<td class=\"{cls}\">{}</td>", esc(&cols[ci][r]));
        }
        out.push_str("</tr>\n");
    }
    out.push_str("</tbody>\n</table>\n");
    for f in &t.footnotes {
        let _ = writeln!(out, "<p class=\"note\">{}</p>", esc(f));
    }
}

fn html_section(out: &mut String, s: &Section) {
    out.push_str("<section>\n");
    if !s.heading.is_empty() {
        let _ = writeln!(out, "<h2>{}</h2>", esc(&s.heading));
    }
    for b in &s.blocks {
        match b {
            Block::Text(t) => {
                let _ = writeln!(out, "<p>{}</p>", esc(t).replace('\n', "<br>\n"));
            }
            Block::Table(t) => html_table(out, t),
            Block::Chart(c) => {
                let svg = to_svg(&chart_scene(c));
                let _ = writeln!(out, "<figure role=\"img\" aria-label=\"{}\">\n{svg}\n</figure>", esc(&c.title));
            }
        }
    }
    for w in &s.warnings {
        let _ = writeln!(out, "<p class=\"warn\"><strong>Warning:</strong> {}</p>", esc(w));
    }
    out.push_str("</section>\n");
}

/// One self-contained HTML page for the report: tables, charts as inline SVG, warnings.
pub fn session_html(report: &SessionReport) -> String {
    let title = if report.title.is_empty() { "OpenStats Session" } else { report.title.as_str() };
    let mut o = String::new();
    o.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    let _ = writeln!(o, "<title>{}</title>", esc(title));
    o.push_str(
        "<style>\nbody{font-family:Helvetica,Arial,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#1a1a1a}\n\
h1{border-bottom:2px solid #333;padding-bottom:.2em}h2{margin-top:1.6em;border-bottom:1px solid #aaa}\n\
table{border-collapse:collapse;margin:1em 0;font-variant-numeric:tabular-nums}\n\
caption{text-align:left;font-weight:bold;margin-bottom:.3em}\n\
th,td{padding:.2em .8em;border-bottom:1px solid #ddd}th{border-bottom:2px solid #333}\n\
.num{text-align:right}.txt{text-align:left}.note{font-size:.9em;color:#444;margin:.1em 0}\n\
.warn{background:#fff4d6;border-left:4px solid #c77c00;padding:.4em .8em}\n\
figure{margin:1em 0}figure svg{max-width:100%;height:auto}\n</style>\n</head>\n<body>\n",
    );
    let _ = writeln!(o, "<h1>{}</h1>", esc(title));
    for s in &report.sections {
        html_section(&mut o, s);
    }
    o.push_str("</body>\n</html>\n");
    o
}

// ------------------------------------------------------------------------------------ Word

struct Doc {
    body: String,
    images: Vec<Vec<u8>>,
}

fn run(text: &str, bold: bool) -> String {
    let rpr = if bold { "<w:rPr><w:b/></w:rPr>" } else { "" };
    // Line breaks inside a paragraph become <w:br/>.
    let mut o = String::new();
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            o.push_str("<w:r><w:br/></w:r>");
        }
        let _ = write!(o, "<w:r>{rpr}<w:t xml:space=\"preserve\">{}</w:t></w:r>", esc(line));
    }
    o
}

fn para(style: Option<&str>, text: &str, bold: bool) -> String {
    let ppr = style.map_or(String::new(), |s| format!("<w:pPr><w:pStyle w:val=\"{s}\"/></w:pPr>"));
    format!("<w:p>{ppr}{}</w:p>", run(text, bold))
}

fn cell(text: &str, right: bool, bold: bool, header: bool) -> String {
    let jc = if right { "<w:jc w:val=\"right\"/>" } else { "" };
    let shade =
        if header { "<w:tcPr><w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"E7E6E6\"/></w:tcPr>" } else { "" };
    format!(
        "<w:tc>{shade}<w:p><w:pPr><w:spacing w:before=\"0\" w:after=\"0\"/>{jc}</w:pPr>{}</w:p></w:tc>",
        run(text, bold)
    )
}

impl Doc {
    fn table(&mut self, t: &Table) {
        if !t.title.is_empty() {
            self.body.push_str(&para(Some("Caption"), &t.title, true));
        }
        self.body.push_str(
            "<w:tbl><w:tblPr><w:tblBorders>\
<w:top w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"333333\"/>\
<w:bottom w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"333333\"/>\
<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"BBBBBB\"/>\
</w:tblBorders><w:tblLayout w:type=\"autofit\"/><w:tblCellMar><w:left w:w=\"80\" w:type=\"dxa\"/><w:right w:w=\"80\" w:type=\"dxa\"/></w:tblCellMar></w:tblPr>",
        );
        let _ = write!(self.body, "<w:tblGrid>{}</w:tblGrid>", "<w:gridCol w:w=\"1500\"/>".repeat(t.columns.len()));
        self.body.push_str("<w:tr><w:trPr><w:tblHeader/></w:trPr>");
        for c in &t.columns {
            self.body.push_str(&cell(&c.header, right_aligned(c.kind), true, true));
        }
        self.body.push_str("</w:tr>");
        let cols = t.formatted_columns();
        for r in 0..t.rows.len() {
            self.body.push_str("<w:tr>");
            for (ci, c) in t.columns.iter().enumerate() {
                self.body.push_str(&cell(&cols[ci][r], right_aligned(c.kind), false, false));
            }
            self.body.push_str("</w:tr>");
        }
        self.body.push_str("</w:tbl>");
        if t.footnotes.is_empty() {
            self.body.push_str(&para(None, "", false));
        }
        for f in &t.footnotes {
            self.body.push_str(&para(Some("Note"), f, false));
        }
    }

    fn chart(&mut self, c: &ChartSpec) {
        let png = render_png(&chart_scene(c), 1.0);
        self.images.push(png);
        let n = self.images.len();
        // Inline picture, 6 inches wide.
        let cx: i64 = 5_486_400;
        let cy: i64 = (cx as f64 * CHART_H / CHART_W) as i64;
        let name = esc(&c.title);
        let _ = write!(
            self.body,
            "<w:p><w:r><w:drawing><wp:inline distT=\"0\" distB=\"0\" distL=\"0\" distR=\"0\">\
<wp:extent cx=\"{cx}\" cy=\"{cy}\"/><wp:docPr id=\"{n}\" name=\"Chart {n}\" descr=\"{name}\"/>\
<a:graphic xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\">\
<a:graphicData uri=\"http://schemas.openxmlformats.org/drawingml/2006/picture\">\
<pic:pic xmlns:pic=\"http://schemas.openxmlformats.org/drawingml/2006/picture\">\
<pic:nvPicPr><pic:cNvPr id=\"{n}\" name=\"chart{n}.png\" descr=\"{name}\"/><pic:cNvPicPr/></pic:nvPicPr>\
<pic:blipFill><a:blip r:embed=\"rIdImg{n}\"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>\
<pic:spPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"{cx}\" cy=\"{cy}\"/></a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></pic:spPr>\
</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"
        );
    }

    fn section(&mut self, s: &Section) {
        if !s.heading.is_empty() {
            self.body.push_str(&para(Some("Heading2"), &s.heading, false));
        }
        for b in &s.blocks {
            match b {
                Block::Text(t) => self.body.push_str(&para(None, t, false)),
                Block::Table(t) => self.table(t),
                Block::Chart(c) => self.chart(c),
            }
        }
        for w in &s.warnings {
            self.body.push_str(&format!(
                "<w:p><w:pPr><w:pStyle w:val=\"Warning\"/></w:pPr>{}{}</w:p>",
                run("Warning: ", true),
                run(w, false)
            ));
        }
    }
}

const STYLES: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\
<w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">\
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii=\"Calibri\" w:hAnsi=\"Calibri\" w:cs=\"Calibri\"/><w:sz w:val=\"22\"/></w:rPr></w:rPrDefault>\
<w:pPrDefault><w:pPr><w:spacing w:after=\"120\"/></w:pPr></w:pPrDefault></w:docDefaults>\
<w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\"><w:name w:val=\"Normal\"/></w:style>\
<w:style w:type=\"paragraph\" w:styleId=\"Title\"><w:name w:val=\"Title\"/><w:basedOn w:val=\"Normal\"/><w:next w:val=\"Normal\"/>\
<w:pPr><w:spacing w:after=\"240\"/></w:pPr><w:rPr><w:b/><w:sz w:val=\"40\"/></w:rPr></w:style>\
<w:style w:type=\"paragraph\" w:styleId=\"Heading2\"><w:name w:val=\"heading 2\"/><w:basedOn w:val=\"Normal\"/><w:next w:val=\"Normal\"/>\
<w:pPr><w:keepNext/><w:spacing w:before=\"280\" w:after=\"100\"/><w:outlineLvl w:val=\"1\"/></w:pPr><w:rPr><w:b/><w:sz w:val=\"28\"/></w:rPr></w:style>\
<w:style w:type=\"paragraph\" w:styleId=\"Caption\"><w:name w:val=\"caption\"/><w:basedOn w:val=\"Normal\"/><w:next w:val=\"Normal\"/>\
<w:pPr><w:keepNext/><w:spacing w:before=\"120\" w:after=\"60\"/></w:pPr></w:style>\
<w:style w:type=\"paragraph\" w:styleId=\"Note\"><w:name w:val=\"Table Note\"/><w:basedOn w:val=\"Normal\"/>\
<w:pPr><w:spacing w:after=\"40\"/></w:pPr><w:rPr><w:sz w:val=\"18\"/><w:color w:val=\"444444\"/></w:rPr></w:style>\
<w:style w:type=\"paragraph\" w:styleId=\"Warning\"><w:name w:val=\"Warning\"/><w:basedOn w:val=\"Normal\"/>\
<w:pPr><w:pBdr><w:left w:val=\"single\" w:sz=\"24\" w:space=\"6\" w:color=\"C77C00\"/></w:pBdr><w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"FFF4D6\"/></w:pPr></w:style>\
</w:styles>";

/// The report as a Word (.docx) package: styled headings, bordered tables with right-aligned
/// numbers, charts as PNG pictures with alternative text.
pub fn session_docx(report: &SessionReport) -> Result<Vec<u8>, String> {
    let mut d = Doc { body: String::new(), images: Vec::new() };
    let title = if report.title.is_empty() { "OpenStats Session" } else { report.title.as_str() };
    d.body.push_str(&para(Some("Title"), title, false));
    for s in &report.sections {
        d.section(s);
    }
    let document = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\
<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" \
xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" \
xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\">\
<w:body>{}<w:sectPr><w:pgSz w:w=\"12240\" w:h=\"15840\"/><w:pgMar w:top=\"1080\" w:right=\"1080\" w:bottom=\"1080\" w:left=\"1080\" w:header=\"720\" w:footer=\"720\" w:gutter=\"0\"/></w:sectPr></w:body></w:document>",
        d.body
    );
    let content_types = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\
<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\
<Default Extension=\"xml\" ContentType=\"application/xml\"/>\
<Default Extension=\"png\" ContentType=\"image/png\"/>\
<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>\
<Override PartName=\"/word/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/>\
<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>\
</Types>";
    let rels = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>\
<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>\
</Relationships>";
    let mut doc_rels = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\
<Relationship Id=\"rIdStyles\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>",
    );
    for i in 1..=d.images.len() {
        let _ = write!(
            doc_rels,
            "<Relationship Id=\"rIdImg{i}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/chart{i}.png\"/>"
        );
    }
    doc_rels.push_str("</Relationships>");
    let core = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\
<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" \
xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:title>{}</dc:title><dc:creator>OpenStats</dc:creator></cp:coreProperties>",
        esc(title)
    );
    let mut z = ZipWriter::new();
    let e = |r: openstats_core::Result<()>| r.map_err(|e| e.to_string());
    e(z.add("[Content_Types].xml", content_types.as_bytes(), 6))?;
    e(z.add("_rels/.rels", rels.as_bytes(), 6))?;
    e(z.add("docProps/core.xml", core.as_bytes(), 6))?;
    e(z.add("word/document.xml", document.as_bytes(), 6))?;
    e(z.add("word/styles.xml", STYLES.as_bytes(), 6))?;
    e(z.add("word/_rels/document.xml.rels", doc_rels.as_bytes(), 6))?;
    for (i, png) in d.images.iter().enumerate() {
        e(z.add(&format!("word/media/chart{}.png", i + 1), png, 0))?;
    }
    Ok(z.finish())
}
