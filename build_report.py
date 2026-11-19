"""Builds the project report .docx from the template cover + computed results.
Usage: python build_report.py TEMPLATE.docx WORK_DIR FIG_DIR CODE_DIR OUT.docx"""
import sys, json, copy, re
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

tpl, work, figs, code_dir, out = sys.argv[1:6]
work, figs, code_dir = Path(work), Path(figs), Path(code_dir)
R12 = json.load(open(work / "u12_results.json")); R3 = json.load(open(work / "u3_results.json"))
R4 = json.load(open(work / "u4_results.json")); R5 = json.load(open(work / "u5_results.json"))
BN = ["B2 Blue", "B3 Green", "B4 Red", "B5 NIR"]; CN = ["Water-like", "Vegetation-like", "Bright low-vegetation", "Bare/other-like"]
f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"; f4 = lambda x: f"{x:.4f}"; n0 = lambda x: f"{int(x):,}"

doc = Document(tpl)
body = doc.element.body
# ---------- trim template after cover ----------
children = list(body.iterchildren())
start = None
for i, el in enumerate(children):
    if el.tag == qn("w:p") and "".join(t.text or "" for t in el.iter(qn("w:t"))).strip() == "ABSTRACT":
        start = i; break
for el in children[start:]:
    if el.tag != qn("w:sectPr"):
        body.remove(el)
# drop trailing page-break-only paragraphs of cover
# ---------- cover edits ----------
TITLE = "Landsat Change Explorer: Machine Learning Analysis of Spectral Change in Landsat 8 Imagery (2015–2024)"
name_paras = []
for p in list(doc.paragraphs):
    t = p.text.strip()
    if t.startswith("Design, Implementation"):
        p.runs[0].text = TITLE
        for r in p.runs[1:]: r.text = ""
        for r in p.runs: r.font.size = Pt(18)
    elif t == "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY":
        for sp_ in list(p._p.iter(qn("w:spacing"))): sp_.getparent().remove(sp_)
    elif t == "Name  [Reg No]" or t == "Name [Reg No]":
        name_paras.append(p)
    elif t == "Faculty Name":
        p.runs[0].text = "Dr. U. Karthikeyan"
        for r in p.runs[1:]: r.text = ""
    elif t.startswith("(Designation"):
        p.runs[0].text = "([Designation], Department of Computing Technologies)"
        for r in p.runs[1:]: r.text = ""
    elif t == "with specialization in AIML":
        p._p.getparent().remove(p._p)
    elif t in ("COMPUTER SCIENCE ENGINEERING",):
        p.runs[0].text = "COMPUTER SCIENCE AND ENGINEERING"
        for r in p.runs[1:]: r.text = ""
    elif t == "DEPARTMENT OF COMPUTATIONAL INTELLIGENCE":
        p.runs[0].text = "DEPARTMENT OF COMPUTING TECHNOLOGIES"
        for r in p.runs[1:]: r.text = ""
    elif t.startswith("NOV"):
        p.runs[0].text = "NOVEMBER 2026"
        for r in p.runs[1:]: r.text = ""

STUDENTS = [("Student 1", "Register number 1"), ("Student 2", "Register number 2"),
            ("Student 3", "Register number 3"), ("Student 4", "Register number 4")]
assert len(name_paras) == 2, name_paras
while len(name_paras) < len(STUDENTS):
    new_p = copy.deepcopy(name_paras[-1]._p); name_paras[-1]._p.addnext(new_p)
    from docx.text.paragraph import Paragraph as _P
    name_paras.append(_P(new_p, name_paras[-1]._parent))
    nxt = new_p.getnext()
    if nxt is not None and nxt.tag == qn("w:p") and not "".join(x.text or "" for x in nxt.iter(qn("w:t"))).strip():
        nxt.getparent().remove(nxt)
for p, (nm, rg) in zip(name_paras, STUDENTS):
    p.runs[0].text = f"{nm} [{rg}]"
    for r in p.runs[1:]: r.text = ""

# ---------- styles ----------
st = doc.styles
def pstyle(name, size, bold=False, align=None, before=0, after=6, outline=None, italic=False, font="Times New Roman", spacing=1.5, keep=False):
    s = st[name] if name in [x.name for x in st] else st.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    s.base_style = None
    f = s.font; f.name = font; f.size = Pt(size); f.bold = bold; f.italic = italic; f.color.rgb = RGBColor(0, 0, 0)
    rpr = s.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
    if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"): rf.set(qn(a), font)
    pf = s.paragraph_format; pf.space_before = Pt(before); pf.space_after = Pt(after); pf.line_spacing = spacing
    if align is not None: pf.alignment = align
    pf.keep_with_next = keep
    if outline is not None:
        ppr = s.element.get_or_add_pPr(); ol = OxmlElement("w:outlineLvl"); ol.set(qn("w:val"), str(outline)); ppr.append(ol)
    return s
J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT
pstyle("RBody", 12, align=J)
pstyle("RChapNo", 16, True, C, 0, 6, keep=True)
pstyle("RChapter", 16, True, C, 0, 18, outline=0, keep=True)
pstyle("RFront", 16, True, C, 0, 18, keep=True)
pstyle("RTitleNT", 16, True, C, 0, 18, keep=True)
pstyle("RSection", 14, True, L, 12, 6, outline=1, keep=True)
pstyle("RSub", 12, True, L, 8, 4, outline=2, keep=True)
pstyle("RFig", 11, True, C, 4, 12, spacing=1.0)
pstyle("RTab", 11, True, C, 10, 4, spacing=1.0, keep=True)
pstyle("RCell", 11, align=L, before=0, after=0, spacing=1.0)
pstyle("REq", 12, align=C, before=4, after=4, italic=False, spacing=1.15)
pstyle("RCode", 10, font="Consolas", align=L, before=0, after=0, spacing=1.0)
pstyle("RBullet", 12, align=J, after=3)
pstyle("RRef", 11, align=J, after=2, spacing=1.0)
for _n in ("toc 1", "toc 2"):
    _s = pstyle(_n, 11, False, L, 0, 2, spacing=1.0)
    _s.paragraph_format.left_indent = Inches(0 if _n == "toc 1" else 0.3)

# bullets numbering via a simple hanging-indent + symbol run (Word list numbering not needed for print)
def para(text="", style="RBody", bold_prefix=None, align=None):
    p = doc.add_paragraph(style=style)
    if bold_prefix:
        r = p.add_run(bold_prefix); r.bold = True
    add_rich(p, text)
    if align is not None: p.alignment = align
    return p

def add_rich(p, text):
    """supports **bold** and _italic_ segments"""
    for part in re.split(r"(\*\*[^*]+\*\*|__[^_]+__)", text):
        if not part: continue
        if part.startswith("**"): r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("__"): r = p.add_run(part[2:-2]); r.italic = True
        else: p.add_run(part)

def bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style="RBullet")
    p.paragraph_format.left_indent = Inches(0.35); p.paragraph_format.first_line_indent = Inches(-0.25)
    p.add_run("•\t")
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.35))
    if bold_prefix: r = p.add_run(bold_prefix); r.bold = True
    add_rich(p, text)
    return p

def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

def chapter(no, title):
    p = doc.add_paragraph(style="RChapNo"); p.add_run(f"CHAPTER {no}")
    p.paragraph_format.page_break_before = True
    doc.add_paragraph(title.upper(), style="RChapter")

def section(t): doc.add_paragraph(t, style="RSection")
def sub(t): doc.add_paragraph(t, style="RSub")

FIGS, TABS = [], []
def figure(fname, caption, width=6.0):
    p = doc.add_paragraph(); p.alignment = C; p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6); p.paragraph_format.line_spacing = 1.0
    p.add_run().add_picture(str(figs / fname), width=Inches(width))
    doc.add_paragraph(caption, style="RFig"); FIGS.append(caption)

def set_cell_bg(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor); tcPr.append(shd)

def borders(table):
    tblPr = table._tbl.tblPr; b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000"); b.append(e)
    tblPr.append(b)

def table(caption, header, rows, widths, align_num=True, size=11):
    if caption:
        doc.add_paragraph(caption, style="RTab"); TABS.append(caption)
    t = doc.add_table(rows=1 + len(rows), cols=len(header)); borders(t); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]; c.text = ""; p = c.paragraphs[0]; p.style = "RCell"; r = p.add_run(str(h)); r.bold = True; r.font.size = Pt(size)
        p.alignment = C; set_cell_bg(c, "D9E2F3")
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.rows[i + 1].cells[j]; c.text = ""; p = c.paragraphs[0]; p.style = "RCell"
            add_rich(p, str(v))
            for r in p.runs: r.font.size = Pt(size)
            if align_num and j > 0 and re.fullmatch(r"[-−+0-9.,%\s/()]+", str(v)): p.alignment = C
    for row in t.rows:
        for j, w in enumerate(widths):
            row.cells[j].width = Inches(w)
        trPr = row._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
    hdr = t.rows[0]._tr.get_or_add_trPr(); th = OxmlElement("w:tblHeader"); hdr.append(th)
    for row in t.rows[:-1]:
        for c in row.cells:
            for pp in c.paragraphs: pp.paragraph_format.keep_with_next = True
    sp = doc.add_paragraph(style="RBody"); sp.paragraph_format.space_after = Pt(2); sp.paragraph_format.line_spacing = 1.0
    return t

def eq(text, no):
    p = doc.add_paragraph(style="REq")
    p.paragraph_format.tab_stops.add_tab_stop(Inches(2.9), WD_TAB_ALIGNMENT.CENTER)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(5.75), WD_TAB_ALIGNMENT.RIGHT)
    p.alignment = L
    p.add_run("\t"); r = p.add_run(text); r.font.name = "Cambria Math"; r.italic = False
    p.add_run(f"\t({no})")

def field(p, instr):
    r = p.add_run(); fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), "begin"); r._r.append(fc)
    r2 = p.add_run(); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr; r2._r.append(it)
    r3 = p.add_run(); fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), "separate"); r3._r.append(fc)
    p.add_run("Right-click and select Update Field.")
    r5 = p.add_run(); fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), "end"); r5._r.append(fc)

def setup_section(sec, fmt, start=None, show_number=True):
    sec.left_margin = Inches(1.5); sec.right_margin = Inches(1); sec.top_margin = Inches(1); sec.bottom_margin = Inches(1)
    sec.page_width = Emu(7556500); sec.page_height = Emu(10699750)
    sp = sec._sectPr
    for old in sp.findall(qn("w:pgNumType")): sp.remove(old)
    pg = OxmlElement("w:pgNumType"); pg.set(qn("w:fmt"), fmt)
    if start is not None: pg.set(qn("w:start"), str(start))
    sp.append(pg)
    ft = sec.footer; ft.is_linked_to_previous = False
    p = ft.paragraphs[0] if ft.paragraphs else ft.add_paragraph()
    for r in list(p.runs): r._r.getparent().remove(r._r)
    p.alignment = C
    if show_number: field(p, "PAGE")

# ======================= BONAFIDE =======================
for t, s in (("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY", 18), ("KATTANKULATHUR – 603 203", 16)):
    p = doc.add_paragraph(style="RTitleNT"); p.paragraph_format.space_after = Pt(4); r = p.add_run(t); r.font.size = Pt(s)
p = doc.add_paragraph(style="RTitleNT"); p.paragraph_format.space_before = Pt(24); r = p.add_run("BONAFIDE CERTIFICATE"); r.underline = True
p = doc.add_paragraph(style="RBody"); p.paragraph_format.space_before = Pt(18)
p.add_run("Certified that the 21CSC305P – MACHINE LEARNING project report titled “")
r = p.add_run(TITLE); r.bold = True
p.add_run("” is the bonafide work of “" + ", ".join(f"{nm} [{rg}]" for nm, rg in STUDENTS) + "” who carried out the project work under my supervision within the allotted time.")
for r in p.runs: r.font.size = Pt(14)
for _ in range(4): doc.add_paragraph()
t = doc.add_table(rows=1, cols=2)
txt = [["SIGNATURE", "", "Dr. U. Karthikeyan", "Course Faculty", "[Designation]", "Department of Computing Technologies", "SRM Institute of Science and Technology", "Kattankulathur"],
       ["SIGNATURE", "", "Dr. G. Niranjana", "Head of the Department", "Professor", "Department of Computing Technologies", "SRM Institute of Science and Technology", "Kattankulathur"]]
for j in range(2):
    c = t.rows[0].cells[j]; c.width = Inches(2.9); c.text = ""
    for k, line in enumerate(txt[j]):
        p = c.paragraphs[0] if k == 0 else c.add_paragraph(); p.style = "RCell"
        r = p.add_run(line); r.bold = k in (0, 2); r.font.size = Pt(12)
        p.alignment = L if j == 0 else WD_ALIGN_PARAGRAPH.RIGHT

# section 1 ends (cover + bonafide), roman numbering
doc.sections[-1]  # placeholder
sec1 = doc.sections[0]
new = doc.add_section(WD_SECTION.NEW_PAGE)

# ======================= ABSTRACT =======================
r12 = R12; reg = r12["regression"]; gbc = R5["gb_change"]
best_b4 = R5["regression"]["Gradient Boosting"]["B4 Red"]["r2"]; lin_b4 = R5["regression"]["Linear (Unit 2, single band)"]["B4 Red"]["r2"]
best_b5 = R5["regression"]["Gradient Boosting"]["B5 NIR"]["r2"]; lin_b5 = R5["regression"]["Linear (Unit 2, single band)"]["B5 NIR"]["r2"]
doc.add_paragraph("ABSTRACT", style="RFront")
abstract = (
    f"Monitoring surface change around rapidly urbanising regions such as Noida requires methods that work without labelled ground truth. "
    f"This project develops the Landsat Change Explorer, an integrated machine learning pipeline that analyses two "
    f"Landsat 8 Collection 2 Level-2 surface-reflectance scenes of path/row 146/040 acquired on 13 December 2015 and 30 July 2024. After checksum verification, radiometric scaling, cloud and shadow masking and alignment to a common "
    f"150 m grid, {n0(r12['common'])} pixels were valid on both dates. Exploratory analysis and polynomial curve fitting were followed by a Gaussian maximum-likelihood "
    f"classifier, logistic regression, a support vector machine and band-wise linear regression whose robust residual score flagged {n0(r12['changed'])} pixels "
    f"({f2(r12['change_pct'])}%) as possible spectral change. K-means, Gaussian mixtures fitted by expectation–maximisation, hierarchical clustering and principal "
    f"component analysis revealed four spectral groups and a dominant temporal-contrast component. A Gaussian hidden Markov model implemented from scratch on row "
    f"transects improved held-out log-likelihood from {f3(R4['gmm_test_ll_per_obs'])} to {f3(R4['hmm_test_ll_per_obs'])} per observation over an independent mixture "
    f"and produced spatially coherent states. Classification and regression trees, random forests, AdaBoost, gradient boosting, additive models and Bayesian model averaging "
    f"raised spatial hold-out R² for near-infrared prediction from {f3(lin_b5)} to {f3(best_b5)}, and the boosted change map agreed with the linear map at a "
    f"Jaccard index of {f2(gbc['jaccard_with_linear'])}. Low R² values confirm strong seasonal differences. The framework is reproducible and transparent, "
    f"but independent labels are required before any land-cover change can be confirmed.")
para(abstract)
ABS_WORDS = len(abstract.split())

# ======================= TOC / LOF / LOT =======================
p = doc.add_paragraph("TABLE OF CONTENTS", style="RFront"); p.paragraph_format.page_break_before = True
p = doc.add_paragraph(style="RBody"); field(p, 'TOC \\h \\z \\t "RChapter,1,RFront,1"')
p = doc.add_paragraph("LIST OF FIGURES", style="RFront"); p.paragraph_format.page_break_before = True
p = doc.add_paragraph(style="RBody"); field(p, 'TOC \\h \\z \\t "RFig,1"')
p = doc.add_paragraph("LIST OF TABLES", style="RFront"); p.paragraph_format.space_before = Pt(18)
p = doc.add_paragraph(style="RBody"); field(p, 'TOC \\h \\z \\t "RTab,1"')

p = doc.add_paragraph("ABBREVIATIONS", style="RFront"); p.paragraph_format.page_break_before = True
abbr = [("ARI", "Adjusted Rand Index"), ("BIC", "Bayesian Information Criterion"), ("BMA", "Bayesian Model Averaging"),
        ("CART", "Classification and Regression Trees"), ("CH", "Calinski–Harabasz index"), ("CRS", "Coordinate Reference System"),
        ("CV", "Cross-Validation"), ("DB", "Davies–Bouldin index"), ("DN", "Digital Number"), ("EDA", "Exploratory Data Analysis"),
        ("EM", "Expectation–Maximisation"), ("GAM", "Generalised Additive Model"), ("GBM", "Gradient Boosting Machine"),
        ("GMM", "Gaussian Mixture Model"), ("HMM", "Hidden Markov Model"), ("L2SP", "Level-2 Science Product (surface reflectance + temperature)"),
        ("MAD", "Median Absolute Deviation"), ("MLE", "Maximum Likelihood Estimation"), ("NDVI", "Normalised Difference Vegetation Index"),
        ("NDWI", "Normalised Difference Water Index"), ("NIR", "Near Infrared"), ("NMI", "Normalised Mutual Information"),
        ("OLS", "Ordinary Least Squares"), ("PCA", "Principal Component Analysis"), ("QA", "Quality Assessment"), ("RF", "Random Forest"),
        ("RMSE", "Root Mean Square Error"), ("SR", "Surface Reflectance"), ("SVM", "Support Vector Machine"), ("USGS", "United States Geological Survey"),
        ("UTM", "Universal Transverse Mercator")]
table(None, ["Abbreviation", "Expansion"], abbr, [1.5, 4.2], align_num=False)

# section 2 ends (front matter, roman continues); chapters arabic from 1
doc.add_section(WD_SECTION.NEW_PAGE)

# ======================= CHAPTER 1 =======================
chapter(1, "Introduction")
section("1.1 Background, Motivation and Scope")
para("Rapid urban expansion in the National Capital Region of India, including Noida and Greater Noida, converts agricultural land, scrub and "
     "seasonal water bodies into built-up surfaces. Planners, environmental agencies, researchers and citizens need objective, repeatable evidence of where "
     "the land surface has changed. Multispectral satellites such as Landsat 8 provide free, radiometrically calibrated observations at 30 m resolution "
     "every 16 days, making them well suited for regional change screening. However, two acquisitions differ not only because of genuine land-cover change but "
     "also because of season, atmosphere, clouds, illumination and geometric registration. Labelled ground truth is rarely available, so methods must be "
     "transparent about what they can and cannot conclude.")
para("The Landsat Change Explorer project addresses this need with a local, GPU-free pipeline and a Streamlit application. The scope covers two full Landsat 8 "
     "Collection 2 Level-2 scenes (path 146, row 040) from 13 December 2015 and 30 July 2024, four reflective bands (blue, green, red and near infrared) and the "
     "QA_PIXEL quality band. The analysis is performed on the full scene footprint at a 150 m demonstration grid; no administrative Noida boundary is applied. "
     "The outputs are possible-spectral-change indicators and provisional spectral classes, not validated land-use statistics.")
section("1.2 Integrated Problem Statement")
para("Given two co-located but seasonally different Landsat 8 surface-reflectance scenes without independent ground-truth labels, design, implement and "
     "evaluate machine learning methods that (i) describe and preprocess the data reliably, (ii) classify pixels into interpretable spectral classes, "
     "(iii) model the expected 2015-to-2024 reflectance relationship and flag pixels that depart from it, (iv) discover latent structure without labels, "
     "(v) exploit the spatial ordering of pixels as sequential data, and (vi) test whether combined non-linear models improve on the linear baseline.")
para("All five units use this one dataset: Unit 4 treats each east\u2013west image row as an ordered observation sequence, so no separate sequential dataset was required, "
     "and Unit 5 revisits the Unit 2 tasks under an identical evaluation design.")
section("1.3 Project Objectives")
objs = [("Objective 1 (Unit 1): ", "Load, verify and statistically describe the scenes (distributions, correlation, NDVI, missing data) and use polynomial curve fitting to choose model complexity for the date-to-date relationship."),
        ("Objective 2 (Unit 2): ", "Implement a Gaussian maximum-likelihood classifier, compare it with logistic regression and an SVM, and fit least-squares, ridge and Bayesian linear regression to derive a robust possible-change score."),
        ("Objective 3 (Unit 3): ", "Apply K-means, Gaussian mixtures (EM) and hierarchical clustering, select the number of clusters, and use PCA to describe dominant spectral and temporal variation."),
        ("Objective 4 (Unit 4): ", "Implement a Gaussian HMM with scaled forward–backward, Baum–Welch and Viterbi algorithms on pixel transects and compare it with an independent mixture model."),
        ("Objective 5 (Unit 5): ", "Evaluate CART, random forest, AdaBoost, gradient boosting, a generalised additive model and Bayesian model averaging against the Unit 2 baselines with spatial block validation.")]
for b, t in objs: bullet(t, b)
section("1.4 Unit-wise Topic and Dataset Allocation")
table("Table 1.1 Unit-wise topic and dataset allocation", ["PBL", "Unit", "Allotted topic / algorithm", "Dataset"], [
    ("PBL 1", "1", "Data loading, descriptive statistics, probability densities, polynomial curve fitting", "Original (Landsat 8, 2015 & 2024)"),
    ("PBL 1", "2", "Least squares / MLE, ridge & Bayesian linear regression, Gaussian generative classifier, logistic regression, SVM", "Original"),
    ("PBL 2", "3", "K-means, mixtures of Gaussians (EM), hierarchical clustering, PCA", "Original"),
    ("PBL 2", "4", "Hidden Markov model: forward–backward, Baum–Welch, Viterbi", "Original (row transects as sequences)"),
    ("PBL 3", "5", "CART, random forest, AdaBoost, gradient boosting, GAM, Bayesian model averaging", "Original")],
    [0.6, 0.45, 3.0, 1.7], align_num=False)

# ======================= CHAPTER 2 =======================
chapter(2, "Data Analysis and Fundamental Machine Learning Techniques (Units 1 and 2)")
section("2.1 Overview of Allotted Topics")
para("**Unit 1** introduces machine learning, probability theory and polynomial curve fitting. In this project, probability densities are used to describe "
     "the reflectance of each band (mean, variance, quantiles and covariance), and polynomial curve fitting is used to study how model complexity affects "
     "the prediction of 2024 reflectance from 2015 reflectance. **Unit 2** covers linear models. Least-squares regression is equivalent to maximum-likelihood "
     "estimation under Gaussian noise and is used to model the expected relationship between the dates; ridge and Bayesian linear regression add "
     "regularisation. For classification, a probabilistic generative model with class-conditional Gaussian densities (the Gaussian MLE classifier) is the core "
     "method, compared with discriminative logistic regression and a kernel SVM. These methods suit the problem because reflectance is continuous, approximately "
     "unimodal within surface types and strongly correlated between visible bands.")
section("2.2 Dataset Description")
e15, e24 = R12["eda"]["2015"], R12["eda"]["2024"]
table("Table 2.1 Dataset summary", ["Item", "Value"], [
    ("Source", "USGS EarthExplorer, Landsat 8 OLI Collection 2 Level-2 (L2SP); project copy on Google Drive with SHA-256 manifest"),
    ("Before scene", "LC08_L2SP_146040_20151213_20200908_02_T1 (13 Dec 2015)"),
    ("After scene", "LC08_L2SP_146040_20240730_20240801_02_T1 (30 Jul 2024)"),
    ("Files", "10 GeoTIFFs (SR_B2, SR_B3, SR_B4, SR_B5, QA_PIXEL per date), 767,745,918 bytes"),
    ("Native grid", "7,681 × 7,531 pixels, 30 m, EPSG:32643 (UTM 43N), uint16"),
    ("Analysis grid", f"{R12['shape'][0]:,} × {R12['shape'][1]:,} pixels (≈150 m), {R12['shape'][0]*R12['shape'][1]:,} cells"),
    ("Missing / invalid", f"2015: {f2(R12['mask_pct']['2015_invalid'])}% of grid; 2024: {f2(R12['mask_pct']['2024_invalid'])}% (fill, cloud, shadow, cirrus, snow)"),
    ("Common valid pixels", f"{n0(R12['common'])}"),
], [1.6, 4.1], align_num=False)
para("The scenes come from different seasons: December is the dry post-monsoon winter, while July is the peak of the south-west monsoon. The 2024 scene "
     "origin is 2,100 m west of the 2015 origin, so geometric alignment is mandatory. Limitations include heavy monsoon cloud in 2024, lack of QA_RADSAT "
     "saturation flags and the absence of any labelled ground truth.")
section("2.3 Data Preprocessing")
para("Two bands supplied as 20 MB transfer parts were reassembled and all ten files matched the SHA-256 manifest. Filenames, required bands, date order, "
     "uint16 type, a shared source grid and geographic overlap are validated before processing. Digital numbers are converted to surface reflectance with the "
     "USGS coefficients (Eq. 2.1); pixels with DN = 0 or any QA_PIXEL bit 0–5 set (fill, dilated cloud, cirrus, cloud, shadow, snow) are masked, while water is retained. "
     "Both dates are reprojected with nearest-neighbour resampling onto a grid derived from the 2015 red band with a factor of 5 (≈150 m). NDVI, NDWI (Eqs. 2.2–2.3) "
     "and visible brightness are derived, and models that need scaling use z-scores fitted on training data. With random seed 42, regression uses 100,000 common "
     "pixels and classifiers use 40,000 training and 20,000 test pixels with unambiguous seed labels.")
eq("ρ = 0.0000275 × DN − 0.2", "2.1")
eq("NDVI = (ρ_NIR − ρ_Red) / (ρ_NIR + ρ_Red)", "2.2")
eq("NDWI = (ρ_Green − ρ_NIR) / (ρ_Green + ρ_NIR)", "2.3")
section("2.4 Exploratory Data Analysis (EDA)")
rows = []
for b in BN:
    a, c = e15[b], e24[b]
    rows.append((b, f"{f4(a['mean'])} / {f4(c['mean'])}", f"{f4(a['std'])} / {f4(c['std'])}", f"{f4(a['median'])} / {f4(c['median'])}", f"{f2(a['skew'])} / {f2(c['skew'])}"))
rows.append(("NDVI", f"{f3(e15['ndvi_mean'])} / {f3(e24['ndvi_mean'])}", f"{f3(e15['ndvi_std'])} / {f3(e24['ndvi_std'])}", "–", "–"))
table("Table 2.2 Descriptive statistics of valid pixels (2015 / 2024)", ["Band", "Mean", "Std. dev.", "Median", "Skewness"], rows, [1.1, 1.25, 1.25, 1.25, 0.9])
para(f"Table 2.2 shows that the near-infrared mean rose from {f3(e15['B5 NIR']['mean'])} to {f3(e24['B5 NIR']['mean'])} and mean NDVI from "
     f"{f3(e15['ndvi_mean'])} to {f3(e24['ndvi_mean'])}, while red reflectance fell. This is the expected signature of monsoon vegetation growth and "
     "indicates that a large part of the date-to-date difference is seasonal rather than permanent change. The 2024 visible bands are strongly right-skewed "
     "because bright built-up surfaces and cloud edges form a long tail.")
para("The distributions confirm the shift: the 2024 NIR distribution moves right and widens, whereas visible bands become darker with a long bright tail, so a single "
     f"global threshold on raw reflectance cannot separate change from season. Pearson correlation shows that the visible bands are highly inter-correlated "
     f"(r \u2248 {f2(e15['corr'][0][1])}\u2013{f2(e15['corr'][1][2])}) while NIR is nearly uncorrelated with them (r = {f2(e15['corr'][0][3])} to {f2(e15['corr'][1][3])} in 2015): "
     "the data have two main dimensions, brightness and vegetation vigour, which motivates full-covariance Gaussian models and PCA.")
figure("eda_ndvi.png", "Figure 2.1 NDVI maps on the common valid area and NDVI distributions", 4.6)
para("Figure 2.1 shows widespread greening in July 2024 and the irregular holes produced by cloud masking, which reduce the common area to "
     f"{f2(100*R12['common']/(R12['shape'][0]*R12['shape'][1]))}% of the grid.")
section("2.5 Proposed System Architecture / Workflow")
figure("wf_overall.png", "Figure 2.2 Overall architecture of the integrated project", 3.8)
figure("wf_unit12.png", "Figure 2.3 Units 1–2 workflow", 3.4)
para("Figure 2.2 shows the complete project from verified inputs through preprocessing to the unit-wise models and outputs. Figure 2.3 details Units 1–2: "
     "seed labels train the Gaussian MLE classifier, band-wise regression predicts 2024 from 2015, and standardised residuals are combined and thresholded robustly.")
section("2.6 Algorithm Theory and Mathematical Formulation")
sub("2.6.1 Polynomial curve fitting")
para("For an input x (2015 reflectance) and target t (2024 reflectance), a polynomial of order M is fitted by minimising the sum-of-squares error:")
eq("y(x, w) = Σⱼ₌₀ᴹ wⱼ xʲ,     E(w) = ½ Σₙ { y(xₙ, w) − tₙ }²", "2.4")
sub("2.6.2 Linear regression as maximum likelihood")
para("Assuming t = wᵀφ(x) + ε with ε ~ N(0, β⁻¹), maximising the likelihood is equivalent to least squares. Ridge regression adds an L2 penalty, "
     "and Bayesian linear regression places a Gaussian prior on w, giving a posterior mean equal to a ridge solution with an automatically estimated penalty:")
eq("w_ML = (ΦᵀΦ)⁻¹ Φᵀt,     w_ridge = (λI + ΦᵀΦ)⁻¹ Φᵀt", "2.5")
sub("2.6.3 Gaussian maximum-likelihood classifier")
para("Each class k has mean μₖ and covariance Σₖ estimated by MLE from at most 5,000 seed pixels (with 10⁻⁶I added for stability). A pixel "
     "x ∈ ℝ⁴ is assigned to the class with the highest Gaussian log-likelihood (equal priors):")
eq("gₖ(x) = −½ ln|Σₖ| − ½ (x − μₖ)ᵀ Σₖ⁻¹ (x − μₖ),     ŷ = argmaxₖ gₖ(x)", "2.6")
para("Seed labels come from quantile rules on 2015 NDVI, NDWI and brightness (e.g. Vegetation-like: NDVI ≥ q₀.₈₀; Water-like: NDWI ≥ q₀.₉₀ and NDVI ≤ q₀.₃₀). "
     "Logistic regression models p(Cₖ|x) with a softmax of linear functions; the SVM uses an RBF kernel (C = 10).")
sub("2.6.4 Robust residual change score")
para("For each band b a simple regression ρ²⁰²⁴ = a_b ρ²⁰¹⁵ + c_b is fitted. Residuals r_b are scaled by their median absolute value and "
     "combined; pixels above a MAD-based threshold are flagged as possible spectral change:")
eq("s = √( ¼ Σ_b (r_b / median|r_b|)² ),   flag: s > med(s) + 3(1.4826)MAD(s)", "2.7")
section("2.7 Results and Performance Evaluation")
sub("2.7.1 Polynomial degree selection (Unit 1)")
pr = R12["poly"]
table("Table 2.3 Test RMSE of polynomial fits (2015 → 2024) by degree M", ["Band", "M = 1", "M = 2", "M = 3", "M = 4", "M = 6"],
      [(b, *[f4(pr[b][str(d)]["test_rmse"]) for d in (1, 2, 3, 4, 6)]) for b in BN], [1.2, 0.9, 0.9, 0.9, 0.9, 0.9])
para("Training error decreases monotonically with M, but test error is lowest at M = 1–2 and grows sharply for M ≥ 4 (Table 2.3): "
     "high-order polynomials over-fit and extrapolate wildly at extreme reflectances. The quadratic term helps only for NIR "
     f"({f4(pr['B5 NIR']['1']['test_rmse'])} → {f4(pr['B5 NIR']['2']['test_rmse'])}). The linear model was therefore retained for the change score, consistent with Occam's razor.")
sub("2.7.2 Linear regression and change detection (Unit 2)")
table("Table 2.4 Band-wise regression of 2024 on 2015 reflectance (100,000 training pixels)", ["Band", "Slope", "Intercept", "R² (sample)", "RMSE (all common)"],
      [(b, f4(reg[k]["slope"]), f4(reg[k]["intercept"]), f4(reg[k]["r2_sample"]), f4(reg[k]["rmse_all"])) for b, k in zip(BN, ["SR_B2", "SR_B3", "SR_B4", "SR_B5"])],
      [1.2, 1.0, 1.0, 1.1, 1.3])
figure("regression_scatter.png", "Figure 2.4 2015 versus 2024 reflectance with fitted regression lines", 4.8)
mr = R12["multi_reg_B4"]
para(f"Slopes well below one and low R² values (Table 2.4, Figure 2.4) show that 2015 reflectance explains only a small part of 2024 reflectance; NIR has the "
     f"weakest relation (R² = {f3(reg['SR_B5']['r2_sample'])}) because monsoon vegetation changes NIR non-uniformly. Using all four bands for red "
     f"(OLS R² = {f3(mr['OLS (4 bands)']['r2'])}, ridge {f3(mr['Ridge (alpha=1)']['r2'])}, Bayesian ridge {f3(mr['Bayesian ridge']['r2'])}) gives only a slight improvement, "
     "and Bayesian ridge selects almost no shrinkage because 80,000 samples dominate the prior. R² here measures model fit, not classification accuracy.")
figure("change_map.png", "Figure 2.5 Residual score map, possible-change map and score distribution", 4.8)
para(f"The robust threshold was {f3(R12['threshold'])}; {n0(R12['changed'])} of {n0(R12['common'])} common pixels ({f2(R12['change_pct'])}%) were flagged and "
     f"{n0(R12['unchanged'])} were unflagged (Figure 2.5). These values exactly reproduce the project's saved baseline. Flagged pixels form compact clusters "
     "and linear features rather than random noise, which is consistent with real surface modification, although clouds edges and seasonal effects may also contribute.")
sub("2.7.3 Gaussian MLE classification (Unit 2)")
mc = R12["mle_common"]
table("Table 2.5 MLE class counts on the common valid area", ["Class", "2015 pixels", "2015 %", "2024 pixels", "2024 %"],
      [(CN[i], n0(mc["2015"][i]), f2(100*mc["2015"][i]/R12['common']), n0(mc["2024"][i]), f2(100*mc["2024"][i]/R12['common'])) for i in range(4)],
      [1.9, 1.0, 0.8, 1.0, 0.8])
figure("mle_maps.png", "Figure 2.6 Gaussian MLE class maps for 2015 and 2024", 4.3)
tr_ = R12["mle_transition"]
para(f"Vegetation-like pixels increase from {f2(100*mc['2015'][1]/R12['common'])}% to {f2(100*mc['2024'][1]/R12['common'])}% (Table 2.5, Figure 2.6). The transition "
     f"table shows that {n0(tr_[3][1])} Bare/other-like pixels of 2015 became Vegetation-like in 2024, which reflects the December-versus-July season rather than "
     "permanent conversion. Because the classes are defined by rules on the 2015 image, the same thresholds are not seasonally invariant when transferred to 2024.")
uc = R12["u2_classifiers"]
para(f"On 20,000 held-out seed pixels the Gaussian MLE, logistic regression and RBF-SVM reached accuracies of {f4(uc['Gaussian MLE']['acc'])}, "
     f"{f4(uc['Logistic regression']['acc'])} and {f4(uc['SVM (RBF)']['acc'])} (full results in Table 4.2). The generative MLE model is slightly lower because the rule-based class "
     "boundaries are not Gaussian; the discriminative models fit the boundaries more flexibly. These values measure agreement with provisional labels, "
     "not true accuracy, since no independent ground truth exists.")
section("2.8 Discussion and Findings")
for t in ["Preprocessing was decisive: cloud masking removed 49% (2015) and 65% (2024) of the grid and alignment corrected a 2.1 km offset; EDA revealed a strong seasonal shift (NDVI +0.22) that explains the low regression R\u00b2.",
          "Polynomial fitting showed that extra flexibility does not generalise, and the robust residual score isolates the 5.24% of pixels least consistent with the global date-to-date relationship, a more defensible criterion than class-map differencing.",
          "Limitations: no ground truth, provisional rule-based classes, coarse 150 m grid and seasonal mismatch."]:
    bullet(t)
section("2.9 Chapter Conclusion")
para("PBL 1 delivered a verified, reproducible preprocessing pipeline, a statistical characterisation of both scenes, a Gaussian MLE classifier and a regression-"
     f"based change detector that flags {f2(R12['change_pct'])}% of the common area. These results form the baseline against which the unsupervised, sequential and ensemble methods of later chapters are compared.")

# ======================= CHAPTER 3 =======================
chapter(3, "Unsupervised Learning and Sequential Data Modelling (Units 3 and 4)")
section("3.1 Overview of Allotted Unit 3 and Unit 4 Topics")
para("**Unit 3** covers K-means clustering, mixtures of Gaussians and the EM algorithm, factor analysis, PCA and hierarchical clustering. These are applied to "
     "discover natural spectral groupings without relying on the provisional rule-based labels and to summarise the correlated bands. **Unit 4** covers "
     "Markov models, hidden Markov models, maximum likelihood for HMMs, the forward–backward algorithm with scaling factors and the Viterbi algorithm. "
     "Because neighbouring pixels on the ground are strongly dependent, each east–west row of the image is treated as an ordered sequence of observations, "
     "and an HMM is used to infer latent surface regimes along it.")
section("3.2 Unit 3 – Dataset Description and Preprocessing")
para(f"The original dataset and preprocessing of Chapter 2 are reused. Additional steps for Unit 3: (i) the 2015 valid pixels ({n0(R3['n_pixels'])}) are "
     "z-score standardised per band, because K-means and hierarchical clustering are distance-based and NIR has the largest variance; (ii) models are fitted on "
     "a random sample of 200,000 pixels and internal indices are computed on 20,000 pixels to keep computation tractable; (iii) for bi-temporal PCA the four bands "
     f"of both dates are stacked into an 8-dimensional vector for the {n0(R12['common'])} common pixels and standardised.")
section("3.3 Unit 3 – Theory, EDA, Implementation and Results")
para("Chapter 2 EDA showed that the visible bands are highly correlated and NIR is nearly independent of them. The additional analysis here concerns the "
     "choice of the number of clusters K, using K-means inertia (elbow), the silhouette coefficient and GMM BIC for K = 2\u20138.")
ks = R3["k_selection"]
para(f"The silhouette is highest at K = 2 ({f3(ks['silhouette'][0])}, essentially vegetation versus non-vegetation) and has a secondary peak at K = 4 "
     f"({f3(ks['silhouette'][2])}). The inertia curve bends at K = 4 and BIC improvements become small beyond K = 4. K = 4 was therefore chosen, which also matches the four MLE classes and allows direct comparison.")
section("3.4 Unit 3 – Algorithm Theory and Mathematical Formulation")
para("**K-means** minimises the within-cluster sum of squares by alternating assignment and mean-update steps:")
eq("J = Σₙ Σₖ rₙₖ ‖xₙ − μₖ‖²,     μₖ = Σₙ rₙₖ xₙ / Σₙ rₙₖ", "3.1")
para("**Gaussian mixture model** p(x) = Σₖ πₖ N(x | μₖ, Σₖ) is fitted by EM. The E-step computes responsibilities and the M-step re-estimates parameters:")
eq("γ(zₙₖ) = πₖ N(xₙ|μₖ,Σₖ) / Σⱼ πⱼ N(xₙ|μⱼ,Σⱼ),   μₖ = Σₙγₙₖxₙ / Nₖ,   πₖ = Nₖ / N", "3.2")
para("**Hierarchical clustering** merges clusters agglomeratively; Ward linkage merges the pair that least increases total within-cluster variance, while average linkage uses mean pairwise distance. "
     "**PCA** finds orthogonal directions of maximum variance as eigenvectors of the sample covariance S:")
eq("S uᵢ = λᵢ uᵢ,     explained ratioᵢ = λᵢ / Σⱼ λⱼ", "3.3")
para("Quality is measured by the silhouette (higher is better), Davies–Bouldin (lower is better) and Calinski–Harabasz (higher is better) indices, and agreement with "
     "the MLE classes by ARI and NMI. A bi-temporal PCA reconstruction error (three components) is thresholded with the Eq. (2.7) rule as an alternative change score.")
section("3.5 Unit 3 Proposed System Architecture / Workflow")
figure("wf_unit3.png", "Figure 3.1 Unit 3 workflow", 3.4)
section("3.6 Unit 3 – Results and Performance Evaluation")
cl = R3["clustering"]
table("Table 3.1 Clustering evaluation (K = 4)", ["Method", "Silhouette ↑", "DB ↓", "CH ↑", "ARI vs MLE", "NMI vs MLE"],
      [(k, f3(v["silhouette"]), f3(v["davies_bouldin"]), n0(v["calinski_harabasz"]), f3(v["ARI_vs_MLE"]), f3(v["NMI_vs_MLE"])) for k, v in cl.items()],
      [1.8, 0.85, 0.7, 0.85, 0.85, 0.85])
para("Hierarchical indices are computed on an 8,000-pixel sample. Average linkage obtains the best silhouette and DB values only because it isolates a handful of outliers "
     f"(cluster sizes {cl['Hierarchical (average)']['sizes']}), so it is not a useful partition; this illustrates why internal indices must be read together with cluster sizes. "
     f"K-means gives the best balanced partition (silhouette {f3(cl['K-means']['silhouette'])}, DB {f3(cl['K-means']['davies_bouldin'])}) and the highest agreement with the MLE classes "
     f"(ARI {f3(cl['K-means']['ARI_vs_MLE'])}). The GMM, which allows elliptical clusters, fits the data density better (EM converged in {R3['gmm_iter']} iterations) but its overlapping "
     "components lower the silhouette. Moderate ARI values show that natural spectral groups only partly coincide with rule-based classes.")
figure("u3_cluster_maps.png", "Figure 3.2 K-means and GMM cluster maps (2015) and Ward dendrogram", 4.6)
para("Figure 3.2 shows that both clustering methods recover the main vegetated belt and the bright south-western region. The dendrogram shows one large early split "
     "(vegetated vs. non-vegetated) followed by three smaller groups, supporting the choice of K = 4.")
p4, p8 = R3["pca4_explained"], R3["pca8_explained"]; pc = R3["pca_change"]
figure("u3_pca.png", "Figure 3.3 Scree plots and bi-temporal PC1–PC2 scatter", 4.6)
para(f"For a single date, PC1 and PC2 explain {f2(100*p4[0])}% and {f2(100*p4[1])}% of the variance: PC1 has near-equal loadings on the visible bands (brightness) and PC2 is dominated by NIR "
     "(vegetation). In the bi-temporal PCA, PC1 is overall brightness on both dates, while PC2 has negative visible loadings for 2015 and positive ones for 2024; it is "
     f"therefore a temporal-contrast (change) axis explaining {f2(100*p8[1])}% of variance. The PCA reconstruction-error score flagged {f2(pc['flag_pct'])}% of pixels but overlapped "
     f"weakly with the regression flags (Jaccard {f3(pc['overlap_jaccard'])}), because a 3-component subspace already contains the temporal-contrast axis and the residual mainly "
     "captures rare band-shape anomalies.")
section("3.7 Unit 4 – Sequential Dataset Description")
para(f"No separate dataset was used. Sequences are east–west transects (image rows) of the 150 m analysis grid. Each observation xₜ ∈ ℝ⁴ is the standardised "
     "B2–B5 reflectance of a pixel; the sequence index t is position along the row (150 m sampling interval). Cloud gaps split rows into contiguous valid runs; runs "
     f"shorter than 30 pixels are discarded. Rows with index r ≡ 0 (mod 15) form the training set ({R4['n_train_seq']} sequences, {n0(R4['n_train_obs'])} observations) and rows "
     f"r ≡ 7 (mod 15) form the test set ({R4['n_test_seq']} sequences, {n0(R4['n_test_obs'])} observations), so no row is shared. Limitations: ordering is spatial rather than temporal, "
     "and gaps from masked pixels break sequences.")
section("3.8 Unit 4 – Sequential Preprocessing and EDA")
para("Row order is preserved, sequences break at every masked pixel and standardisation uses 2015 statistics. Along a row, red and NIR reflectance change in blocks "
     "tens of pixels long; this spatial autocorrelation is what an HMM exploits and an independent mixture ignores.")
section("3.9 Unit 4 – Mathematical Formulation")
para("Hidden states zₜ ∈ {1,…,K} (K = 4 latent surface regimes) follow a first-order Markov chain with initial distribution π and transition matrix A; observations "
     "have Gaussian emissions with full covariance:")
eq("p(X, Z | θ) = p(z₁|π) Πₜ₌₂ᵀ p(zₜ|zₜ₋₁, A) Πₜ₌₁ᵀ N(xₜ | μ_{zₜ}, Σ_{zₜ})", "3.4")
para("**Scaled forward–backward:** with normalisers cₜ, α̂(zₜ) = p(xₜ|zₜ) Σ α̂(zₜ₋₁) A / cₜ and β̂(zₜ) = Σ β̂(zₜ₊₁) p(xₜ₊₁|zₜ₊₁) A / cₜ₊₁, giving")
eq("γ(zₜ) = α̂(zₜ) β̂(zₜ),   ξ(zₜ₋₁, zₜ) = α̂(zₜ₋₁) p(xₜ|zₜ) A β̂(zₜ) / cₜ,   ln p(X) = Σₜ ln cₜ", "3.5")
para("**Baum–Welch (EM)** re-estimates Aⱼₖ = Σₜ ξ(zₜ₋₁,ⱼ, zₜ,ₖ) / Σₜ Σₗ ξ(zₜ₋₁,ⱼ, zₜ,ₗ), πₖ ∝ γ(z₁ₖ), and μₖ, Σₖ as "
     "γ-weighted means and covariances (plus 10⁻⁴I). Parameters were initialised from K-means with Aₖₖ = 0.9. **Viterbi** decoding finds the most probable state path:")
eq("ω(zₜ) = ln p(xₜ|zₜ) + max_{zₜ₋₁} { ln A(zₜ₋₁, zₜ) + ω(zₜ₋₁) }", "3.6")
para("The model is evaluated by held-out log-likelihood per observation against a GMM with the same K and emission family (an HMM without transitions), by mean run length of decoded labels, and by applying the trained model to 2024 to obtain a 2015→2024 state transition table. "
     "All HMM algorithms were implemented from scratch in NumPy.")
section("3.10 Unit 4 Proposed System Architecture / Workflow")
figure("wf_unit4.png", "Figure 3.4 Unit 4 workflow", 3.4)
para("Training stops when the log-likelihood change falls below 10⁻⁵ or after 40 iterations; Viterbi then labels every row of both dates.")
section("3.11 Results and Performance Evaluation")
para("Unit 4 results are summarised in Table 3.2 and Figures 3.5\u20133.6.")
A_ = R4["A"]
table("Table 3.2 HMM evaluation against an independent Gaussian mixture (K = 4)", ["Measure", "GMM (i.i.d.)", "HMM"], [
    ("Held-out log-likelihood per observation", f3(R4["gmm_test_ll_per_obs"]), f3(R4["hmm_test_ll_per_obs"])),
    ("Mean run length of labels along rows (pixels)", f2(R4["mean_run_gmm"]), f2(R4["mean_run_viterbi"])),
    ("ARI with MLE classes (2015)", f3(R4["ARI_gmm_vs_mle"]), f3(R4["ARI_viterbi_vs_mle"])),
    ("Self-transition probabilities Aₖₖ", "–", ", ".join(f2(A_[i][i]) for i in range(4))),
    ("Baum–Welch iterations / training time", "–", f"{R4['iterations']} / {R4['train_seconds']:.0f} s")], [3.0, 1.2, 1.5])
figure("u4_hmm_training.png", "Figure 3.5 Baum–Welch convergence, learned transition matrix and 2015→2024 state transitions", 4.8)
para(f"The HMM improves the held-out log-likelihood by {f3(R4['hmm_test_ll_per_obs'] - R4['gmm_test_ll_per_obs'])} nats per observation, confirming that spatial dependence "
     f"carries real information. The learned transition matrix is strongly diagonal (self-transition {f2(min(A_[i][i] for i in range(4)))}–{f2(max(A_[i][i] for i in range(4)))}), "
     f"so regimes persist for about {min(R4['expected_run_length_px']):.0f}–{max(R4['expected_run_length_px']):.0f} pixels (0.7–1.4 km). Log-likelihood increased monotonically across iterations (Figure 3.5), as guaranteed by EM; "
     "the 40-iteration limit was reached with only very small remaining gains.")
figure("u4_viterbi_maps.png", "Figure 3.6 GMM labels versus HMM Viterbi state maps", 4.8)
para(f"Viterbi labels form contiguous patches with a mean run of {f2(R4['mean_run_viterbi'])} pixels compared with {f2(R4['mean_run_gmm'])} for the GMM (Figure 3.6), suppressing "
     "salt-and-pepper noise. ARI with the MLE classes falls because the HMM states describe spatial regimes rather than pixel-level rule classes. When the 2015-trained model is applied to 2024, "
     f"{f2(R4['state_change_pct'])}% of pixels change state, mostly into the high-NIR state (Figure 3.5, right), again reflecting monsoon greening. Interestingly, pixels flagged by the "
     f"regression change score change state less often ({f2(R4['state_change_rate_flagged'])}%) than unflagged ones ({f2(R4['state_change_rate_unflagged'])}%): absolute state transfer captures the "
     "global seasonal shift, whereas the regression residual removes it and highlights local anomalies.")
section("3.12 Discussion, Findings and Limitations")
for t in ["K-means gave the most balanced clusters and GMM fitted the density better; average linkage shows that internal indices can reward degenerate partitions. PCA exposed a temporal-contrast axis that is a natural change direction.",
          "HMM states are spatial regimes along rows, not temporal evolution; row-only direction, cloud gaps and residual correlation between nearby train/test rows are limitations, and a 2-D Markov random field is the natural extension."]:
    bullet(t)
section("3.13 Chapter Conclusion")
para("Unit 3 justified four spectral groups and revealed brightness, vegetation and temporal-contrast axes; Unit 4 showed that an HMM exploiting spatial sequence "
     "dependence gives a better probabilistic model and coherent maps. The units answer different questions, so their scores are not compared directly.")

# ======================= CHAPTER 4 =======================
chapter(4, "Advanced Predictive Modelling and Ensemble Learning (Unit 5)")
section("4.1 Overview of Allotted Unit 5 Topics")
para("Unit 5 covers Bayesian model averaging, boosting, adaptive basis function models, CART, generalised additive models and ensemble learning. The low R\u00b2 of the "
     "Unit 2 regression suggests a non-linear, surface-type-dependent 2015\u21922024 relationship, so Unit 5 tests whether flexible and combined models predict 2024 "
     "reflectance better and whether this changes the possible-change map; the Unit 2 classification is also repeated with tree ensembles.")
section("4.2 Dataset Description and Preprocessing")
para(f"The original dataset and preprocessing of Chapter 2 are reused. Additional steps: features are the four 2015 bands plus NDVI and NDWI (six features); the "
     f"target is the 2024 reflectance of each band. To limit optimistic bias from spatial autocorrelation, the grid is divided into 50 × 50-pixel (7.5 km) blocks and "
     f"{R5['n_test_blocks']} of {R5['n_blocks']} blocks are held out for testing; {n0(R5['n_train'])} training and {n0(R5['n_test'])} test pixels are sampled from the "
     "respective blocks. Tree models need no scaling. The classification task reuses exactly the Unit 2 split.")
section("4.3 EDA and Feature Analysis")
imp = R5["rf_importance_B5"]
para(f"Random-forest importance for 2024 NIR (Figure 4.2, right) ranks NDWI ({f2(imp['NDWI'])}) and NDVI ({f2(imp['NDVI'])}) above every raw band "
     f"({f2(min(imp['B2'],imp['B3'],imp['B4'],imp['B5']))}\u2013{f2(max(imp['B2'],imp['B3'],imp['B4'],imp['B5']))}): the seasonal response depends mainly on surface type, which a single-band linear model cannot represent.")
section("4.4 Proposed Workflow / Architecture")
figure("wf_unit5.png", "Figure 4.1 Unit 5 workflow", 3.0)
para("Blocks are split spatially, CART depth is tuned by grouped CV, BMA weights come from a validation subset, and the best model's residuals rebuild the Eq. (2.7) change score.")
section("4.5 Algorithm Theory and Mathematical Formulation")
para("**CART** recursively partitions feature space with axis-aligned splits chosen to minimise squared error (regression) or Gini impurity (classification), giving a piecewise-constant adaptive basis function model:")
eq("f(x) = Σₘ wₘ I(x ∈ Rₘ),     Gini = Σₖ pₖ (1 − pₖ)", "4.1")
para("**Random forest** averages B decorrelated trees grown on bootstrap samples with random feature subsets; **AdaBoost.R2** fits trees sequentially to reweighted samples; "
     "**gradient boosting** adds trees that fit the negative gradient of the loss with shrinkage ν:")
eq("f_RF(x) = (1/B) Σ_b T_b(x),     Fₘ(x) = Fₘ₋₁(x) + ν hₘ(x),   hₘ ≈ −∂L/∂F", "4.2")
para("The **generalised additive model** uses cubic B-spline bases per feature with a ridge penalty, f(x) = β₀ + Σⱼ fⱼ(xⱼ). **Bayesian model averaging** weights "
     "models by their posterior probability, approximated by the Gaussian likelihood of a 15,000-pixel validation subset:")
eq("p(y|x, D) = Σₘ p(y|x, Mₘ) p(Mₘ|D),     p(Mₘ|D) ∝ exp(ℓₘ)", "4.3")
para("Hyperparameter settings for every model are listed in Appendix A (Table A.1).")
section("4.6 Results and Performance Evaluation")
rg = R5["regression"]
rows = [(k, *[f3(rg[k][b]["r2"]) for b in BN], f4(rg[k]["B5 NIR"]["rmse"])) for k in rg]
bm = R5["bma"]
rows.append(("Bayesian model averaging", *[f3(bm[b]["r2"]) for b in BN], f4(bm["B5 NIR"]["rmse"])))
table("Table 4.1 Spatial hold-out R² for predicting 2024 reflectance (30,000 test pixels)", ["Model", "B2", "B3", "B4", "B5", "RMSE B5"], rows, [2.2, 0.65, 0.65, 0.65, 0.65, 0.9])
figure("u5_regression.png", "Figure 4.2 CART depth selection, hold-out R² by model and band, and feature importance", 4.7)
para(f"Grouped CV selected a CART depth of {R5['cart_cv']['best_depth']}; deeper trees over-fit (Figure 4.2, left). Gradient boosting achieved the highest R\u00b2 on every band "
     f"(NIR {f3(rg['Gradient Boosting']['B5 NIR']['r2'])} vs {f3(rg['Linear (Unit 2, single band)']['B5 NIR']['r2'])} for the Unit 2 model), closely followed by the GAM and random forest, "
     "so most of the gain comes from smooth non-linear effects of the indices. AdaBoost.R2 was unstable on visible bands (negative R\u00b2) because its reweighting focuses on "
     f"cloud-edge outliers. BMA put almost all weight on the best models (NIR: GBM {f2(bm['B5 NIR']['weights']['Gradient Boosting'])}, RF {f2(bm['B5 NIR']['weights']['Random Forest'])}), "
     "matching but not exceeding the best single model.")
gc = R5["gb_change"]
figure("u5_change_compare.png", "Figure 4.3 Gradient-boosting residual change map and agreement with the linear change map", 3.4)
para(f"Gradient-boosting residuals in Eq. (2.7) flagged {f2(gc['flag_pct'])}% of common pixels versus {f2(gc['lin_flag_pct'])}% for the linear model, with Jaccard agreement "
     f"{f2(gc['jaccard_with_linear'])} (Figure 4.3): the core change clusters are stable across models, and disagreements lie mainly at class boundaries.")
cr = R5["classification"]
section("4.7 Comparison with Previous Models")
rows = [(k + " (Unit 2)", f4(v["acc"]), f4(v["f1"])) for k, v in R12["u2_classifiers"].items()] + [(k + " (Unit 5)", f4(v["acc"]), f4(v["f1"])) for k, v in cr.items()]
table("Table 4.2 Classification on the identical Unit 2 split (agreement with provisional seed labels)", ["Model", "Accuracy", "Macro F1"], rows, [2.8, 1.2, 1.2])
li = rg["Linear (Unit 2, single band)"]; gb = rg["Gradient Boosting"]
table("Table 4.3 Regression: Unit 2 linear baseline vs best Unit 5 model (same test blocks)", ["Band", "Linear R²", "GBM R²", "Linear RMSE", "GBM RMSE", "RMSE change"],
      [(b, f3(li[b]["r2"]), f3(gb[b]["r2"]), f4(li[b]["rmse"]), f4(gb[b]["rmse"]), f"{100*(gb[b]['rmse']-li[b]['rmse'])/li[b]['rmse']:.1f}%") for b in BN],
      [1.1, 0.9, 0.9, 1.0, 1.0, 1.0])
para("On the same task, test blocks and metric, every Unit 5 ensemble except AdaBoost outperformed the Unit 2 linear regression (Table 4.3), with RMSE reductions of about "
     f"{abs(100*(gb['B2 Blue']['rmse']-li['B2 Blue']['rmse'])/li['B2 Blue']['rmse']):.0f}\u2013{abs(100*(gb['B5 NIR']['rmse']-li['B5 NIR']['rmse'])/li['B5 NIR']['rmse']):.0f}%. In classification (Table 4.2), "
     f"gradient boosting ({f4(cr['Gradient Boosting']['acc'])}) and random forest ({f4(cr['Random Forest']['acc'])}) improve on the Gaussian MLE ({f4(R12['u2_classifiers']['Gaussian MLE']['acc'])}); "
     "since every model exceeds 0.98 agreement with rule-derived labels, this shows that trees recover threshold rules, not true accuracy.")
section("4.8 Discussion, Findings and Limitations")
para(f"Spatial block hold-out gives realistic estimates; CART and GAM stay interpretable, while RF and GBM need feature importance. GBM trained in \u2248{rg['Gradient Boosting']['B4 Red']['fit_s']:.0f} s per band "
     f"(RF \u2248{rg['Random Forest']['B4 Red']['fit_s']:.0f} s, AdaBoost \u2248{rg['AdaBoost (R2)']['B4 Red']['fit_s']:.0f} s), the best accuracy\u2013cost trade-off. Even the best R\u00b2 (\u22480.17) is low: season and clouds, not model capacity, limit performance.")
section("4.9 Chapter Conclusion")
para("Non-linear ensembles captured surface-type-dependent seasonal responses missed by the linear baseline, improving spatial hold-out prediction, and the possible-change "
     "hotspots proved robust to the model choice. Gradient boosting is recommended, with the linear model kept as a transparent reference.")

# ======================= CHAPTER 5 =======================
chapter(5, "Integrated Discussion and Findings")
para("Across PBL 1, PBL 2 and PBL 3, the project moved from description to supervised modelling, unsupervised structure, sequential dependence and model combination on one dataset.")
for b, t in [("Data quality dominates. ", "Masking and alignment determined which 632,388 pixels could be compared at all; without them every later model would mostly detect clouds and misregistration."),
             ("Season is the largest signal. ", "EDA (NDVI +0.22), MLE transitions, the PCA temporal-contrast axis and HMM state transfer all show a strong December–July vegetation shift. Methods that model the global relationship (regression residuals) are therefore more suitable for change screening than direct differencing of class maps."),
             ("Comparable models. ", "On the identical regression task, gradient boosting reduced RMSE by up to "
              f"{abs(100*(gb['B5 NIR']['rmse']-li['B5 NIR']['rmse'])/li['B5 NIR']['rmse']):.0f}% relative to linear regression; on the identical classification split, ensembles slightly improved on the Gaussian MLE. The HMM is compared only with the GMM, its natural non-sequential counterpart."),
             ("Robust change hotspots. ", f"Linear and boosted residual scores flagged {f2(R12['change_pct'])}% and {f2(gc['flag_pct'])}% of pixels with Jaccard agreement {f2(gc['jaccard_with_linear'])}, indicating stable hotspots."),
             ("Structure and context matter. ", "Clustering showed that natural spectral groups only partly match rule-based classes, and the HMM showed that spatial context improves the probabilistic model and map coherence.")]:
    bullet(t, b)
para("**Study limitations.** There is no independent ground truth, so no accuracy, confusion matrix against reality or area estimate of land-use change can be claimed. "
     "The two dates are seasonally mismatched, the analysis grid is coarse (150 m), QA_RADSAT saturation was not used, and the scene is not cropped to an administrative Noida boundary. "
     "Seed labels are rule-based, which makes classifier accuracies optimistic by construction.")

# ======================= CHAPTER 6 =======================
chapter(6, "Overall Conclusion and Future Scope")
para(f"The project built a verified, reproducible Landsat change-analysis pipeline and evaluated eleven model families from the course syllabus on it. The Unit 2 regression "
     f"residual method flagged {f2(R12['change_pct'])}% of the common area as possible spectral change; Unit 3 justified four spectral groups and exposed a temporal-contrast axis; "
     f"the Unit 4 HMM improved held-out log-likelihood by {f3(R4['hmm_test_ll_per_obs'] - R4['gmm_test_ll_per_obs'])} per observation and produced coherent maps; and Unit 5 ensembles improved "
     "spatial hold-out prediction over the linear baseline while confirming stable change hotspots. All results are delivered through notebooks, scripts and a Streamlit application.")
para("**Future scope:**")
for t in ["Use same-season acquisitions (e.g. two December scenes) or harmonised time series to separate change from phenology.",
          "Collect independently labelled validation points (field survey or high-resolution imagery) to report true accuracy and area estimates with confidence intervals.",
          "Work at native 30 m resolution with a defined Noida study boundary, and include QA_RADSAT and additional bands (SWIR, thermal).",
          "Extend the HMM to two-dimensional Markov random fields or temporal HMMs over many dates; explore deep sequence models such as LSTMs.",
          "Tune ensembles with Bayesian optimisation and add SHAP explanations; deploy the Streamlit app with cloud storage for larger datasets."]:
    bullet(t)

# ======================= REFERENCES =======================
p = doc.add_paragraph("REFERENCES", style="RFront"); p.paragraph_format.page_break_before = True
refs = [
 "C. M. Bishop, Pattern Recognition and Machine Learning. New York, NY, USA: Springer, 2006.",
 "K. P. Murphy, Machine Learning: A Probabilistic Perspective. Cambridge, MA, USA: MIT Press, 2012.",
 "U.S. Geological Survey, “Landsat 8 Collection 2 Level 2 Science Product Guide,” USGS EROS, Sioux Falls, SD, USA, 2020.",
 "U.S. Geological Survey, “Landsat Collection 2,” U.S. Geological Survey Fact Sheet 2021–3002, 2021.",
 "U.S. Geological Survey, “Landsat Collection 2 Quality Assessment Bands,” [Online]. Available: https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands",
 "J. W. Rouse, R. H. Haas, J. A. Schell, and D. W. Deering, “Monitoring vegetation systems in the Great Plains with ERTS,” in Proc. 3rd ERTS Symp., NASA SP-351, 1974, pp. 309–317.",
 "S. K. McFeeters, “The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features,” Int. J. Remote Sens., vol. 17, no. 7, pp. 1425–1432, 1996.",
 "Z. Zhu and C. E. Woodcock, “Continuous change detection and classification of land cover using all available Landsat data,” Remote Sens. Environ., vol. 144, pp. 152–171, 2014.",
 "C. Leys, C. Ley, O. Klein, P. Bernard, and L. Licata, “Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median,” J. Exp. Soc. Psychol., vol. 49, no. 4, pp. 764–766, 2013.",
 "S. Lloyd, “Least squares quantization in PCM,” IEEE Trans. Inf. Theory, vol. 28, no. 2, pp. 129–137, 1982.",
 "A. P. Dempster, N. M. Laird, and D. B. Rubin, “Maximum likelihood from incomplete data via the EM algorithm,” J. R. Stat. Soc. B, vol. 39, no. 1, pp. 1–38, 1977.",
 "P. J. Rousseeuw, “Silhouettes: A graphical aid to the interpretation and validation of cluster analysis,” J. Comput. Appl. Math., vol. 20, pp. 53–65, 1987.",
 "L. R. Rabiner, “A tutorial on hidden Markov models and selected applications in speech recognition,” Proc. IEEE, vol. 77, no. 2, pp. 257–286, 1989.",
 "A. J. Viterbi, “Error bounds for convolutional codes and an asymptotically optimum decoding algorithm,” IEEE Trans. Inf. Theory, vol. 13, no. 2, pp. 260–269, 1967.",
 "L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, Classification and Regression Trees. Belmont, CA, USA: Wadsworth, 1984.",
 "L. Breiman, “Random forests,” Mach. Learn., vol. 45, no. 1, pp. 5–32, 2001.",
 "Y. Freund and R. E. Schapire, “A decision-theoretic generalization of on-line learning and an application to boosting,” J. Comput. Syst. Sci., vol. 55, no. 1, pp. 119–139, 1997.",
 "H. Drucker, “Improving regressors using boosting techniques,” in Proc. 14th Int. Conf. Mach. Learn. (ICML), 1997, pp. 107–115.",
 "J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” Ann. Statist., vol. 29, no. 5, pp. 1189–1232, 2001.",
 "T. Hastie and R. Tibshirani, Generalized Additive Models. London, U.K.: Chapman & Hall, 1990.",
 "J. A. Hoeting, D. Madigan, A. E. Raftery, and C. T. Volinsky, “Bayesian model averaging: A tutorial,” Statist. Sci., vol. 14, no. 4, pp. 382–417, 1999.",
 "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph(style="RRef"); p.paragraph_format.left_indent = Inches(0.45); p.paragraph_format.first_line_indent = Inches(-0.45)
    p.add_run(f"[{i}]\t{r}"); p.paragraph_format.tab_stops.add_tab_stop(Inches(0.45))

# ======================= APPENDIX =======================
p = doc.add_paragraph("APPENDIX A", style="RFront"); p.paragraph_format.page_break_before = True; p.paragraph_format.space_after = Pt(2)
doc.add_paragraph("CODING", style="RFront")
para("**Repository:** https://github.com/Nithyon/landsat-change-explorer (Streamlit app, corrected Colab notebook Corrected_Landsat_Change_Detection_Colab.ipynb, verification scripts).")
para("**Dataset:** https://drive.google.com/drive/folders/1-mpfiI23HzrTgpsQpVCW8KUmGytMLkH- (ten GeoTIFFs and transfer parts).")
para("**Colab:** open the notebook from the repository in Google Colab, mount Google Drive and set LANDSAT_DATA_DIR to the dataset folder. The report "
     "scripts below run from the repository root after prepare_data.py (run order and settings in REPORT_ANALYSIS.md). The complete source code is in the repository "
     "and the accompanying code archive; it is not reproduced here to respect the 30-page report limit.")
table("Table A.1 Implementation files and configuration by chapter", ["Chapter / Unit", "File", "Methodology and configuration"], [
    ("Ch. 2 / Units 1\u20132", "landsat_analysis.py, u12_pipeline.py", "Scaling, QA masking, alignment; EDA; polynomial M = 1\u20136; ridge (\u03b1 = 1), Bayesian ridge; Gaussian MLE, logistic regression, RBF-SVM (C = 10); MAD change score; seed 42"),
    ("Ch. 3 / Unit 3", "u3_clustering_pca.py", "K = 2\u20138 selection; K-means (10 inits), full-covariance GMM, Ward/average linkage; 4- and 8-feature PCA"),
    ("Ch. 3 / Unit 4", "u4_hmm.py", "NumPy Gaussian HMM, K = 4; scaled forward\u2013backward, Baum\u2013Welch (\u2264 40 iter, tol 1e-5), Viterbi"),
    ("Ch. 4 / Unit 5", "u5_ensembles.py", f"50\u00d750 block split; CART depth {R5['cart_cv']['best_depth']}, leaf 20; RF 200 trees, depth 14; AdaBoost.R2 100 depth-6 trees; HistGBM 400 iter, lr 0.05; spline GAM (8 knots); BMA by validation likelihood"),
    ("App / checks", "app.py, verify_*.py, tests/", "Streamlit interface; SHA-256 manifest; regression test against saved baseline; 15 unit tests")],
    [1.15, 1.6, 2.95], align_num=False, size=10)

# ---------- sections: numbering ----------
secs = doc.sections
setup_section(secs[1], "lowerRoman", 2, show_number=True)
setup_section(secs[2], "lowerRoman", None, show_number=True)
setup_section(secs[3], "decimal", 1, show_number=True)
secs[0].footer.is_linked_to_previous = False
# continue roman in section 2 from section 1 (cover+bonafide = i, ii)
doc.save(out)
print("saved", out, "abstract words:", ABS_WORDS, "figs", len(FIGS), "tables", len(TABS))
