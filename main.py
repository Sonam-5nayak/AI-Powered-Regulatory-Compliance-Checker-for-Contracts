import os
import json
import time
import threading
from copy import deepcopy
from datetime import datetime
import re
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# External libraries
try:
    import PyPDF2
except Exception:
    raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")

try:
    import matplotlib.pyplot as plt
except Exception:
    raise ImportError("matplotlib not installed. Run: pip install matplotlib")

# spaCy optional
USE_SPACY = False
nlp = None
try:
    import spacy
    from spacy.matcher import Matcher
    try:
        nlp = spacy.load("en_core_web_sm")
        USE_SPACY = True
    except Exception:
        USE_SPACY = False
except Exception:
    USE_SPACY = False

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
GENERAL_DIR = os.path.join(DATA_DIR, "general")
CONTRACTS_DIR = os.path.join(DATA_DIR, "contracts")
VERSIONS_DIR = os.path.join(DATA_DIR, "versions")
CHARTS_DIR = os.path.join(DATA_DIR, "charts")
MAPPING_FILE = os.path.join(GENERAL_DIR, "mappings.json")
LOCK_FILE = os.path.join(GENERAL_DIR, ".mapping_lock")

for d in [GENERAL_DIR, CONTRACTS_DIR, VERSIONS_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------- Filename Sanitizer Utility ----------------
def sanitize_filename(name: str) -> str:
    """
    Removes invalid characters from filenames.
    Keeps only letters, digits, underscore, hyphen.
    Spaces are replaced with underscores.
    """
    name = str(name)
    name = re.sub(r'[^a-zA-Z0-9_\- ]+', '', name)  # remove invalid chars
    name = name.strip().replace(" ", "_")           # convert spaces to _
    return name


# ---------------- Utilities ----------------
def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_atomic(data, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)

def save_json_locked(data, path):
    start = time.time()
    while True:
        try:
            fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.05)
            if time.time() - start > 5:
                try: os.remove(LOCK_FILE)
                except Exception: pass
                break
    try:
        save_json_atomic(data, path)
    finally:
        try: os.remove(LOCK_FILE)
        except Exception: pass

# ---------------- Mapping Data ----------------
def ensure_mapping_template():
    template = {
        "jurisdiction_mapping": {},
        "laws": {},
        "contract_risk_indicators": {},
        "metadata": {"version": "1.0", "last_updated": str(datetime.now())}
    }
    if not os.path.exists(MAPPING_FILE):
        save_json_locked(template, MAPPING_FILE)
    return load_json(MAPPING_FILE, template)

mapping_data = ensure_mapping_template()
mapping_data_lock = threading.Lock()

# ---------------- PDF Reading ----------------
def extract_text_from_pdf(path):
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text_parts = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(text_parts)
    except Exception as e:
        print(f"[error] reading PDF {path}: {e}")
        return ""

def read_contracts_from_pdfs():
    contracts = []
    files = sorted(os.listdir(CONTRACTS_DIR))
    idx = 1
    for filename in files:
        if filename.lower().endswith(".pdf"):
            path = os.path.join(CONTRACTS_DIR, filename)
            text = extract_text_from_pdf(path)
            contracts.append({
                "id": idx,
                "name": filename,
                "text": text,
                "path": path,
                "last_modified": str(datetime.fromtimestamp(os.path.getmtime(path)))
            })
            idx += 1
    return contracts

# ---------------- Risk Detection ----------------
def get_regulations_by_jurisdiction(country_code):
    global mapping_data
    with mapping_data_lock:
        mapping_data = load_json(MAPPING_FILE, mapping_data)
        regs = []
        applicable = mapping_data.get("jurisdiction_mapping", {}).get(country_code, [])
        for law_id in applicable:
            law = mapping_data.get("laws", {}).get(law_id)
            if law:
                kws = law.get("keywords") or law.get("name","").lower().replace("(","").replace(")","").split()
                regs.append({"id": law_id, "name": law.get("name", law_id), "keywords": kws, "description": law.get("description","")})
    return regs

def map_risk_level(risk_label_or_regname):
    global mapping_data
    mapping = mapping_data.get("contract_risk_indicators", {}) or {}
    label = (risk_label_or_regname or "").lower()
    for key, val in mapping.items():
        if key.lower() in label or label in key.lower():
            return val.get("risk_level", "medium")
    if "high" in label: return "high"
    if "low" in label: return "low"
    return "medium"

def detect_risks_keyword(contract_text, regulations):
    risks = []
    text_lower = (contract_text or "").lower()
    parts = [p.strip() for p in re.split(r"\n{1,}|\.\s+", contract_text) if p.strip()]
    for reg in regulations:
        for kw in reg.get("keywords", []):
            if not kw: continue
            kw_l = kw.lower()
            if kw_l in text_lower:
                snippet = next((p for p in parts if kw_l in p.lower()), contract_text[:200])
                risks.append({"sentence": snippet, "regulation": reg["name"], "labels": ["KEYWORD"], "description": reg.get("description","")})
    return risks

def detect_risks(contract, regulations):
    text = contract.get("text", "") or ""
    if USE_SPACY and nlp:
        try:
            return detect_risks_keyword(text, regulations)
        except Exception: pass
    return detect_risks_keyword(text, regulations)

# ---------------- Amendments & Versioning ----------------
def save_version(contract, tag="before"):
    """
    Save a PDF version of the contract.
    Handles both string and dict contract["text"].
    """
    safe_name = sanitize_filename(contract["name"])
    filename = f"{safe_name}_{tag}_{int(time.time())}.pdf"
    path = os.path.join(VERSIONS_DIR, filename)
    try:
        styles = getSampleStyleSheet()
        normal_style = styles["Normal"]
        green_style = ParagraphStyle(
            "GreenHighlight", parent=normal_style, backColor=colors.lightgreen
        )
        story = []

        # Ensure text is string
        text = contract["text"]
        if isinstance(text, dict):
            text = "\n\n".join(text.values())
        text = str(text)

        # Highlight amendments if present
        parts = re.split(r"(\[AMENDMENT.*?\])", text, flags=re.DOTALL)
        for part in parts:
            style = green_style if part.startswith("[AMENDMENT") else normal_style
            story.append(Paragraph(part.replace("\n", "<br/>"), style))

        pdf = SimpleDocTemplate(path, pagesize=A4)
        pdf.build(story)
    except Exception as e:
        print(f"[error] saving PDF version: {e}")

# ---------------- Amendments Suggestion ----------------
def generate_amendment_suggestion(contract, risk):
    """
    Generate a suggested amendment text based on the contract and risk detected.
    Currently a placeholder — you can replace with AI/ML logic later.
    """
    reg_name = risk.get("regulation", "Unknown regulation")
    sentence = risk.get("sentence", "")
    return f"'{sentence}' to comply with {reg_name}."


# ---------------- Amendments ----------------
def apply_amendment_append(contract, risk):
    """
    Append amendment to contract text safely.
    Returns amended contract (dict), suggestion (str).
    """
    amended = deepcopy(contract)
    suggestion = generate_amendment_suggestion(contract, risk)
    amendment_text = f"\n\n[AMENDMENT - {datetime.now().isoformat()}]\n{suggestion}"

    # Ensure text is string before appending
    if isinstance(amended["text"], dict):
        amended["text"] = "\n\n".join(amended["text"].values())
    amended["text"] += amendment_text
    amended["last_modified"] = str(datetime.now())
    save_version(contract, tag="before_append")
    save_version(amended, tag="after_append")
    return amended, suggestion

def replace_clause_with_amendment(contract, risk):
    """
    Replace a risky clause with suggested amendment.
    Returns amended contract (dict), suggestion (str), replaced (bool).
    """
    amended = deepcopy(contract)
    original_clause = risk.get("sentence", "").strip()
    suggestion = generate_amendment_suggestion(contract, risk)
    amendment_text = f"[AMENDMENT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n{suggestion}"

    # Ensure text is string
    if isinstance(amended["text"], dict):
        amended["text"] = "\n\n".join(amended["text"].values())
    
    replaced = False
    if original_clause and original_clause in amended["text"]:
        amended["text"] = amended["text"].replace(original_clause, amendment_text)
        replaced = True

    amended["last_modified"] = str(datetime.now())
    save_version(amended, tag="after_replace")
    return amended, suggestion, replaced

# ---------------- Helper for Streamlit PDF ----------------
def contract_text_to_string(contract):
    """
    Converts contract["text"] (dict or string) to a plain string for PDF or display.
    """
    text = contract.get("text", "")
    if isinstance(text, dict):
        text = "\n\n".join(text.values())
    return str(text)

# In main_backend.py
def split_contract_text(text, chunk_size=500, chunk_overlap=50):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


# ---------------- Risk Pie Chart ----------------
# ---------------- Risk Donut Chart ----------------
def generate_risk_assessment_graph(risks, contract_name):
    """
    Generates a neon-gradient donut chart showing risk assessment (high, medium, low)
    Returns the path to the saved PNG image.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    import numpy as np

    # Count risks by level
    counts = {"high": 0, "medium": 0, "low": 0}
    for r in risks:
        level = map_risk_level(", ".join(r.get("labels", []))).lower()
        if level not in counts:
            level = "medium"
        counts[level] += 1

    total = sum(counts.values())
    if total == 0:
        counts["medium"] = 1  # prevent empty chart

    # Data for donut
    labels = list(counts.keys())
    values = [counts[l] for l in labels]
    colors_map = {"high": "#ff4c4c", "medium": "#ffcc00", "low": "#4caf50"}
    colors_list = [colors_map[l] for l in labels]

    # Create donut chart
    fig, ax = plt.subplots(figsize=(5,5), facecolor="none")
    wedges, _ = ax.pie(
        values,
        labels=labels,
        colors=colors_list,
        startangle=90,
        wedgeprops=dict(width=0.3, edgecolor='black', linewidth=1.5)
    )

    # Neon gradient effect
    for w, c in zip(wedges, colors_list):
        w.set_edgecolor("white")
        w.set_linewidth(2)
        w.set_alpha(0.9)

    ax.set(aspect="equal")
    plt.title(f"Risk Assessment: {contract_name}", color="white", fontsize=14)
    fig.patch.set_facecolor("none")

    # Save figure
    safe_name = sanitize_filename(contract_name)
    filename = os.path.join(CHARTS_DIR, f"{safe_name}_risk_{int(time.time())}.png")
    plt.savefig(filename, dpi=120, transparent=True, bbox_inches='tight')
    plt.close()
    return filename