"""
ui/app.py — Gradio web interface for Legal Contract Analyzer

Launch:
    uv run python ui/app.py
    → http://localhost:7860
"""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

from scripts.inference import (
    CLAUSE_DESCRIPTIONS,
    RISK_CATEGORIES,
    LegalContractAnalyzer,
    extract_text_from_file,
)

analyzer = LegalContractAnalyzer()

# ─── Demo texts ───────────────────────────────────────────────────────────────

DEMO_CLAUSES = {
    "Governing Law": (
        "This Agreement shall be governed by and construed in accordance with the laws "
        "of the State of Delaware, without regard to its conflict of laws provisions."
    ),
    "Non-Compete": (
        "During the Term and for a period of two (2) years following termination, "
        "the Contractor shall not engage in any business that competes directly or "
        "indirectly with the Company in any market where the Company currently operates."
    ),
    "Uncapped Liability": (
        "Each party shall be liable to the other for all direct, indirect, incidental, "
        "special, consequential, and punitive damages with no limitation whatsoever, "
        "regardless of whether such damages were foreseeable."
    ),
    "Arbitration": (
        "Any dispute arising out of or relating to this Agreement shall be settled by "
        "binding arbitration under the rules of the AAA in New York, New York."
    ),
    "IP Assignment": (
        "The Contractor assigns to the Company all right, title, and interest in all "
        "inventions and works of authorship created in connection with the services."
    ),
}

DEMO_CONTRACT = """CONSULTING SERVICES AGREEMENT

This Agreement is entered into as of January 1, 2024, between TechCorp Inc. ("Company")
and John Smith Consulting LLC ("Contractor").

1. SERVICES
Contractor shall provide software development consulting services per Schedule A.

2. TERM AND RENEWAL
This Agreement commences on the Effective Date and continues for one (1) year.
It shall automatically renew for successive one-year terms unless either party
provides 30 days written notice of non-renewal.

3. INTELLECTUAL PROPERTY
Contractor assigns to Company all inventions, discoveries, and works of authorship
created in connection with the services.

4. NON-COMPETE
During the Term and for two (2) years after termination, Contractor shall not
engage in any business competing with Company in any market where Company operates.

5. LIABILITY
EACH PARTY SHALL BE LIABLE FOR ALL DAMAGES WITH NO LIMITATION WHATSOEVER,
INCLUDING INDIRECT, CONSEQUENTIAL, AND PUNITIVE DAMAGES.

6. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.

7. ARBITRATION
All disputes shall be resolved by binding arbitration in New York, NY under AAA rules.

8. INSURANCE
Contractor shall maintain professional liability insurance of at least $1,000,000
per occurrence throughout the Term.
"""

# ─── Handler functions ────────────────────────────────────────────────────────

def _error4(msg): return msg, "", "", ""
def _error3(msg): return msg, "", ""


def analyze_clause(text: str):
    if not text or not text.strip():
        return _error3("⚠️ Please enter some clause text.")
    try:
        r = analyzer.analyze_clause(text)
    except FileNotFoundError as e:
        return _error3(f"❌ Model not found.\n\n{e}")
    except Exception as e:
        return _error3(f"❌ Error: {e}")

    risk_icon = "⚠️ RISKY" if r["is_risky"] else "✅ Low Risk"
    desc      = r.get("description") or CLAUSE_DESCRIPTIONS.get(r["predicted_category"], "")
    summary   = (
        f"**Category:** {r['predicted_category']}\n\n"
        f"**Confidence:** {r['confidence']:.1%}\n\n"
        f"**Risk:** {risk_icon}\n\n"
        + (f"**Note:** {r['risk_reason']}\n\n" if r["risk_reason"] else "")
        + (f"**Description:** {desc}" if desc else "")
    )
    top_md = "| # | Category | Score |\n|---|----------|-------|\n"
    for i, p in enumerate(r["top_predictions"], 1):
        bar     = "█" * int(p["score"] * 10) + "░" * (10 - int(p["score"] * 10))
        top_md += f"| {i} | {p['label']} | {p['score']:.1%} `{bar}` |\n"
    return summary, top_md, json.dumps(r, indent=2)


def _format_contract_result(r: dict):
    """Shared formatter for both text and file contract analysis."""
    score   = r["risk_score"]
    emoji   = "🔴" if score >= 60 else "🟡" if score >= 30 else "🟢"

    overview = (
        f"## Summary\n\n"
        f"- **Chunks analyzed:** {r['total_chunks']}\n"
        f"- **Clause types detected:** {len(r['detected_categories'])}\n"
        f"- **Risk flags:** {len(r['risk_flags'])}\n"
        f"- **Risk score:** {emoji} {score}/100\n"
    )

    clauses_md  = "## Detected Clauses\n\n"
    clauses_md += "| Clause Detected | Prediction Confidence | Risk Level ⚠️ or ✅ |\n|--------|------------|------|\n"
    for cat, cr in sorted(r["detected_categories"].items(), key=lambda x: -x[1]["confidence"]):
        clauses_md += f"| {cat} | {cr['confidence']:.1%} | {'⚠️' if cr['is_risky'] else '✅'} |\n"

    if r["risk_flags"]:
        risk_md = "## ⚠️ Risk Flags\n\n"
        for rf in r["risk_flags"]:
            risk_md += (
                f"### {rf['predicted_category']}\n"
                f"> {rf['text_snippet']}\n\n"
                f"*{rf.get('risk_reason', '')}*\n\n---\n\n"
            )
    else:
        risk_md = "## ✅ No significant risk flags detected."

    export = {
        "risk_score": r["risk_score"],
        "detected_categories": {
            cat: {"confidence": cr["confidence"], "is_risky": cr["is_risky"]}
            for cat, cr in r["detected_categories"].items()
        },
        "risk_flags": [
            {"category": rf["predicted_category"], "snippet": rf["text_snippet"]}
            for rf in r["risk_flags"]
        ],
    }
    return overview, clauses_md, risk_md, json.dumps(export, indent=2)


def analyze_contract(text: str):
    if not text or not text.strip():
        return _error4("⚠️ Please enter contract text.")
    try:
        r = analyzer.analyze_contract(text)
    except FileNotFoundError as e:
        return _error4(f"❌ Model not found.\n\n{e}")
    except Exception as e:
        return _error4(f"❌ Error: {e}")
    return _format_contract_result(r)


def analyze_uploaded_file(file_obj):
    """
    Handle a file uploaded via Gradio's File component.
    Supports .pdf, .docx, and .txt files.
    Returns: (status_msg, extracted_text, overview, clauses, risks, json_str)
    """
    if file_obj is None:
        return "⚠️ No file uploaded.", "", "", "", "", ""

    # Gradio 6+: file_obj is a filepath string, not an object with .name
    file_path = Path(file_obj) if isinstance(file_obj, str) else Path(file_obj.name)
    suffix    = file_path.suffix.lower()

    # Validate format before trying to load the model
    if suffix not in (".pdf", ".docx", ".doc", ".txt", ".md"):
        return (
            f"❌ Unsupported file type: `{suffix}`\n\nSupported: PDF, DOCX, TXT",
            "", "", "", "", ""
        )

    # Extract text
    try:
        text = extract_text_from_file(file_path)
    except (ImportError, ValueError) as e:
        return f"❌ {e}", "", "", "", "", ""
    except Exception as e:
        return f"❌ Failed to read file: {e}", "", "", "", "", ""

    word_count = len(text.split())
    status     = (
        f"✅ **{file_path.name}** extracted successfully\n\n"
        f"- **Words:** {word_count:,}\n"
        f"- **Characters:** {len(text):,}\n"
        f"- **Format:** {suffix.upper().lstrip('.')}\n\n"
        f"*Analyzing...*"
    )

    # Analyze
    try:
        r = analyzer.analyze_contract(text)
    except FileNotFoundError as e:
        return f"❌ Model not found.\n\n{e}", text[:2000], "", "", "", ""
    except Exception as e:
        return f"❌ Analysis error: {e}", text[:2000], "", "", "", ""

    overview, clauses_md, risk_md, json_str = _format_contract_result(r)

    final_status = (
        f"✅ **{file_path.name}** — analysis complete\n\n"
        f"- **Words:** {word_count:,}  |  **Chunks:** {r['total_chunks']}\n"
        f"- **Clause types found:** {len(r['detected_categories'])}\n"
        f"- **Risk flags:** {len(r['risk_flags'])}\n"
        f"- **Risk score:** {'🔴' if r['risk_score']>=60 else '🟡' if r['risk_score']>=30 else '🟢'} "
        f"{r['risk_score']}/100"
    )

    return final_status, text[:3000] + ("\n\n*(truncated for display)*" if len(text) > 3000 else ""), \
           overview, clauses_md, risk_md, json_str


# ─── Build UI ─────────────────────────────────────────────────────────────────

import gradio as gr

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.blue,
    font=gr.themes.GoogleFont("IBM Plex Sans"),
).set(
    body_background_fill="#f8fafc",
    block_background_fill="white",
    block_border_width="1px",
)

CSS = "footer { display: none !important; }"


def build_ui():
    with gr.Blocks(title="⚖️ Legal Contract Analyzer") as demo:

        gr.Markdown(
            "# ⚖️ Legal Contract Analyzer\n"
            "**AI-powered clause detection and risk analysis** — "
            "fine-tuned DistilBert on the CUAD dataset (CC BY 4.0)\n\n"
            "> ⚠️ *For informational purposes only. Not legal advice.*"
        )

        with gr.Tabs():

            # ── Tab 1: Upload File ────────────────────────────────────────────
            with gr.TabItem("📎 Upload Contract"):
                gr.Markdown(
                    "Upload a contract file to automatically extract text and analyze it.\n\n"
                    "**Supported formats:** PDF · DOCX · TXT · MD"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        file_upload = gr.File(
                            label="Upload Contract",
                            file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                            type="filepath",
                        )
                        upload_btn = gr.Button("🔍 Analyze File", variant="primary", size="lg")
                        upload_status = gr.Markdown()

                    with gr.Column(scale=1):
                        upload_overview = gr.Markdown()

                with gr.Row():
                    upload_clauses = gr.Markdown()
                    upload_risks   = gr.Markdown()

                with gr.Accordion("📄 Extracted Text (preview)", open=False):
                    extracted_text = gr.Textbox(
                        label="Extracted Text",
                        lines=12,
                        interactive=False,
                        placeholder="Extracted contract text will appear here...",
                    )

                with gr.Accordion("JSON output", open=False):
                    upload_json = gr.Code(language="json")

                upload_btn.click(
                    fn=analyze_uploaded_file,
                    inputs=file_upload,
                    outputs=[
                        upload_status, extracted_text,
                        upload_overview, upload_clauses, upload_risks, upload_json
                    ],
                )

            # ── Tab 2: Paste Text ─────────────────────────────────────────────
            with gr.TabItem("📄 Paste Contract"):
                with gr.Row():
                    with gr.Column():
                        demo_contract_btn = gr.Button("Load demo contract", size="sm")
                        contract_in = gr.Textbox(
                            label="Contract Text",
                            placeholder="Paste full contract text here...",
                            lines=18,
                        )
                        contract_btn = gr.Button("Analyze Contract", variant="primary")
                    with gr.Column():
                        contract_overview = gr.Markdown()

                with gr.Row():
                    contract_clauses = gr.Markdown()
                    contract_risks   = gr.Markdown()

                with gr.Accordion("JSON output", open=False):
                    contract_json = gr.Code(language="json")

                demo_contract_btn.click(fn=lambda: DEMO_CONTRACT, outputs=contract_in)
                contract_btn.click(
                    fn=analyze_contract, inputs=contract_in,
                    outputs=[contract_overview, contract_clauses, contract_risks, contract_json],
                )

            # ── Tab 3: Single Clause ──────────────────────────────────────────
            with gr.TabItem("🔍 Single Clause"):
                with gr.Row():
                    with gr.Column():
                        demo_pick = gr.Dropdown(
                            choices=list(DEMO_CLAUSES.keys()),
                            label="Load demo clause",
                        )
                        clause_in = gr.Textbox(
                            label="Contract Clause",
                            placeholder="Paste a single clause here...",
                            lines=6,
                        )
                        clause_btn = gr.Button("Analyze Clause", variant="primary")
                    with gr.Column():
                        clause_summary = gr.Markdown()
                        clause_top     = gr.Markdown()

                with gr.Accordion("JSON output", open=False):
                    clause_json = gr.Code(language="json")

                demo_pick.change(
                    fn=lambda k: DEMO_CLAUSES.get(k, ""),
                    inputs=demo_pick, outputs=clause_in
                )
                clause_btn.click(
                    fn=analyze_clause, inputs=clause_in,
                    outputs=[clause_summary, clause_top, clause_json]
                )

            # ── Tab 4: About ──────────────────────────────────────────────────
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
## About

Fine-tuned **DistilBERT-base-uncased** (66M parameters) on the
**CUAD v1 dataset** (CC BY 4.0) to classify contract clauses into
up to 82 categories (41 positive + 41 negative) and flag risk areas.

**Model:** `distilbert-base-uncased` — full fine-tuning, fp32

**Supported input formats:**
- 📄 PDF files (text-based; scanned PDFs require OCR)
- 📝 DOCX Word documents
- 🔤 Plain text (.txt, .md)
- ✂️ Pasted text

**Risk flags:** Uncapped Liability · Liquidated Damages · Non-Compete ·
Anti-Assignment · Change Of Control · No-Solicit · Termination For Convenience

**Dataset:** https://www.kaggle.com/datasets/konradb/atticus-open-contract-dataset-aok-beta
                """)

    return demo


def main():
    ui_cfg = CFG["ui"]

    import socket
    port = ui_cfg["port"]
    for candidate in range(port, port + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", candidate)) != 0:
                port = candidate
                break

    if port != ui_cfg["port"]:
        print(f"Port {ui_cfg['port']} in use, using {port} instead.")

    build_ui().launch(
        server_name=ui_cfg["host"],
        server_port=port,
        share=ui_cfg["share"],
        theme=THEME,
        css=CSS,
    )


if __name__ == "__main__":
    main()
