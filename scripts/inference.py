"""
scripts/inference.py
────────────────────────────────────────────────────────────────
Inference using the locally fine-tuned DistilBERT-base-uncased model.

Usage (CLI):
    uv run python scripts/inference.py --text "This Agreement shall be governed by..."
    uv run python scripts/inference.py --file contract.txt
    uv run python scripts/inference.py --file contract.pdf
    uv run python scripts/inference.py --file contract.docx
    uv run python scripts/inference.py --interactive
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

MODEL_PATH = ROOT / CFG["inference"]["model_path"]
THRESHOLD  = CFG["inference"]["threshold"]

# Read MAX_LEN from config — must match what was used during training.
# Kaggle training used 512; local training used 256.
MAX_LEN = CFG["dataset"]["max_seq_length"]

# ─── Risk & description metadata ─────────────────────────────────────────────

RISK_CATEGORIES = {
    "Uncapped Liability", "Liquidated Damages", "Non-Compete",
    "Anti-Assignment", "Change Of Control", "No-Solicit Of Employees",
    "No-Solicit Of Customers", "Termination For Convenience",
}

CLAUSE_DESCRIPTIONS = {
    "Governing Law":                      "Specifies which jurisdiction's laws govern the contract.",
    "Termination For Convenience":        "Either party can end the contract without cause.",
    "Non-Compete":                        "Restricts one party from competing with the other.",
    "Uncapped Liability":                 "No limit on financial liability — potentially very risky.",
    "Cap On Liability":                   "Limits the maximum financial liability of a party.",
    "Liquidated Damages":                 "Pre-agreed penalty amount for specific breaches.",
    "IP Ownership Assignment":            "Transfers intellectual property rights between parties.",
    "License Grant":                      "Grants rights to use intellectual property.",
    "Arbitration":                        "Disputes resolved by arbitration, not court.",
    "Warranty Duration":                  "How long product/service warranties are valid.",
    "Change Of Control":                  "Provisions triggered if a company is acquired.",
    "Anti-Assignment":                    "Restricts transferring the contract to a third party.",
    "Audit Rights":                       "Grants right to inspect books/records.",
    "Insurance":                          "Requires one or both parties to maintain insurance.",
    "Renewal Term":                       "How the contract automatically renews.",
    "Notice Period To Terminate Renewal": "Lead time required to stop auto-renewal.",
    "Minimum Commitment":                 "Minimum purchase/revenue obligations.",
    "Exclusivity":                        "Restricts parties from working with competitors.",
    "No-Solicit Of Employees":            "Prevents poaching the other party's staff.",
    "No-Solicit Of Customers":            "Prevents approaching the other party's customers.",
}


# ─── Document text extraction ─────────────────────────────────────────────────

def extract_text_from_file(file_path: str | Path) -> str:
    """Extract plain text from .pdf, .docx, or .txt files."""
    path   = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _extract_docx(path)
    elif suffix in (".txt", ".text", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}\n"
            "Supported formats: .pdf, .docx, .txt, .md"
        )


def _extract_pdf(path: Path) -> str:
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
            text  = "\n\n".join(p.strip() for p in pages)
    except ImportError:
        pass
    except Exception:
        text = ""

    if not text.strip():
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages  = [p.extract_text() for p in reader.pages if p.extract_text()]
            text   = "\n\n".join(p.strip() for p in pages)
        except ImportError:
            raise ImportError("Run: uv pip install pypdf pdfplumber")

    if not text.strip():
        raise ValueError(
            "No text extracted from the PDF.\n"
            "The file may be a scanned image PDF (no embedded text).\n"
            "Try copying the text manually and using the text input instead."
        )
    return text


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Run: uv pip install python-docx")
    doc = Document(str(path))
    return "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())


# ─── Analyzer ────────────────────────────────────────────────────────────────

class LegalContractAnalyzer:
    """Runs inference using the fine-tuned DistilBERT model. Lazy-loads on first use."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self._model     = None
        self._tokenizer = None
        self._id2label  = None
        self._device    = None

    def _load(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}.\n"
                "Download best_model.zip from the GitHub Releases page or run\n"
                "notebooks/02_finetune.ipynb, then make sure the model files are in\n"
                "models/finetuned/best_model/ (see models/README.md)."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self._model     = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path), torch_dtype=torch.float32
        )
        self._model.eval()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model  = self._model.to(self._device)

        # Load label map — saved alongside weights by the training notebook
        label_file = self.model_path / "id2label.json"
        if label_file.exists():
            with open(label_file) as f:
                self._id2label = {int(k): v for k, v in json.load(f).items()}
        else:
            self._id2label = {int(k): v for k, v in self._model.config.id2label.items()}

        print(f"Model loaded: {self.model_path.name} "
              f"({sum(p.numel() for p in self._model.parameters())/1e6:.0f}M params) "
              f"on {self._device} | {len(self._id2label)} classes | MAX_LEN={MAX_LEN}")

    def _predict(self, text: str, top_k: int = 5) -> list:
        import torch
        self._load()
        inputs = self._tokenizer(
            text, truncation=True, max_length=MAX_LEN, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._model(**inputs).logits, dim=-1)[0].cpu().tolist()
        top_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]
        return [{"label": self._id2label[i], "score": probs[i]} for i in top_idx]

    def analyze_clause(self, text: str, top_k: int = 5) -> dict:
        predictions = self._predict(text, top_k=top_k)
        best        = predictions[0]
        category    = best["label"]
        is_risky    = category in RISK_CATEGORIES
        return {
            "predicted_category": category,
            "confidence":         best["score"],
            "top_predictions":    predictions,
            "is_risky":           is_risky,
            "risk_reason":        f"'{category}' clauses require careful review." if is_risky else None,
            "description":        CLAUSE_DESCRIPTIONS.get(category),
        }

    def analyze_contract(self, contract_text: str) -> dict:
        self._load()
        paragraphs = [p.strip() for p in contract_text.split("\n\n") if len(p.strip()) > 30]
        if not paragraphs:
            paragraphs = [contract_text[i:i+400] for i in range(0, len(contract_text), 400)]

        detected, risk_flags, chunk_results = {}, [], []
        for para in paragraphs:
            result = self.analyze_clause(para)
            result["text_snippet"] = para[:200] + ("..." if len(para) > 200 else "")
            chunk_results.append(result)
            cat, conf = result["predicted_category"], result["confidence"]
            if conf >= THRESHOLD and not cat.startswith("No "):
                if cat not in detected or conf > detected[cat]["confidence"]:
                    detected[cat] = result
                if result["is_risky"]:
                    risk_flags.append(result)

        return {
            "total_chunks":        len(chunk_results),
            "detected_categories": detected,
            "risk_flags":          risk_flags,
            "risk_score":          min(100, len(risk_flags) * 15),
            "chunk_results":       chunk_results,
        }

    def analyze_file(self, file_path: str | Path) -> tuple[dict, str]:
        """Extract text from a file and analyze it. Returns (result, extracted_text)."""
        text = extract_text_from_file(file_path)
        return self.analyze_contract(text), text


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Legal Contract Analyzer (DistilBERT)")
    parser.add_argument("--text",        type=str, help="Single clause text")
    parser.add_argument("--file",        type=str, help="Path to .pdf, .docx, or .txt file")
    parser.add_argument("--model",       type=str, default=str(MODEL_PATH))
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    from rich.console import Console
    from rich.panel   import Panel
    console  = Console()
    analyzer = LegalContractAnalyzer(model_path=args.model)

    def show(r: dict):
        color = "red" if r["is_risky"] else "green"
        console.print(Panel(
            f"[bold]Category:[/bold]   {r['predicted_category']}\n"
            f"[bold]Confidence:[/bold] {r['confidence']:.1%}\n"
            f"[bold]Risk:[/bold]       [{'red' if r['is_risky'] else 'green'}]"
            f"{'⚠️  RISKY' if r['is_risky'] else '✅ OK'}[/]\n"
            + (f"\n{r['risk_reason']}"   if r["risk_reason"]  else "")
            + (f"\n\n{r['description']}" if r["description"]  else ""),
            border_style=color,
        ))

    if args.interactive:
        console.print("[bold cyan]⚖️  Legal Clause Analyzer[/bold cyan]  (Ctrl+C to quit)\n")
        try:
            while True:
                text = input("> Clause: ").strip()
                if text:
                    show(analyzer.analyze_clause(text))
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
    elif args.file:
        path = Path(args.file)
        console.print(f"[cyan]Reading {path.name} ...[/cyan]")
        r, _ = analyzer.analyze_file(path)
        console.print(f"Risk score: {r['risk_score']}/100  |  "
                      f"Detected: {len(r['detected_categories'])} clause types\n")
        for cat, cr in sorted(r["detected_categories"].items(),
                               key=lambda x: -x[1]["confidence"]):
            console.print(f"  {'⚠️' if cr['is_risky'] else '✅'} "
                          f"[bold]{cat}[/bold] ({cr['confidence']:.1%})")
    elif args.text:
        show(analyzer.analyze_clause(args.text))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
