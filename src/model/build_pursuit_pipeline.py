"""
build_pursuit_pipeline.py (model) — split the model lead list into
ALREADY-CAUGHT vs NOT-YET-CAUGHT against public enforcement lists, and re-rank
the survivors as the pursuit pipeline.

Rationale (TOP10_LEAD_RESEARCH_REPORT.md): the top of the model list skews
toward providers the government already suspended/terminated/charged — great
for validation, weak for qui tam (first-to-file / public-disclosure bars). The
pursuit-worthy leads are the pattern-matches with NO public action.

Enforcement sources (downloaded copies under Model/enforcement_lists/ — re-pull
periodically, they update monthly):
  - AHCCCS Provider Suspensions/Terminations PDF (azahcccs.gov) — NPI + name
  - Nevada Medicaid Sanctions/OIG Exclusions PDF (nevadamedicaid.nv.gov) — NPI
  - Federal OIG LEIE (local raw Caught.csv) — NPI, any exclusion type
Matching: NPI first (authoritative). Name matching only against the AHCCCS
list (token-containment, >=3 tokens, conservative) because suspended entities
sometimes re-enroll under fresh NPIs; matches carry the matched name so they
can be audited.

Output: ONE CSV — Model/output/final/pursuit_pipeline.csv — all input rows,
clear (not-yet-caught) leads first with pursuit_rank, caught rows after with
their enforcement evidence.

Run:
    python -m src.model.build_pursuit_pipeline
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from . import config
from .data import log, require

ENFORCE_DIR = config.MODEL_DATA_DIR / "enforcement_lists"
NPI_RE = re.compile(r"\b[12]\d{9}\b")
SUFFIXES = {"LLC", "INC", "CORP", "PLLC", "LTD", "CO", "PC", "LC", "LLP"}


def norm_tokens(name: str) -> tuple:
    toks = re.sub(r"[^A-Z0-9 ]+", " ", str(name or "").upper()).split()
    return tuple(t for t in toks if t not in SUFFIXES)


def pdf_text(path: Path) -> str:
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)


def parse_ahcccs(path: Path):
    """-> (npi -> 'action date', list of (token-tuple, raw name))."""
    npis, names = {}, []
    for line in pdf_text(path).splitlines():
        m = re.match(r"^(.*?)\s+([12]\d{9})\s+(Suspend|Terminate)[\s-]*([\d/]*)", line.strip())
        if m:
            name, npi, action, date = m.groups()
            npis[npi] = f"AHCCCS {action} {date}".strip()
            toks = norm_tokens(name)
            if len(toks) >= 3:
                names.append((toks, name.strip()))
    return npis, names


def name_hit(company: str, ahcccs_names) -> str:
    """Conservative containment: the shorter token tuple (>=3 tokens) must be a
    prefix-ordered subset of the longer. Returns the matched list name or ''."""
    c = norm_tokens(company)
    if len(c) < 3:
        return ""
    cs = set(c)
    for toks, raw in ahcccs_names:
        if set(toks) <= cs or (len(c) >= 3 and set(c) <= set(toks)):
            return raw
    return ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_csv", type=str, default=str(
        config.MODEL_DATA_DIR / "output" / "final" / "model_leads_CLEANED.csv"))
    p.add_argument("--out", type=str, default=str(
        config.MODEL_DATA_DIR / "output" / "final" / "pursuit_pipeline.csv"))
    args = p.parse_args()

    log("[1/4] Parsing enforcement lists")
    az_npis, az_names = parse_ahcccs(ENFORCE_DIR / "ahcccs_suspensions.pdf")
    nv_npis = set(NPI_RE.findall(pdf_text(ENFORCE_DIR / "nv_exclusions.pdf")))
    leie = pd.read_csv(Path.home() / "Desktop/data/preclean/Caught.csv",
                       dtype=str, keep_default_na=False)
    leie_npis = set(leie.loc[leie["NPI"].str.match(r"^[12]\d{9}$"), "NPI"])
    require("AHCCCS list parsed", len(az_npis) > 100, f"{len(az_npis)} NPIs")
    require("NV list parsed", len(nv_npis) > 50, f"{len(nv_npis)} NPIs")
    require("LEIE loaded", len(leie_npis) > 1000, f"{len(leie_npis)} NPIs")
    log(f"    AHCCCS {len(az_npis)} NPIs / {len(az_names)} names | "
        f"NV {len(nv_npis)} NPIs | LEIE {len(leie_npis):,} NPIs")

    log("[2/4] Cross-referencing leads")
    leads = pd.read_csv(args.in_csv, dtype={"npi_list": str})
    rows = []
    for r in leads.itertuples():
        npis = str(r.npi_list).split("|")
        az_hits = sorted({f"{n}: {az_npis[n]}" for n in npis if n in az_npis})
        nv_hits = sorted({n for n in npis if n in nv_npis})
        leie_hits = sorted({n for n in npis if n in leie_npis})
        nm = name_hit(r.company_name, az_names) if not az_hits else ""
        evidence = "; ".join(
            az_hits + [f"AHCCCS name match: {nm}"] * bool(nm)
            + [f"NV exclusion: {n}" for n in nv_hits]
            + [f"LEIE: {n}" for n in leie_hits])
        rows.append({"enforcement_status": "already_caught" if evidence else "clear",
                     "enforcement_evidence": evidence})
    out = pd.concat([leads, pd.DataFrame(rows, index=leads.index)], axis=1)
    require("every lead classified", out["enforcement_status"].notna().all())

    log("[3/4] Re-ranking the clear (not-yet-caught) pipeline")
    out = out.sort_values(["enforcement_status", "new_rank"],
                          ascending=[False, True], kind="stable")  # 'clear' < 'already_caught' reversed
    out = pd.concat([out[out["enforcement_status"] == "clear"],
                     out[out["enforcement_status"] == "already_caught"]])
    clear_n = int((out["enforcement_status"] == "clear").sum())
    out["pursuit_rank"] = [i + 1 if i < clear_n else None for i in range(len(out))]
    cols = ["pursuit_rank", "enforcement_status", "enforcement_evidence"] + \
           [c for c in leads.columns]
    out = out[cols]

    log("[4/4] Writing")
    out.to_csv(args.out, index=False)
    caught = out[out["enforcement_status"] == "already_caught"]
    log(f"    {len(out):,} leads -> {clear_n:,} CLEAR (pursuit pipeline) + "
        f"{len(caught):,} already caught "
        f"(AHCCCS/NV/LEIE) → {args.out}")


if __name__ == "__main__":
    main()
