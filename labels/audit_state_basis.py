"""
Audit the fraud-basis of Model D's state-exclusion positives.

Downloads the 30 OpenSanctions us_<st>_med_exclusions datasets (nested JSON),
extracts every NPI + its sanction description/authority text, classifies the
exclusion basis, and joins against our in-universe state positives.

Categories (provider gets the *strongest* evidence across all its records):
  fraud            — conviction/exclusion text explicitly about fraud, false
                     claims, kickbacks, theft/embezzlement, misrepresentation
  license          — license/certification/board actions, surrenders
  federal_mirror   — state mirroring a federal (LEIE/Medicare) exclusion;
                     basis lives in the LEIE, checked separately by EXCLTYPE
  abuse_neglect    — patient abuse/neglect
  drug             — controlled-substance convictions/actions
  other_nonfraud   — loan default, non-payment, quality, failure to disclose...
  ambiguous        — bare "conviction", "program violation", etc.
  no_reason        — no description text at all
Output: state_npi_basis.csv (npi, state, category, evidence) + summary printed.
"""
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path.home() / "Desktop" / "Data" / "Model" / "labels" / "state_reason_audit"
OUT_DIR.mkdir(exist_ok=True)
STATES = "ca ny pa mi nj oh nv tx de sc la ky ia co md mo ne ma az wa wv nc ms in ks nh nd mt ga tn".split()

FRAUD_RE = re.compile(
    r"fraud|false claim|kickback|bribe|theft|embezzl|forger|larceny|misrepresent"
    r"|money launder|identity theft|financial exploit|billing for services not"
    r"|services not rendered|upcod|unbundl|grand theft", re.I)
LICENSE_RE = re.compile(r"licens|certif|board action|surrender|credential|registration.{0,20}(revok|suspend|action)", re.I)
FEDERAL_RE = re.compile(r"federal|medicare exclusion|oig|mandated exclusion|leie|hhs exclusion|1128|1156", re.I)
ABUSE_RE = re.compile(r"abuse|neglect|mistreat|assault|battery|exploitation of (a )?(patient|client|elder)", re.I)
DRUG_RE = re.compile(r"controlled substance|drug diversion|narcotic|prescri.{0,30}(divert|unlawful)|csa\b", re.I)
OTHER_NF_RE = re.compile(r"loan default|default on|non.?payment|failure to (disclose|repay|report)|quality of care"
                         r"|unprofessional|program requirement|documentation|record.?keeping", re.I)
AMBIG_RE = re.compile(r"conviction|felony|misdemeanor|court action|program violation|integrity", re.I)

PRIORITY = ["fraud", "abuse_neglect", "drug", "license", "federal_mirror", "other_nonfraud", "ambiguous", "no_reason"]


def classify(texts):
    cats = set()
    joined = " || ".join(texts)
    if not joined.strip():
        return "no_reason", ""
    if FRAUD_RE.search(joined): cats.add("fraud")
    if LICENSE_RE.search(joined): cats.add("license")
    if FEDERAL_RE.search(joined): cats.add("federal_mirror")
    if ABUSE_RE.search(joined): cats.add("abuse_neglect")
    if DRUG_RE.search(joined): cats.add("drug")
    if OTHER_NF_RE.search(joined): cats.add("other_nonfraud")
    if not cats and AMBIG_RE.search(joined): cats.add("ambiguous")
    if not cats: cats.add("ambiguous")
    for p in PRIORITY:
        if p in cats:
            return p, joined[:300]
    return "ambiguous", joined[:300]


def main():
    rows = []
    failed = []
    for st in STATES:
        ds = f"us_{st}_med_exclusions"
        path = OUT_DIR / f"{ds}.json"
        if not path.exists() or path.stat().st_size < 1000:
            url = f"https://data.opensanctions.org/datasets/latest/{ds}/targets.nested.json"
            r = subprocess.run(["curl", "-sL", "-o", str(path), "-w", "%{http_code}", url],
                               capture_output=True, text=True)
            if r.stdout.strip() != "200" or path.stat().st_size < 1000:
                failed.append((st, r.stdout.strip(), path.stat().st_size if path.exists() else 0))
                print(f"  [{st}] DOWNLOAD FAILED http={r.stdout.strip()}", flush=True)
                continue
        n_ent, n_npi = 0, 0
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_ent += 1
                props = e.get("properties", {})
                npis = props.get("npiCode", [])
                if not npis:
                    continue
                n_npi += 1
                texts = []
                for s in props.get("sanctions", []):
                    sp = s.get("properties", {})
                    for fld in ("description", "reason", "summary", "program", "provisions"):
                        texts.extend(sp.get(fld, []))
                cat, evidence = classify(texts)
                for npi in npis:
                    npi = re.sub(r"\D", "", str(npi))
                    if len(npi) == 10:
                        rows.append((st.upper(), npi, cat, evidence.replace("\n", " ")))
        print(f"  [{st}] {n_ent} entities, {n_npi} with NPI", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows, columns=["state", "npi", "category", "evidence"])
    df.to_csv(OUT_DIR / "state_npi_basis.csv", index=False)
    print(f"\nwrote {len(df):,} rows -> {OUT_DIR/'state_npi_basis.csv'}")
    if failed:
        print(f"FAILED downloads: {failed}")

    # join to OUR in-universe state positives
    ours = pd.read_csv(Path.home() / "Desktop/Data/Model/labels/all_state_exclusions_npis.csv", dtype={"npi": str})
    uni = pd.read_parquet(Path.home() / "Desktop/data/features/provider_features.parquet", columns=["npi"])
    ours = ours[ours["npi"].isin(set(uni["npi"]))]
    ours_npis = set(ours["npi"])
    print(f"\nour in-universe state positives: {len(ours_npis):,}")

    # strongest category per NPI across all states/records
    strength = {c: i for i, c in enumerate(PRIORITY)}
    best = (df[df["npi"].isin(ours_npis)]
            .assign(rank=lambda d: d["category"].map(strength))
            .sort_values("rank").groupby("npi").first())
    print(f"matched to OpenSanctions basis: {len(best):,} | unmatched: {len(ours_npis) - len(best):,}")
    print("\nbasis composition of Model D state positives:")
    print(best["category"].value_counts().to_string())
    best.reset_index()[["npi", "state", "category", "evidence"]].to_csv(
        OUT_DIR / "model_d_state_positive_basis.csv", index=False)
    print(f"\nper-NPI basis -> {OUT_DIR/'model_d_state_positive_basis.csv'}")


if __name__ == "__main__":
    main()
