#!/usr/bin/env python3
"""
build_graph.py — Neo4j fraud-lead graph ETL.

Builds shared-identifier links between ~1,752 entity-resolved fraud-lead
companies: same authorized official, address, phone, fax, owner, or excluded
person. "Independent" companies actually run by one operator then surface as
connected subgraphs. EVERYTHING here is an investigative LEAD requiring
records-level verification — never a conclusion of fraud.

Stages (each later stage activates only when its file arg is passed):
  1  spine            from model_leads_CLEANED.csv (companies + NPI edges)
  2  NPPES            stream the ~10GB bulk file; filter to my NPIs, then a
                      perimeter pass keeping NPIs that SHARE an address/phone/
                      fax/authorized-official key (one-hop shells). --no-widen
                      disables the perimeter pass.
  3  ownership        CMS owner files -> OWNS edges, bridged owner->NPI via PECOS
  4  PECOS            associate IDs confirm/merge person nodes -> ASSOCIATED_WITH
  5  LEIE             excluded NPIs + excluded persons (matched by name_base)

Outputs: ALL CSVs to --out (default ~/Desktop/data/graphing). Read-only on every
input. pandas + stdlib only; NPPES is streamed in chunks (memory bounded).

Run:
  python -m src.graph_work.etl.build_graph \
    --leads "<final>/model_leads_CLEANED.csv" \
    --nppes "<pre clean>/NPPES.csv" \
    --owners "<pre clean>/owners/*.csv" \
    --pecos  "<pre clean>/PECOS.csv" \
    --leie   "<pre clean>/Caught.csv" \
    --out ~/Desktop/data/graphing [--no-widen]
Stage 1 runs on --leads alone.
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd

CHUNK = 200_000
# Perimeter widening cap: a rendezvous key (building/phone/person) shared by more
# than this many NON-lead NPIs is shared INFRASTRUCTURE (billing service, office
# tower, hospital switchboard, common name), not a shell ring — it contributes no
# perimeter nodes. Keeps the one-hop ring to genuine small-overlap identifiers.
MAX_PERIMETER_PER_KEY = 15

# pandas read kwargs tolerant of the latin-1/CP1252 bytes (0xa0 etc.) these CMS
# extracts contain; reference files are read-only.
READ = dict(dtype=str, on_bad_lines="skip", encoding="utf-8", encoding_errors="replace")

# ---------- logging ----------

def log(m=""):
    print(m, flush=True)


def stage(n, title):
    log(f"\n{'='*64}\nSTAGE {n} — {title}\n{'='*64}")


# ---------- normalization (get these wrong and rendezvous nodes silently fail) ----------

_STREET_ABBR = {
    "STREET": "ST", "AVENUE": "AVE", "BOULEVARD": "BLVD", "DRIVE": "DR",
    "ROAD": "RD", "LANE": "LN", "COURT": "CT", "PLACE": "PL", "SUITE": "STE",
    "HIGHWAY": "HWY", "PARKWAY": "PKWY", "CIRCLE": "CIR", "TERRACE": "TER",
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
    "NORTHEAST": "NE", "NORTHWEST": "NW", "SOUTHEAST": "SE", "SOUTHWEST": "SW",
}
_UNIT_RE = re.compile(
    r"\b(STE|SUITE|UNIT|APT|APARTMENT|BLDG|BUILDING|FL|FLOOR|RM|ROOM|#)\b\.?\s*([A-Z0-9-]+)")
_NONWORD = re.compile(r"[^A-Z0-9 ]+")
_WS = re.compile(r"\s+")


def _clean(v):
    """Empty string for missing values — incl. the literal 'nan'/'none' that
    dtype=str + NaN produces (else they become bogus rendezvous keys)."""
    s = str(v).strip() if v is not None else ""
    return "" if s.lower() in ("", "nan", "none") else s


def norm_phone(v):
    """Digits only, strip a leading US '1', require exactly 10 digits."""
    d = re.sub(r"\D", "", _clean(v))
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    return d if len(d) == 10 else ""


def _std_tokens(s):
    return " ".join(_STREET_ABBR.get(t, t) for t in s.split())


def norm_address(line1, line2, city, state, postal):
    """Returns (address_id, suite). suite is split OUT so 'STE 104' / 'SUITE 104'
    / no-suite all merge on the same building key = street|city|state|zip5."""
    raw = f"{_clean(line1)} {_clean(line2)}".upper()
    raw = _NONWORD.sub(" ", raw)
    suites = _UNIT_RE.findall(raw)
    suite = suites[0][1] if suites else ""
    street = _UNIT_RE.sub(" ", raw)
    street = _WS.sub(" ", _std_tokens(street)).strip()
    city_n = _WS.sub(" ", _NONWORD.sub(" ", _clean(city).upper())).strip()
    state_n = _clean(state).upper()[:2]
    zip5 = re.sub(r"\D", "", _clean(postal))[:5]
    if not (street and zip5):
        return "", suite
    return f"{street}|{city_n}|{state_n}|{zip5}", suite


def norm_name_base(last, first):
    """Conservative person key root: LAST|FIRST, alnum-uppercase. Tiebreakers
    are added by the caller (PECOS/owner associate id > auth-official phone >
    middle initial) — same base + different tiebreaker => DISTINCT nodes."""
    last_n = _NONWORD.sub("", _clean(last).upper()).strip()
    first_n = _NONWORD.sub("", _clean(first).upper()).strip()
    if not (last_n and first_n):
        return ""
    return f"{last_n}|{first_n}"


def org_key(name):
    return _WS.sub(" ", _NONWORD.sub(" ", _clean(name).upper())).strip()


# ---------- Stage 1 ----------

FLAG_MAP = {
    "possible_same_operator_family": "flag_same_operator_family",
    "low_linkage_confidence": "flag_low_linkage",
    "verify_not_government": "flag_verify_government",
    "verify_not_hospital_system": "flag_verify_hospital",
    "contains_excluded_provider": "flag_contains_excluded",
}


def stage1_spine(leads_path, out):
    stage(1, "spine from model_leads_CLEANED.csv")
    df = pd.read_csv(leads_path, dtype=str).fillna("")
    df["new_rank"] = df["new_rank"].astype(int)
    df["company_id"] = "C" + df["new_rank"].astype(str).str.zfill(5)

    for raw, col in FLAG_MAP.items():
        df[col] = df["review_flags"].apply(
            lambda s: raw in [t.strip() for t in str(s).split(";")])

    num = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["n_npis_int"] = num("n_npis").astype(int)
    df["whitelist_candidate"] = (
        (df["n_npis_int"] >= 100) | df["flag_verify_government"] | df["flag_verify_hospital"])

    company_cols = ["company_id", "new_rank", "orig_rank", "company_name",
                    "company_model_score_max", "company_model_score_wmean",
                    "company_net_paid", "n_npis", "n_npis_reliable", "n_leie_npis",
                    "merge_confidence", "specialty", "is_individual_provider",
                    *FLAG_MAP.values(), "whitelist_candidate"]
    nodes_company = df[company_cols]
    nodes_company.to_csv(out / "nodes_company.csv", index=False)

    # explode npi_list -> edges; validate 10-digit
    rows, bad = [], 0
    for r in df.itertuples():
        for n in str(r.npi_list).split("|"):
            n = n.strip()
            if re.fullmatch(r"\d{10}", n):
                rows.append((r.company_id, n, r.merge_confidence))
            elif n:
                bad += 1
    edges = pd.DataFrame(rows, columns=["company_id", "npi", "merge_confidence"])
    edges.to_csv(out / "edges_company_npi.csv", index=False)
    npis = pd.DataFrame({"npi": sorted(edges["npi"].unique())})
    npis.to_csv(out / "nodes_npi.csv", index=False)

    log(f"  companies: {len(nodes_company):,} | whitelist_candidate: "
        f"{int(df['whitelist_candidate'].sum())}")
    log(f"  HAS_NPI edges: {len(edges):,} (dropped {bad} malformed) | "
        f"unique NPIs: {len(npis):,}")
    return set(npis["npi"])


# ---------- Stage 2 ----------

NPPES_USE = {
    "NPI": "npi", "Entity Type Code": "entity_type",
    "Provider Organization Name (Legal Business Name)": "org_name",
    "Provider Last Name (Legal Name)": "p_last", "Provider First Name": "p_first",
    "Provider First Line Business Mailing Address": "mail1",
    "Provider Second Line Business Mailing Address": "mail2",
    "Provider Business Mailing Address City Name": "mail_city",
    "Provider Business Mailing Address State Name": "mail_state",
    "Provider Business Mailing Address Postal Code": "mail_zip",
    "Provider Business Mailing Address Telephone Number": "mail_phone",
    "Provider Business Mailing Address Fax Number": "mail_fax",
    "Provider First Line Business Practice Location Address": "prac1",
    "Provider Second Line Business Practice Location Address": "prac2",
    "Provider Business Practice Location Address City Name": "prac_city",
    "Provider Business Practice Location Address State Name": "prac_state",
    "Provider Business Practice Location Address Postal Code": "prac_zip",
    "Provider Business Practice Location Address Telephone Number": "prac_phone",
    "Provider Business Practice Location Address Fax Number": "prac_fax",
    "Provider Enumeration Date": "enum_date",
    "Authorized Official Last Name": "ao_last",
    "Authorized Official First Name": "ao_first",
    "Authorized Official Middle Name": "ao_mid",
    "Authorized Official Title or Position": "ao_title",
    "Authorized Official Telephone Number": "ao_phone",
}


def _nppes_keys(row):
    """All rendezvous keys for one NPPES row: addresses, phones, person."""
    addrs, phones, person = [], [], None
    a_pr, s_pr = norm_address(row.prac1, row.prac2, row.prac_city, row.prac_state, row.prac_zip)
    a_ma, s_ma = norm_address(row.mail1, row.mail2, row.mail_city, row.mail_state, row.mail_zip)
    if a_pr:
        addrs.append(("LOCATED_AT", a_pr, s_pr))
    if a_ma:
        addrs.append(("MAILS_TO", a_ma, s_ma))
    for rel, raw in (("HAS_PHONE", row.prac_phone), ("HAS_PHONE", row.mail_phone),
                     ("HAS_FAX", row.prac_fax), ("HAS_FAX", row.mail_fax)):
        d = norm_phone(raw)
        if d:
            phones.append((rel, d))
    base = norm_name_base(row.ao_last, row.ao_first)
    if base:
        ao_ph = norm_phone(row.ao_phone)
        tb = ao_ph or _clean(row.ao_mid).upper()[:1]
        person = (base, tb, _clean(row.ao_title), ao_ph)
    return addrs, phones, person


def stage2_nppes(nppes_path, lead_npis, out, widen=True):
    stage(2, f"NPPES enrichment{' (+perimeter widen)' if widen else ' (no widen)'}")
    cols = list(NPPES_USE)

    # Pass A: keep my NPIs; collect their rendezvous keys for perimeter.
    # Person widen keys are ONLY name_bases carrying a phone tiebreaker (bare
    # names like SMITH|JOHN are too generic to widen on).
    keep, key_addr, key_phone, key_person = {}, set(), set(), set()
    n_seen = 0
    for ch in pd.read_csv(nppes_path, usecols=cols, chunksize=CHUNK, **READ):
        n_seen += len(ch)
        ch = ch.rename(columns=NPPES_USE)
        sub = ch[ch["npi"].isin(lead_npis)]
        for row in sub.itertuples(index=False):
            keep[row.npi] = row
            a, p, person = _nppes_keys(row)
            key_addr.update(x[1] for x in a)
            key_phone.update(x[1] for x in p)
            if person and person[3]:  # person[3] = AO phone (the tiebreaker)
                key_person.add(person[0])
        if n_seen % (CHUNK * 10) == 0:
            log(f"    pass A … {n_seen:,} rows scanned, {len(keep):,} lead NPIs found")
    log(f"  pass A done: {len(keep):,}/{len(lead_npis):,} lead NPIs located in NPPES")

    perimeter = {}
    if widen:
        # Bounded one-hop ring: count how many non-lead NPIs each shared key pulls,
        # store rows only while a key stays under the cap, then keep a perimeter NPI
        # only if it still has >=1 surviving (rare) shared key.
        key_count, cand_rows, cand_keys = {}, {}, {}
        for ch in pd.read_csv(nppes_path, usecols=cols, chunksize=CHUNK, **READ):
            ch = ch.rename(columns=NPPES_USE)
            ch = ch[~ch["npi"].isin(keep)]
            for row in ch.itertuples(index=False):
                a, p, person = _nppes_keys(row)
                matched = [x[1] for x in a if x[1] in key_addr]
                matched += [x[1] for x in p if x[1] in key_phone]
                if person and person[0] in key_person:
                    matched.append("PERSON:" + person[0])
                kept_any = False
                for k in matched:
                    key_count[k] = key_count.get(k, 0) + 1
                    if key_count[k] <= MAX_PERIMETER_PER_KEY:
                        cand_keys.setdefault(row.npi, set()).add(k)
                        kept_any = True
                if kept_any:
                    cand_rows[row.npi] = row
        infra = {k for k, c in key_count.items() if c > MAX_PERIMETER_PER_KEY}
        for npi, keys in cand_keys.items():
            if keys - infra:  # at least one rare shared key survives
                perimeter[npi] = cand_rows[npi]
        log(f"  perimeter widen (cap {MAX_PERIMETER_PER_KEY}/key): +{len(perimeter):,} "
            f"shared-identifier NPIs ({len(infra):,} infra keys dropped)")

    allrows = {**keep, **perimeter}

    # Build node/edge tables from the kept + perimeter rows.
    npi_rows, e_addr, e_phone, e_person = [], [], [], []
    addr_nodes, phone_nodes, person_nodes = {}, {}, {}
    for npi, row in allrows.items():
        npi_rows.append({
            "npi": npi, "entity_type": row.entity_type,
            "org_name": row.org_name or f"{row.p_last} {row.p_first}".strip(),
            "enum_date": row.enum_date, "in_lead_set": npi in lead_npis})
        a, p, person = _nppes_keys(row)
        for rel, aid, suite in a:
            addr_nodes[aid] = aid
            e_addr.append({"npi": npi, "address_id": aid, "rel": rel, "suite": suite})
        for rel, d in p:
            phone_nodes[d] = d
            e_phone.append({"npi": npi, "digits": d, "rel": rel})
        if person:
            base, tb, title, ao_ph = person
            pid = f"{base}|{tb}" if tb else base
            person_nodes[pid] = {"person_id": pid, "name_base": base,
                                 "tiebreaker": tb, "ao_phone": ao_ph, "source": "nppes_ao"}
            e_person.append({"npi": npi, "person_id": pid, "rel": "AUTHORIZED_BY",
                             "title": title})

    def dump(name, records, cols):
        pd.DataFrame(records, columns=cols).to_csv(out / name, index=False)

    dump("nodes_npi_enriched.csv", npi_rows, ["npi", "entity_type", "org_name", "enum_date", "in_lead_set"])
    pd.DataFrame({"address_id": list(addr_nodes)}).to_csv(out / "nodes_address.csv", index=False)
    pd.DataFrame({"digits": list(phone_nodes)}).to_csv(out / "nodes_phone.csv", index=False)
    pd.DataFrame(list(person_nodes.values())).to_csv(out / "nodes_person.csv", index=False)
    dump("edges_npi_address.csv", e_addr, ["npi", "address_id", "rel", "suite"])
    dump("edges_npi_phone.csv", e_phone, ["npi", "digits", "rel"])
    dump("edges_npi_person.csv", e_person, ["npi", "person_id", "rel", "title"])

    log(f"  nodes: {len(npi_rows):,} npi, {len(addr_nodes):,} address, "
        f"{len(phone_nodes):,} phone, {len(person_nodes):,} person")
    log(f"  edges: {len(e_addr):,} npi-address, {len(e_phone):,} npi-phone, "
        f"{len(e_person):,} npi-person")
    # person_id -> name_base map for Stage 4/5 reconciliation
    return {pid: v["name_base"] for pid, v in person_nodes.items()}


# ---------- Stage 3 + 4 (need PECOS to bridge enrollment <-> NPI) ----------

def load_pecos(pecos_path, lead_npis):
    """PECOS bridges owner-file ENROLLMENT IDs to NPIs and gives a high-confidence
    person key (PECOS_ASCT_CNTL_ID). Returns (enroll->npi map, assoc rows)."""
    enroll_to_npi, assoc = {}, []
    for ch in pd.read_csv(pecos_path, chunksize=CHUNK, **READ):
        ch = ch.fillna("")
        sub = ch[ch["NPI"].isin(lead_npis)]
        for r in sub.itertuples(index=False):
            if r.ENRLMT_ID:
                enroll_to_npi[r.ENRLMT_ID] = r.NPI
            assoc.append({"npi": r.NPI, "pac_id": r.PECOS_ASCT_CNTL_ID,
                          "enrollment_id": r.ENRLMT_ID,
                          "name_base": norm_name_base(r.LAST_NAME, r.FIRST_NAME),
                          "org_name": r.ORG_NAME})
    return enroll_to_npi, assoc


def stage3_owners(owner_globs, enroll_to_npi, out):
    stage(3, "ownership -> OWNS edges (bridged to NPI via PECOS enrollment)")
    files = []
    for g in owner_globs:
        files.extend(glob.glob(g))
    if not files:
        log("  no owner files matched; skipping")
        return
    expected = {"ENROLLMENT ID", "ASSOCIATE ID - OWNER", "TYPE - OWNER",
                "ROLE TEXT - OWNER", "FIRST NAME - OWNER", "LAST NAME - OWNER",
                "ORGANIZATION NAME - OWNER", "PERCENTAGE OWNERSHIP"}
    owns = []
    for f in files:
        df = pd.read_csv(f, **READ).fillna("")
        missing = expected - set(df.columns)
        if missing:
            log(f"  COLUMN MISMATCH in {Path(f).name}")
            log(f"    expected (missing): {sorted(missing)}")
            log(f"    found: {list(df.columns)}")
            sys.exit("Owner-file columns differ from CMS layout — stopping rather than guessing.")
        # spaced column names break itertuples attr access; use vectorized + iterrows
        df["_npi"] = df["ENROLLMENT ID"].map(enroll_to_npi)
        hit = df[df["_npi"].notna()]
        for _, r in hit.iterrows():
            is_org = bool(r["ORGANIZATION NAME - OWNER"].strip())
            owner_id = ("ORG:" + org_key(r["ORGANIZATION NAME - OWNER"]) if is_org
                        else "PERSON:" + norm_name_base(r["LAST NAME - OWNER"], r["FIRST NAME - OWNER"]))
            if owner_id in ("ORG:", "PERSON:"):
                continue
            owns.append({"owner_id": owner_id, "owner_type": "Org" if is_org else "Person",
                         "owner_assoc_id": r["ASSOCIATE ID - OWNER"],
                         "owner_name": (r["ORGANIZATION NAME - OWNER"] if is_org
                                        else f'{r["LAST NAME - OWNER"]} {r["FIRST NAME - OWNER"]}'.strip()),
                         "npi": r["_npi"], "role": r["ROLE TEXT - OWNER"],
                         "pct": r["PERCENTAGE OWNERSHIP"]})
    pd.DataFrame(owns, columns=["owner_id", "owner_type", "owner_assoc_id",
                                "owner_name", "npi", "role", "pct"]).to_csv(
        out / "edges_owns.csv", index=False)
    log(f"  OWNS edges (owner -> lead NPI): {len(owns):,} from {len(files)} owner file(s)")


def stage4_pecos(assoc, person_name_bases, out):
    stage(4, "PECOS associate IDs -> ASSOCIATED_WITH + person confirmation")
    rows = []
    for a in assoc:
        if not a["pac_id"]:
            continue
        confirms = a["name_base"] in person_name_bases.values() if a["name_base"] else False
        rows.append({"npi": a["npi"], "pac_id": a["pac_id"],
                     "enrollment_id": a["enrollment_id"], "name_base": a["name_base"],
                     "confirms_nppes_person": confirms})
    pd.DataFrame(rows, columns=["npi", "pac_id", "enrollment_id", "name_base",
                                "confirms_nppes_person"]).to_csv(
        out / "edges_associated_with.csv", index=False)
    log(f"  ASSOCIATED_WITH edges: {len(rows):,} "
        f"({sum(r['confirms_nppes_person'] for r in rows):,} confirm an NPPES person)")


# ---------- Stage 5 ----------

def stage5_leie(leie_path, lead_npis, person_name_bases, out):
    stage(5, "LEIE — excluded NPIs + excluded persons (name_base LEADS)")
    df = pd.read_csv(leie_path, **READ).fillna("")
    df["NPI"] = df["NPI"].str.strip()
    npi_hits = df[df["NPI"].isin(lead_npis) & df["NPI"].str.fullmatch(r"\d{10}")]
    npi_hits[["NPI", "EXCLTYPE", "EXCLDATE", "LASTNAME", "FIRSTNAME", "BUSNAME"]].rename(
        columns={"NPI": "npi"}).to_csv(out / "flags_leie_npi.csv", index=False)

    bases = set(person_name_bases.values())
    rows = []
    for r in df.itertuples(index=False):
        base = norm_name_base(r.LASTNAME, r.FIRSTNAME)
        if base and base in bases:
            rows.append({"name_base": base, "excltype": r.EXCLTYPE,
                         "excldate": r.EXCLDATE, "leie_last": r.LASTNAME,
                         "leie_first": r.FIRSTNAME, "leie_city": r.CITY,
                         "leie_state": r.STATE,
                         "match_note": "NAME-ONLY LEAD — verify identity (DOB/address) before use"})
    pd.DataFrame(rows, columns=["name_base", "excltype", "excldate", "leie_last",
                                "leie_first", "leie_city", "leie_state", "match_note"]).to_csv(
        out / "flags_leie_person.csv", index=False)
    log(f"  excluded lead NPIs: {len(npi_hits):,} | excluded-person name LEADS: {len(rows):,}")
    log("  (person matches are name-base only — LEADS requiring identity verification, not IDs)")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leads", required=True)
    ap.add_argument("--nppes")
    ap.add_argument("--owners", nargs="+", default=[],
                    help="one or more globs/paths to CMS owner CSVs")
    ap.add_argument("--pecos")
    ap.add_argument("--leie")
    ap.add_argument("--out", default="~/Desktop/data/graphing")
    ap.add_argument("--no-widen", dest="widen", action="store_false",
                    help="disable Stage-2 perimeter expansion")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log(f"output dir: {out}")

    lead_npis = stage1_spine(args.leads, out)

    person_name_bases = {}
    if args.nppes:
        person_name_bases = stage2_nppes(args.nppes, lead_npis, out, widen=args.widen)
    else:
        log("\n(skipping stages 2-5: no --nppes)")

    enroll_to_npi, assoc = ({}, [])
    if args.pecos:
        enroll_to_npi, assoc = load_pecos(args.pecos, lead_npis)
        log(f"\nPECOS: {len(enroll_to_npi):,} enrollment->NPI bridges, {len(assoc):,} assoc rows")

    if args.owners:
        if not enroll_to_npi:
            log("\nSTAGE 3 needs --pecos to bridge owner enrollment IDs to NPIs; skipping owners")
        else:
            stage3_owners(args.owners, enroll_to_npi, out)

    if args.pecos:
        stage4_pecos(assoc, person_name_bases, out)

    if args.leie:
        stage5_leie(args.leie, lead_npis, person_name_bases, out)

    log(f"\nDONE — all outputs in {out}")
    log("Everything produced is an investigative LEAD requiring records-level verification.")


if __name__ == "__main__":
    main()
