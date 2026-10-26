#!/usr/bin/env python3
"""Build the scoped bibliography and retrieve available public PDF copies.

Run from the repository root with:
  uv run --no-project --with pypdf --with fonttools code/literature/build_library.py --download
No core project dependencies or lockfiles are changed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "doc/literature"
PAPERS = ROOT / "literature/papers"
TEXT = ROOT / "literature/text"
LEVEL = {"metadata": 0, "abstract": 1, "full_text": 2}


def title_tokens(value):
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return set(re.findall(r"[a-z0-9]+", value)) - {"a", "an", "the", "of", "in", "and", "for", "to"}


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def collect():
    records, aliases = [], {}
    for path in sorted((DOC / "evidence").glob("*-references.json")):
        for ref in json.loads(path.read_text()):
            if not all(ref.get(k) for k in ("id", "title", "authors", "year", "url")):
                raise ValueError(f"Incomplete reference in {path}: {ref.get('id')}")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", ref["id"]):
                raise ValueError(f"Unsafe reference ID: {ref['id']}")
            keys = ["title:" + re.sub(r"\W+", "", ref["title"]).casefold()]
            if ref.get("arxiv_id"):
                keys.append("arxiv:" + re.sub(r"v\d+$", "", ref["arxiv_id"]))
            if ref.get("doi"):
                keys.append("doi:" + ref["doi"].lower())
            existing = next((aliases[k] for k in keys if k in aliases), None)
            if existing is None:
                ref = dict(ref)
                ref["aliases"] = [ref["id"]]
                ref["evidence_files"] = [str(path.relative_to(DOC))]
                records.append(ref)
                existing = ref
            else:
                for field in ("pdf_urls", "themes", "source_urls"):
                    existing[field] = list(dict.fromkeys(existing.get(field, []) + ref.get(field, [])))
                existing["aliases"] = list(dict.fromkeys(existing["aliases"] + [ref["id"]]))
                existing["evidence_files"].append(str(path.relative_to(DOC)))
                if LEVEL.get(ref.get("evidence_level"), 0) > LEVEL.get(existing.get("evidence_level"), 0):
                    for field in ("evidence_level", "evidence_locator", "limitations"):
                        existing[field] = ref.get(field)
            for key in keys:
                aliases[key] = existing
    records.sort(key=lambda r: (r["year"], r["id"]))
    assert len({r["id"] for r in records}) == len(records), "Duplicate citation IDs"
    return records


def download(ref, previous, refresh=False):
    from pypdf import PdfReader

    path = PAPERS / (ref["id"] + ".pdf")
    base = {"id": ref["id"], "title": ref["title"], "checked_at": datetime.now(timezone.utc).isoformat(), "attempts": list(previous.get("attempts", []))}
    candidates = list(dict.fromkeys(ref.get("pdf_urls", [])))
    if ref.get("arxiv_id"):
        fallback = "https://arxiv.org/pdf/" + ref["arxiv_id"]
        if fallback not in candidates:
            candidates.append(fallback)
    if path.exists() and not refresh:
        candidates.insert(0, "local-cache")
    # Reuse the three initially downloaded, version-pinned audit papers.
    initial = list(PAPERS.glob((ref.get("arxiv_id") or "NO_ARXIV") + "v*.pdf"))
    if not path.exists() and initial:
        initial[0].rename(path)
        candidates.insert(0, "local-cache")
    for url in candidates[:4]:
        try:
            if url == "local-cache":
                data = path.read_bytes()
                final_url = previous.get("source_url") or (candidates[1] if len(candidates) > 1 else None)
            else:
                req = Request(url, headers={"User-Agent": "representation-collapse-literature-audit/1.0"})
                with urlopen(req, timeout=30) as response:
                    data = response.read(50 * 1024 * 1024 + 1)
                    final_url = response.url
                if len(data) > 50 * 1024 * 1024:
                    raise ValueError("PDF exceeds 50 MiB retrieval limit")
            if not data.lstrip().startswith(b"%PDF-"):
                raise ValueError("Response is not a PDF")
            reader = PdfReader(io.BytesIO(data))
            if not len(reader.pages):
                raise ValueError("PDF has no pages")
            if url != "local-cache":
                tmp = path.with_suffix(".pdf.part")
                tmp.write_bytes(data)
                tmp.replace(path)
            text_path = TEXT / (ref["id"] + ".txt")
            extraction_error = None
            try:
                full_text = "\n".join(f"\n--- PDF PAGE {i+1} ---\n" + (page.extract_text() or "") for i, page in enumerate(reader.pages))
                text_path.write_text(full_text)
                chars = len(full_text)
                expected = title_tokens(ref["title"])
                opening = title_tokens(" ".join((p.extract_text() or "") for p in reader.pages[:2]))
                title_match = len(expected & opening) / max(1, len(expected))
            except Exception as exc:
                chars, extraction_error, title_match = 0, str(exc), None
            base.update(status="downloaded", path=str(path.relative_to(ROOT)), source_url=final_url,
                        bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), pages=len(reader.pages),
                        extracted_characters=chars, extraction_error=extraction_error,
                        title_token_coverage_first_two_pages=title_match,
                        identity_review_required=title_match is None or title_match < 0.8,
                        pdf_metadata_title=str((reader.metadata or {}).get("/Title", "")))
            return base
        except Exception as exc:
            base["attempts"].append({"url": url, "error": str(exc)[:400]})
    base["status"] = "unavailable" if candidates else "no_public_pdf_identified"
    return base


def bib_escape(value):
    return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def build_outputs(refs, manifest):
    by_id = {r["id"]: r for r in manifest}
    atomic_json(DOC / "references.json", refs)
    bib = []
    lines = ["# Annotated bibliography", "", "Search cutoff: 2026-10-11. This is a scoped research bibliography, not a proof of exhaustive coverage.",
             "", "Evidence levels describe what was inspected for this audit; downloading a PDF does not upgrade its evidence level.",
             "Publication years and venues follow the verified source; preprint and publication dates can differ. Null DOI fields are intentional.", "",
             "See [novelty review](novelty-review.md), [search method](search-log.md), [BibTeX](references.bib), and [download manifest](download-manifest.json).", ""]
    for r in refs:
        venue = r.get("venue") or "Publication venue unverified"
        kind = r.get("bib_type", "misc")
        fields = {"title": "{" + bib_escape(r["title"]) + "}", "author": " and ".join(bib_escape(a) for a in r["authors"]), "year": r["year"], "url": r["url"]}
        if kind == "article": fields["journal"] = bib_escape(venue)
        elif kind == "inproceedings": fields["booktitle"] = bib_escape(venue)
        else: fields["note"] = bib_escape(venue)
        if r.get("doi"): fields["doi"] = r["doi"]
        for key in ("volume", "number", "pages", "publisher"):
            if r.get(key): fields[key] = bib_escape(r[key])
        if r.get("arxiv_id"): fields.update(eprint=r["arxiv_id"], archivePrefix="arXiv")
        bib.append("@" + kind + "{" + r["id"] + ",\n" + "\n".join(f"  {k} = {{{v}}}," for k,v in fields.items()) + "\n}\n")
        m = by_id.get(r["id"], {})
        lines += [f"<a id=\"{r['id']}\"></a>", f"## {r['title']}", "",
                  f"**{r['id']}** — {'; '.join(r['authors'])}. {r['year']}. {venue}.", "",
                  f"[Primary source]({r['url']})" + (f" · [DOI](https://doi.org/{r['doi']})" if r.get('doi') else ''), "",
                  f"**Themes:** {', '.join(r.get('themes', []))}. **Evidence:** {r.get('evidence_level', 'metadata')}; {r.get('evidence_locator') or 'no full-text locator recorded'}.", "",
                  f"**Relevance:** {r.get('relevance', '')}", "",
                  f"**Novelty boundary:** {r.get('novelty_threat', '')}", "",
                  f"**Limits:** {r.get('limitations', '')}", ""]
        if r.get("reviewed_arxiv_version"):
            lines += [f"**Reviewed version:** `{r['reviewed_arxiv_version']}` (the first PDF candidate is pinned to this version).", ""]
        if r.get("version_note"):
            lines += [f"**Copy/version note:** {r['version_note']}", ""]
        if m.get("status") == "downloaded":
            lines += [f"**Local PDF:** [{r['id']}.pdf](../../{m['path']}) — {m['pages']} pages; SHA-256 `{m['sha256']}`.", ""]
        else:
            lines += [f"**Local PDF:** {m.get('status', 'not_attempted')}; see retrieval attempts in the manifest.", ""]
    (DOC / "references.bib").write_text("\n".join(bib))
    # Keep machine-readable bibliography assets at their established locations.
    # Human-readable output follows the shared dated documentation convention.
    from zoneinfo import ZoneInfo
    day = datetime.now(ZoneInfo("America/Toronto"))
    destination = ROOT / "docs" / day.strftime("%B") / day.date().isoformat() / "local" / "literature" / "bibliography.md"
    import importlib.util
    spec = importlib.util.spec_from_file_location("documentation", ROOT / "scripts/documentation.py")
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    index = ROOT / "documentation-index.json"
    mapping = {m["old"]: m["new"] for record in json.loads(index.read_text()).get("migrations", []) for m in record["moves"]} if index.exists() else {}
    layout.write(destination, layout.rewrite_links("\n".join(lines), DOC / "bibliography.md", destination, ROOT, mapping))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", nargs="+", help="Download only these canonical citation IDs; rebuild the complete bibliography")
    parser.add_argument("--refresh", action="store_true", help="Retrieve selected PDFs again rather than reuse the local cache")
    args = parser.parse_args()
    refs = collect()
    if args.only and set(args.only) - {r["id"] for r in refs}:
        parser.error("Unknown canonical IDs: " + ", ".join(sorted(set(args.only) - {r["id"] for r in refs})))
    manifest_path = DOC / "download-manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    current = {m["id"]: m for m in old}
    if args.download:
        PAPERS.mkdir(parents=True, exist_ok=True)
        TEXT.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download, r, current.get(r["id"], {}), args.refresh): r for r in refs if not args.only or r["id"] in args.only}
            for future in as_completed(futures):
                result = future.result()
                current[result["id"]] = result
                atomic_json(manifest_path, sorted(current.values(), key=lambda m: m["id"]))
                print(result["id"], result["status"], flush=True)
    manifest = [current[r["id"]] for r in refs if r["id"] in current]
    identity_path = DOC / "manual-identity-checks.json"
    identity_checks = json.loads(identity_path.read_text()) if identity_path.exists() else []
    for item in manifest:
        check = next((c for c in identity_checks if c["id"] == item["id"] and c["sha256"] == item.get("sha256")), None)
        if check:
            item["manual_identity_check"] = check["note"]
            item["identity_review_required"] = False
    atomic_json(manifest_path, manifest)
    build_outputs(refs, manifest)
    print(json.dumps({"references": len(refs), "downloaded": sum(m["status"] == "downloaded" for m in manifest)}))


if __name__ == "__main__":
    main()
