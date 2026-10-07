# Project Reference Library

## Current snapshot: 2026-10-07

The WSI library now contains **43 registered papers, 46 claims, 19 topic routes,
and 48 reviewed evidence atoms**, with **8 additional candidates** in a separate
review queue. Reviewed means the pages supporting the indexed statements were
checked, including visual PDF review. It does not mean every table, supplement,
or quantitative subgroup in every paper has been audited.

The autopsy/biopsy additions are Nakhleh 1992, McDonald 2022, Zaizen 2022,
Lutnick 2022 and Burk 2025. The `autopsy_biopsy_transfer` route distinguishes
cardiac sampling and clinicopathological studies from noncardiac AI transfer
examples. Three sources have public article PDFs with selected pages reviewed;
Nakhleh and Burk use explicitly labeled Europe PMC abstract snapshots only.
Burk's journal issue is 2025 (online 2024). Zaizen's Methods mentions 14 validation
cases whereas its abstract reports 42; no efficacy or validation-size claim is
indexed. These sources do not establish improved cardiac rejection prediction
from combined biopsy/autopsy training. The user confirmed that an SMC autopsy
dataset exists; its inventory and labels have not been audited here.
See the [registration and dataset follow-up note](../experiment_notes/2026-10-07-autopsy-biopsy-literature-registration.md).

Further autopsy/biopsy review adds Zerbe 1988, Hauck 1989, Husain 2017,
Amemiya 2024, Shen 2025 (SongCi), and Mori 2025. The same route now covers
11 papers and 12 evidence atoms. SongCi reports transfer to clinical pathology
benchmarks in its main text, but Supplementary Tables 9-10 and their tuning,
overlap and quantitative results were not audited. Amemiya provides paired
biopsy/whole-heart evidence in native-heart DCM, not allograft rejection.
Mori's experiment uses six autopsy subjects despite its biopsy-oriented title.
Zerbe, Hauck and Husain were reviewed from labeled abstract snapshots; the
other three from selected public PDF pages. Lutnick's USCAP 2020 autopsy
artifact-translation abstract remains in the candidate queue because the
official PDF was not acquired; it is excluded from default evidence search.
See the [expanded review and research implications](../experiment_notes/2026-10-07-autopsy-biopsy-expanded-review.md).

The 2026-10-07 additions connect imbalance handling, decision thresholds,
complementary metrics, model-selection bias and small-sample validation:
van den Goorbergh 2022, Carriero 2025, Chicco 2020, Lipton 2014,
Cawley 2010, Varoquaux 2018 and Riley 2020. Their indexed statements were
checked on selected abstract/summary pages, not by full-paper audit. All seven
atoms qualify transfer to cardiac WSI. Varoquaux's acquired source is the 2017
author preprint of the 2018 article; Lipton uses the acquired arXiv v2 title.
See the [research synthesis](../docs/smc_wsi_research_synthesis_20261007.md)
and [registration note](../experiment_notes/2026-10-07-literature-and-experiment-synthesis.md).

The 2026-10-04 additions are Agniel 2018 (healthcare-process signals), Che 2018
(GRU-D informative missingness), and Saito 2015 (precision-recall evaluation).
The `presence_mask_and_missingness` topic connects them to the existing stain
fusion and site-signature evidence. Che and Saito have publisher PDF copies;
Agniel uses an explicitly labeled PubMed abstract snapshot, not a full-text PDF.
Its reviewed atom is limited to abstract-level claims. The raw Europe PMC API
record and the related Figure 3b correction are recorded in provenance.

The nine additions cover DSMIL, HIPT, DTFD-MIL, acquisition/site shortcuts,
Virchow, probability calibration, MADELEINE, multistain graph fusion, and CARE
cardiac grade-versus-clinical-severity analysis. CARE's PDF and extracted text
already existed locally; this update adds its missing structured registration.

See [`indexes/category_gaps.md`](indexes/category_gaps.md) for the generated
paper comparison, facet counts, explicit evidence gaps, and review queue.
The editorial facets in [`indexes/paper_facets.json`](indexes/paper_facets.json)
record organ, analysis unit, stains, methods, validation, endpoint, review scope,
and source version. These descriptors aid retrieval; they do not establish
clinical validity. Noncardiac method evidence is explicitly qualified.

Priority candidates include AMR-H image classification, CSCL cross-stain
alignment, cardiac stromal remodeling, EHR-plus-morphology risk prediction,
and rare-positive PR evaluation. CONCH, CTransPath, and TITAN broaden the
encoder/slide-representation comparison. Candidates are not default evidence.

Curated cards, evidence, indexes, schemas, tools and the provenance manifest
are now Git-managed. Source PDFs, full extracted text, page images and browser
caches remain local and ignored. A Git-only checkout can validate metadata and
search evidence without the source copies. `--strict-sources` and the complete
retrieval test suite additionally require the locally retained PDFs/text.
Source-page verification requires the manifest's exact source version; a new
download may change pagination and hash and must be checked before reuse.

This directory is the project-wide literature repository. It stores source
provenance, reusable paper summaries, claim-sized evidence, and transparent
retrieval indexes. It is deliberately separate from the chronological records
in [`../experiment_notes/`](../experiment_notes/README.md).

The default retrieval path is progressive:

```text
topic route
  -> claim
  -> reviewed evidence atom
  -> paper card
  -> exact extracted-text or PDF location for verification
```

Full paper text is a verification fallback, not the default prompt context.

Codex can run the same workflow through the repository skill
`$smc-literature-research`. Its instructions live at
`../.agents/skills/smc-literature-research/SKILL.md`. The initial curated
coverage and known gaps are summarized in [`REVIEW_2026-09-26.md`](REVIEW_2026-09-26.md).

## Directory contract

| Path | Purpose | Current storage |
|---|---|---:|
| `paper_origin/` | Local source PDFs; Git-managed provenance manifest | mixed |
| `extracted_text/` | Page-preserving text extracted from source papers | local |
| `cards/` | One concise, structured summary per paper | Git |
| `evidence/` | Claim-sized JSONL evidence atoms with source locators | Git |
| `indexes/papers.json` | Paper metadata and card paths | Git |
| `indexes/claims.json` | Research claims mapped to evidence atoms | Git |
| `indexes/topics.json` | Query/topic routes into claims and papers | Git |
| `indexes/taxonomy.json` | Controlled facets and explicit research gaps | Git |
| `indexes/paper_facets.json` | Multi-label paper descriptors and review scope | Git |
| `indexes/review_queue.json` | Prioritized candidates awaiting source review | Git |
| `indexes/category_gaps.md` | Generated coverage and comparison report | Git |
| `schema/` | Machine-checkable JSON Schemas | Git |
| `tools/` | Validation and retrieval utilities | Git |

## Separation from experiment records

`reference/` answers reusable questions such as:

- What does the literature support?
- Which source and exact location support that statement?
- What does the evidence not establish?
- Which papers are relevant to a topic across multiple experiments?

`experiment_notes/` answers project-history questions such as:

- What did we run?
- Which cohort, split, code revision, and estimand were used?
- What happened and what did we decide next?

Experiment notes may cite stable IDs from this library. Literature records
must not depend on one experiment note to remain interpretable.

## Ingestion workflow

1. Add a source PDF to `paper_origin/` locally. PDFs are intentionally ignored
   by Git.
2. Record its bibliographic information, acquisition URL, license/access note,
   and SHA-256 hash in `paper_origin/manifest.json`.
3. Extract page-preserving text into `extracted_text/`. Never guess broken
   equations or table contents; verify exact claims against the PDF.
4. Add one paper card under `cards/` using
   `schema/paper_card.schema.json`.
5. After manual source review, add small evidence atoms to a JSONL file under
   `evidence/` using `schema/evidence_atom.schema.json`.
6. Register the paper, claims, and topic routes in `indexes/`.
   Update its `paper_facets.json` entry using the controlled values in
   `taxonomy.json`. Remove the candidate from `review_queue.json` only after
   completing source review and registration.
7. Run `python reference/tools/validate_reference.py --strict-sources` when
   the local source corpus is available. This also verifies PDF hashes,
   own-claim references, review-state consistency, and facet/queue integrity.
8. Regenerate the report with `python reference/tools/coverage.py`, then run
   `python -m unittest discover -s reference/tests -v` after retrieval changes.

## Evidence policy

- A paper card is a navigation aid, not evidence by itself.
- Default retrieval returns only reviewed atoms with `direct` or `qualified`
  confidence.
- Every evidence atom states both `supports` and `does_not_establish`.
- Numerical, clinical, causal, or conflicting claims must be checked at the
  exact source locator.
- `unresolved` and `draft` records are discovery candidates and must not be
  cited as established evidence.
- A missing indexed result means “not reviewed yet,” not “absent from the
  literature.”
- Literature evidence and this project's experimental results remain distinct.

## Initial topic routes

`indexes/topics.json` starts with routes for cardiac allograft pathology,
computational pathology MIL, foundation encoders, multiscale representation,
stain/domain shift, weak or longitudinal labels, patient-grouped evaluation,
external validation, calibration/ensembles, and interpretability. These routes
can evolve without changing the directory boundary.

## Commands

Validate structure and cross-references:

```bash
python reference/tools/validate_reference.py
python reference/tools/validate_reference.py --strict-sources
```

Search reviewed evidence with transparent lexical/topic routing:

```bash
python reference/tools/search_reference.py "patient grouped MIL external validation"
python reference/tools/search_reference.py "심장이식 거부반응 외부 검증"
python reference/tools/search_reference.py "MADELEINE 염색 결합" --focus stain_fusion --stain IHC
python reference/tools/search_reference.py "Exp3 MRXS shortcut" --focus shortcut --json
python reference/tools/search_reference.py "다중 배율 scale fusion" --focus multiscale
python reference/tools/search_reference.py "불균형 양성 부족 평가지표 임계값" --top-k 7
python reference/tools/review_queue.py --priority 1
python reference/tools/coverage.py
```

From the outer `Project root`, use:

```powershell
python tools/search_wsi_reference.py "Exp3 MRXS 취득 편향" --focus shortcut
```

Filters (`--organ`, `--unit`, `--stain`, `--method`, `--validation`, `--focus`)
are exact facet matches and combine with AND. The default applies no organ
filter so relevant noncardiac method analogues remain discoverable.
`--organ heart` returns only papers tagged heart; use unfiltered search for
general methodology and qualified cross-organ comparisons. `--json` emits a
clean bundle with atom boundaries, card paths, source-text paths, and linked
local experiment artifacts. Topic-diverse results help comparison questions.

Fetch an open-access PMC PDF and create page-preserving text plus visual QA
samples:

```bash
python reference/tools/fetch_pmc_pdf.py paper_id=PMC1234567
python reference/tools/extract_pdf_text.py reference/paper_origin/paper_id.pdf \
  --render sample --render-dir tmp/pdfs
```

This lexical layer is intentionally simple and auditable. Embedding retrieval
can later be added as a candidate generator while retaining the same reviewed
evidence and source-verification boundary.
