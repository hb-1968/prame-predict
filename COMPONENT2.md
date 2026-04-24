# Component 2: PRAME-Conditioned Melanoma Classifier

Component 2 extends the PRAME expression predictor (Component 1,
see [README.md](README.md)) into a melanoma-vs-not-melanoma
diagnostic classifier that takes both H&E morphology features and
PRAME expression (measured or predicted) as inputs. This document
is the working record for Component 2 development — data
acquisition plan, pipeline setup around scripts `06` and `07`,
open risks, and the decision history that got us here. It will
continue to update as acquisition completes and modeling lands.

## Status

| Phase | Status |
|---|---|
| Design / acquisition plan | Finalized — **Plan E** (2026-04-23) |
| `06_prepare_diagnostic_labels.py` | Implemented, smoke-tested (203-row baseline output) |
| `07_download_component2.py` | Planned, not yet written |
| External-source access | HEST-1k awaiting HuggingFace approval |
| Tiling / extraction | `02_tile_wsi.py` + `03_extract_features.py` ready (per-slide, manifest-agnostic) |
| Training / modeling | Not started |
| Results writeup (figures, tables) | Deferred — applies once Component 2 produces training artifacts |

The project convention for results writeups is to embed training
and ROC PNGs plus a regularization subsection (not just tables).
That convention does not apply yet — no training outputs exist
for Component 2 — but will govern the eventual results section of
this document.

## Motivation

Component 1 hit a validation-AUC ceiling of **0.741 ± 0.035**
(UNI features, patient-level stratified 5-fold CV on 200 slides /
194 patients). See [README.md § Results](README.md#results) for
the full numbers. Held-out attention heatmaps surfaced two
symmetric failure modes that account for most of the gap to a
perfect classifier:

- **False positives** — low-PRAME slides whose morphology mimics
  a classical high-PRAME tumor (dense focal tumor nests inside
  quiet slides). Morphology alone cannot separate these from
  true high-PRAME cases.
- **False negatives** — high-PRAME slides whose tumor-nest
  signal is drowned out by necrotic cavities, tissue tears, or
  other artifacts. The model is confident *because* no focal
  tumor-nest pattern is detected; the PRAME-expressing regions
  exist but are architecturally masked.

Component 2 targets these errors by conditioning the diagnostic
call on the **predicted PRAME distribution** rather than a hard
threshold. When PRAME is reliably extreme (top / bottom
quartile), the classifier weighs it alongside morphology; when
PRAME is indeterminate, Component 3's routing logic falls back
to standard CONCH/UNI classification without PRAME conditioning.

## Training Cohort Design

### Positives (200 slides, existing)

Same 200 TCGA-SKCM slides used throughout Component 1.
Biopsy/resection tissue. Measured PRAME from RNA-Seq
(`ENSG00000185686`). Already present in
`data/expression/diagnostic_manifest.csv` as the
`skcm_melanoma` rows.

### Negatives (403 slides from four external sources — Plan E)

| Source | Count | PRAME label | Tissue | Format | License |
|---|---|---|---|---|---|
| GTEx normal skin | 200 | Measured (RNA-Seq) | **Post-mortem** | `.svs` @ 20× | Open / citation |
| COBRA BCC | ~115 | **NaN** (missing) | Biopsy / excision | WSI (verify on ingest) | CC-BY-SA-NC 4.0 |
| HEST-1k + individual Visium | ~85 | Pseudobulk from ST spots | Biopsy | Pyramidal TIFF | CC-BY-NC-SA 4.0 |
| SKCM tumor-free (bonus) | 3 | Measured (RNA-Seq) | Biopsy | `.svs` | TCGA |

**Totals**: 200 positives + 403 negatives = **603 rows**.
Measured or pseudobulk PRAME on ~488 / 603 rows; ~115 COBRA rows
are NaN-PRAME and require special handling at training time.

COBRA and HEST+Visium counts float against each other. The
100-each target slides as HEST + enumerated Visium studies
yield; current best-estimate yield is ~85, with the ~15
shortfall padded into COBRA (~115 instead of 100). This keeps
the 400-negative headline target and the biopsy fraction intact,
at the cost of widening the NaN-PRAME bucket.

### Why this composition

The initial premise was that a single clean paired
(H&E + RNA-Seq) non-melanoma skin cohort would work. In
practice, no public cohort combines **biopsy tissue + whole-slide
H&E + paired RNA-Seq** for non-melanoma skin at 100+ scale.
Verified candidates rejected during the research pass:

- GSE125285 (25 BCC + 10 SCC + matched normals, bulk RNA) —
  GEO deposits only the RNA matrix; H&E used internally for QC
  but never released.
- BCC Nat Commun 2022 multi-omic — 5 samples total.
- Ji 2020 cSCC (Cell) — 10 tumor-normal pairs, ST format not
  bulk.
- The broad bulk-RNA skin landscape (psoriasis controls, 96-
  sample healthy atlas, etc.) — deposits RNA but not WSIs.

Plan E therefore hedges across the three achievable axes:

- **GTEx** brings a large paired-RNA **normal-skin** cohort (with
  the post-mortem caveat — see Open Risks).
- **COBRA** brings **biopsy-derived BCC subtype diversity**
  (without RNA labels — accepted NaN on these rows).
- **HEST-1k + individual Visium studies** bring biopsy-derived
  **non-melanoma skin disease with RNA** (per-spot ST, collapsed
  to slide-level pseudobulk PRAME).

Full source catalog and rejected-candidate rationale live in
`docs/dataset-research.md`. That file is **gitignored and
local-only** by explicit user decision earlier in development;
do not expect it on CI or other machines.

### Scope expansion (2026-04-23)

The original hard filter included BCC, SCC, dysplastic nevus,
benign nevus, and normal skin. Plan E adds **inflammatory skin
disease** (dermatitis, psoriasis, lichen planus, etc.) because
HEST-1k's non-melanoma skin content — verified against the
HEST-1k paper (arXiv 2406.16192v1) — is dominated by the
"Spatial transcriptomics landscape of non-communicable
inflammatory skin diseases" cohort (~54 samples, Visium), not by
cancer. This broadens the negative pool from "non-melanoma
cancer" to "non-melanoma skin lesions" and is compatible with
the three-tier clinical framing — inflammatory dermatoses are
diagnostically distinct from melanoma and teach the classifier a
more general non-melanoma morphology signal.

## 06 + 07 Pipeline

The Component-2 acquisition pipeline sits immediately after
Component 1's pipeline. Scripts `06` and `07` produce the full
diagnostic manifest; existing `02_tile_wsi.py` and
`03_extract_features.py` then run unchanged on the new slides.

### `06_prepare_diagnostic_labels.py` (done)

Queries GDC for non-melanoma TCGA cohorts and merges them with
the existing SKCM manifest (`data/expression/slide_manifest.csv`).
Output: `data/expression/diagnostic_manifest.csv`.

**Invocation for Plan E**:

```bash
python 06_prepare_diagnostic_labels.py --skip-other-cancer --skip-pan-normal
# Output: data/expression/diagnostic_manifest.csv (203 rows)
```

**Skip rationale**: both skipped sources fail the
skin-tissue-only hard filter.

- `--skip-pan-normal` drops Solid-Tissue-Normal slides from
  non-SKCM TCGA projects (kidney, pancreas, etc. — not skin).
- `--skip-other-cancer` drops Primary-Tumor slides from
  non-melanoma TCGA projects (BRCA, LUAD, COAD, PRAD, STAD —
  not skin).

The 203-row output is the GDC-sourced baseline: 200 SKCM
melanoma positives (re-queried to pick up `project_id` /
`sample_type` / `experimental_strategy` fields that
`slide_manifest.csv` doesn't carry) + 3 SKCM tumor-free
negatives.

**Features**:

- GDC POST requests with JSON body (avoids 414 URI-too-long on
  large `file_id IN (...)` clauses, which is mandatory when
  re-querying all 200 positives).
- Diagnostic-slide → any-slide-image fallback for rare normal
  cohorts that don't carry the Diagnostic-Slide flag in GDC
  metadata. Applied to `skcm_normal` and `pan_normal` queries;
  not applied to `other_cancer` (tumor slides reliably carry
  the flag, so dropping Tissue-Slides there is a feature).
- Disk-cached JSON responses at
  `data/expression/.06_gdc_cache/<sha256>.json` (default 30-day
  TTL via `--cache-days`). Re-runs after filter tweaks are
  near-instant.
- `--dry-run` prints counts, writes nothing.
- `--neg-cap N` stratified sampling (unused under Plan E —
  no caps needed with the narrow skin-only filter).

**Output schema** (13 columns):

```
file_id, file_name, file_size_gb, case_id, submitter_id,
project_id, sample_type, experimental_strategy,
melanoma_label, source_group,
prame_tpm, prame_group, prame_label
```

Positive rows carry the PRAME columns from the existing
Component-1 manifest. The 3 SKCM-normal rows have NaN PRAME
columns (no PRAME measured on tumor-free TCGA tissue). Every
row has `melanoma_label ∈ {0, 1}`.

### `07_download_component2.py` (planned)

Extends 06's manifest with the 400 Plan E negatives from the
three external sources. Writes to a new
`data/expression/diagnostic_manifest_full.csv` rather than
overwriting 06's output — keeps 06 and 07 composable and
prevents a re-run of 06 from silently discarding 07's rows.

**Per-source flow**:

1. **GTEx ingest (200 rows)**
   - Download `GTEx_Analysis_v10_RNASeQCv...gene_tpm.gct.gz`
     from the GTEx Portal Downloads page (open access, no DUA).
   - Extract the PRAME row (`ENSG00000185686`) across all skin
     samples.
   - Join per-sample `SAMPID` to its histology SVS URL via the
     GTEx Histology Viewer (NCI Biospecimen Research Database).
   - Stratified sampling: 100 from "Skin - Sun Exposed (Lower
     leg)" + 100 from "Skin - Not Sun Exposed (Suprapubic)",
     picked for widest PRAME TPM distribution (covers both
     clinical extremes).
   - Schema: `melanoma_label=0`, `source_group="gtex_normal"`,
     `prame_tpm=<measured TPM>`, `prame_source="gtex"`,
     `has_prame=True`.

2. **COBRA ingest (~115 rows)**
   - `aws s3 ls --no-sign-request s3://cobra-pathology/` to
     inventory the bucket.
   - Enumerate BCC subtypes (Ia nodular, Ib superficial, II
     medium, III high) and select a stratified pick.
   - **Avoid the risky-cancer bucket** — it lumps melanoma with
     SCC and Merkel; melanoma contamination into the negatives
     is unacceptable.
   - Schema: `melanoma_label=0`, `source_group="cobra_bcc"`,
     `prame_tpm=NaN`, `prame_source="cobra_missing"`,
     `has_prame=False`.

3. **HEST-1k + individual Visium (~85 rows)**
   - Download HEST-1k metadata CSV from HuggingFace
     (`MahmoodLab/hest`, gated — see Open Risks). Filter to
     `organ=skin`, excluding melanoma cohorts.
   - Enumerate published Visium skin studies **not already in
     HEST-1k**: Ji et al. 2020 (cSCC), Yost et al. 2019 (BCC),
     Sorin et al. 2023 (BCC niche), etc.
   - For each slide: aggregate in-tissue spot raw counts →
     sum-normalize to TPM → slide-level pseudobulk PRAME.
   - Schema: `melanoma_label=0`, `source_group="hest_visium"`,
     `prame_tpm=<pseudobulk TPM>`,
     `prame_source="hest_pseudobulk"`, `has_prame=True`.

**New schema additions on top of 06**:

- `has_prame` (bool) — derivable from `prame_tpm.notna()` but
  explicit makes the downstream classifier's missing-handling
  logic clearer.
- `prame_source` (str enum) — one of `tcga`, `gtex`,
  `hest_pseudobulk`, `cobra_missing`. Flags pipeline provenance
  so cross-source confounders are explicit at training time.

**Cross-source confounders** that 07 **flags but does not fix**:

- Three scanners — Aperio (TCGA), GTEx Biospecimen Core, and
  per-Visium-study scanners.
- Three RNA-Seq pipelines — STAR+RSEM (TCGA), GTEx pipeline
  (GTEx), and per-study pipelines (HEST/Visium).
- PRAME TPMs are not directly comparable across sources
  without re-processing all raw FASTQs through a single unified
  pipeline. That's a major engineering effort; whether to invest
  in it is deferred to the Component-2 modeling phase.

**Reuse**: the existing `02_tile_wsi.py` and
`03_extract_features.py` run unchanged against the new
negatives — both are per-slide and manifest-agnostic, and the
`if .h5 exists: skip` pattern makes re-runs safe.

### Disk + download budget (rough)

| Source | Slides | Approx. size |
|---|---|---|
| GTEx | 200 | ~40 GB |
| COBRA | ~115 | ~100 GB |
| HEST-1k + Visium | ~85 | ~200–500 GB (the 2 TB HEST-1k total is mostly non-skin) |

Same per-slide download/tile/extract/delete pattern as Component
1 keeps local disk bounded.

## Open Risks

1. **HEST-1k access gating.** The HuggingFace dataset
   (`MahmoodLab/hest`) returns `GatedRepoError` without approval.
   MahmoodLab already granted UNI/CONCH access to this project,
   so turnaround should be fast, but 07 is blocked on this
   approval landing before the HEST slot can be inventoried
   precisely.
2. **GTEx post-mortem morphology caveat.** 200 of the 400
   negatives come from autopsy tissue — possible autolysis /
   freezing artifacts distinct from biopsy morphology. A
   pre-training sanity check (run a handful of GTEx SVSs through
   the UNI + CONCH feature extractors, spot-check embeddings)
   is warranted before committing full training runs.
3. **COBRA NaN-PRAME handling.** ~115 rows will have
   `prame_tpm = NaN`. The Component-2 training code must either
   (a) gate on `has_prame` and zero-out the PRAME branch, (b)
   route NaN rows through Component 1's prediction at training
   time (pseudo-labeling), or (c) drop them entirely. Deferred
   to the modeling design decision.
4. **Cross-source PRAME comparability.** Three RNA pipelines
   produce three TPM distributions. Either accept as a known
   confounder in modeling or re-process all raw FASTQs through
   a single unified pipeline (large effort).
5. **HEST skin yield uncertainty.** Paper reports 59 total skin
   samples, ~54 non-melanoma (mostly inflammatory, a few SCC).
   Published Visium skin studies not in HEST add an estimated
   ~20–40 more. Realistic yield: ~75–95. If final yield falls
   below 85, the contingency is to pad shortfall with more
   COBRA — preserves the biopsy axis, widens the NaN-PRAME
   bucket.

## Acquisition Next Steps

1. Request HEST-1k access on HuggingFace
   (https://huggingface.co/datasets/MahmoodLab/hest). Same
   MahmoodLab access process as UNI and CONCH.
2. Verify exact HEST-1k skin slide count by querying the
   metadata CSV (short Python + HF-hub script — not yet run,
   gated on step 1).
3. Write `07_download_component2.py` — the Component-2
   counterpart to `01_download_data.py`, pulling from three
   external sources with per-source PRAME-label handling.
4. Run `07_download_component2.py` →
   `data/expression/diagnostic_manifest_full.csv` (~603 rows).
5. Re-use the existing `02_tile_wsi.py` +
   `03_extract_features.py` pipeline on the new negatives.
   Pipeline mode recommended — same ~300 MB temp-disk-per-slide
   budget as Component 1.
6. Write `08_train_component2.py` — Component-2 training
   classifier. Must decide on NaN-PRAME handling (Open Risk 3)
   and cross-source normalization (Open Risk 4) before the
   first production run.

## Decision History (abbreviated)

All 2026-04-23. Full research-pass rationale in
`docs/dataset-research.md`.

- **Plan A** (COBRA only, 400 BCC slides) — rejected. No
  measured PRAME on any negative; loses the clean-label signal
  that motivates PRAME conditioning.
- **Plan B** (COBRA + Heidelberg diversified) — rejected.
  heiDATA Heidelberg release is tile-only in its primary form;
  full WSIs not straightforwardly downloadable.
- **Plan C** (GTEx only, 400 normal-skin slides) — rejected.
  Collapses the diagnostic task to "melanoma vs normal skin",
  inviting a trivial "diseased-vs-pristine" shortcut that
  doesn't test the PRAME-conditioning hypothesis.
- **Plan D** (200 GTEx + 100 GSE125285) — rejected. GEO
  verification showed GSE125285 deposits only the 6 MB RNA
  expression matrix. The H&E slides were used internally for
  tumor-content QC but are not part of the public release.
- **Plan E** (200 GTEx + ~115 COBRA + ~85 HEST/Visium +
  existing 3 SKCM-normal) — **adopted**. HEST-1k skin content
  subsequently verified from the paper (~54 non-melanoma,
  predominantly inflammatory); final composition adjusted from
  the initial 100-each target to ~115 COBRA / ~85 HEST after
  yield verification. Scope expanded to include inflammatory
  skin disease to match HEST's actual content.
