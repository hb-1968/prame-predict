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
| Pipeline restructure (v2, 2026-04-23) | Done — 06/07/08 split by concern |
| `06_predict_cobra_prame.py` | Implemented, plumbing smoke-tested |
| `07_aggregate_hest_prame.py` | Implemented, blocked on HEST-1k HF access |
| `08_build_diagnostic_manifest.py` | Implemented, dry-run verified (203 rows from SKCM sources) |
| `09_train_component2.py` | Planned, not yet written |
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

## 06 / 07 / 08 Pipeline

Component-2 acquisition sits immediately after Component 1.
Scripts split by concern:

- **06** scores COBRA slides (which have no paired RNA) through
  Component 1's UNI fold ensemble to produce predicted PRAME.
- **07** aggregates HEST-1k Visium spot counts into per-slide
  pseudobulk PRAME.
- **08** composes everything (SKCM positives + SKCM tumor-free +
  GTEx normals + COBRA from 06 + HEST from 07) into a single
  `diagnostic_manifest.csv`.
- **09** (planned) consumes 08's manifest and trains the
  PRAME-conditioned classifier.

Existing `02_tile_wsi.py` + `03_extract_features.py` run
unchanged against new negatives — both are per-slide and
manifest-agnostic.

### `06_predict_cobra_prame.py` (done)

Ensembles Component 1's five UNI fold checkpoints
(`results/uni/fold{1..5}_model.pt`) over COBRA embeddings. Every
COBRA slide was never seen in Component-1 training, so all five
folds apply — no held-out-fold replay needed (unlike
`05_generate_heatmaps.py`). Per-slide output carries the mean,
std, and median of the 5 fold probabilities plus a `confidence`
score for Component 3's routing gate.

**Invocation**:

```bash
# Predict on every COBRA embedding after 02+03 has run
python 06_predict_cobra_prame.py --no-manifest-filter

# Filter against the manifest after 08 has produced it once
python 06_predict_cobra_prame.py \
    --manifest data/expression/diagnostic_manifest.csv \
    --source-group cobra_bcc
```

**Output**: `data/expression/cobra_prame_predictions.csv` with
`file_id, file_name, prob_fold1..prob_fold5, prob_mean,
prob_std, prob_median, pred_label, confidence, n_patches,
prame_source, component1_model`.

Consumed by 08 via an `file_id` stem join. If 06 hasn't run,
08's COBRA rows carry `prame_tpm=NaN` and
`prame_source="cobra_missing"`.

### `07_aggregate_hest_prame.py` (done, gated)

Fetches the HEST-1k metadata CSV from HuggingFace, filters to
non-melanoma skin (dermatitis, psoriasis, SCC, normal),
downloads per-slide `.h5ad` Visium AnnData, sums raw counts
across in-tissue spots, normalizes to per-million (pseudobulk
CPM — labeled `prame_tpm_pseudobulk` for terminological
consistency with TCGA/GTEx TPM), and extracts the PRAME row.

**Invocation**:

```bash
python 07_aggregate_hest_prame.py --dry-run        # HF auth check + counts
python 07_aggregate_hest_prame.py --limit 5        # 5 slides for testing
python 07_aggregate_hest_prame.py                  # full non-melanoma skin set
```

**Output**: `data/expression/hest_prame_aggregate.csv` with
`file_id, hest_cohort, disease, platform, n_in_tissue_spots,
total_raw_counts, prame_raw_count, prame_tpm_pseudobulk`.

**Blocked on HuggingFace access.** `MahmoodLab/hest` is gated;
the script fails gracefully with a `GatedRepoError` and prints
the access-request URL.

**New deps**: `scanpy`, `anndata` — plus the already-installed
`huggingface_hub`.

Individual Visium skin studies outside HEST-1k (Ji 2020 cSCC,
Yost 2019 BCC, Sorin 2023 BCC) are **out of scope** for 07 —
deferred to a future `07b_aggregate_visium_external.py` or a
later in-place extension once HEST's yield is known.

### `08_build_diagnostic_manifest.py` (done)

Composes five sources into the unified manifest at
`data/expression/diagnostic_manifest.csv`:

1. **SKCM positives (200)** — read from
   `data/expression/slide_manifest.csv` with measured PRAME.
2. **SKCM tumor-free (3)** — GDC mini-query (embedded in 08)
   for the small paired-normal cohort; Diagnostic-Slide →
   any-slide fallback since Solid-Tissue-Normal TCGA slides
   often lack the Diagnostic flag.
3. **GTEx normal skin (~200)** — direct download of
   `GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz`, lazy
   parse for only the PRAME row, join with sample annotations
   (`SMTSD`); stratified 100 sun-exposed + 100 not-sun-exposed
   picked across the PRAME TPM distribution for widest spread.
4. **COBRA BCC (~115)** — `boto3` unsigned listing of
   `s3://cobra-pathology/`, heuristic BCC-subtype classifier
   (rejects risky-cancer bucket and SCC keywords), stratified
   subtype pick; joins with 06's predictions on `file_id`
   stem to fill PRAME where available.
5. **HEST/Visium (~85)** — reads 07's aggregate CSV; rows
   carry `prame_source="hest_pseudobulk"` with the CPM value.

**Output schema** (16 columns):

```
file_id, file_name, file_size_gb, case_id, submitter_id,
project_id, sample_type, experimental_strategy,
melanoma_label, source_group,
prame_tpm, prame_group, prame_label,
has_prame, prame_source, download_url
```

`prame_source` enum distinguishes `tcga` / `tcga_unmeasured` /
`gtex` / `hest_pseudobulk` / `component1_predicted` /
`cobra_missing`. Plan E decision: `prame_tpm` stays in
source-native units; harmonization across sources is 09's job.

**Invocation**:

```bash
# Full build (requires 06 + 07 outputs for the predicted / pseudobulk slots)
python 08_build_diagnostic_manifest.py

# Partial build — SKCM-only, fast, useful for plumbing checks
python 08_build_diagnostic_manifest.py \
    --skip-gtex --skip-cobra --skip-hest --dry-run
```

Graceful skips: if 06 or 07 hasn't produced its output, 08 ships
the manifest with those rows marked `cobra_missing` /
`hest_missing` and prints a warning with the script to run.

**New dep**: `boto3` for the COBRA S3 inventory. Not required if
you pass `--skip-cobra`.

### Cross-source confounders (flagged, not fixed)

- Three scanners: Aperio (TCGA), GTEx Biospecimen Core, and
  per-Visium-study (HEST cohorts).
- Three RNA pipelines: STAR+RSEM (TCGA), GTEx's own pipeline,
  and per-study pipelines (HEST/Visium pseudobulk ≠ proper
  gene-length-corrected TPM).
- PRAME values are not cross-comparable without re-processing
  all raw FASTQs through one unified pipeline. Deferred to 09's
  modeling decision (per-source normalization layer, quantile
  harmonization, or per-source covariate).

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
2. Install the two new deps (`pip install boto3 scanpy
   anndata`). `boto3` unlocks 08's COBRA ingest; `scanpy` +
   `anndata` unlock 07's HEST pseudobulk.
3. Run `notebooks/cobra_predict_colab.ipynb` on a T4/L4 GPU
   Colab runtime — bundles the S3 download, 20x tiling (via
   `02_tile_wsi.py` in in-memory mode), UNI feature
   extraction, and the 5-fold PRAME ensemble into a single
   workflow. Produces both `embeddings/uni_cobra/*.h5` and
   `data/expression/cobra_prame_predictions.csv` on Drive.
   Reason to run on Colab instead of the laptop: GPU + S3
   bandwidth, not convenience.
4. Run `07_aggregate_hest_prame.py` once HEST access is
   granted — will download the `.h5ad` files on demand and
   cache via HF hub. Local CPU is fine (no GPU needed).
5. Run `08_build_diagnostic_manifest.py` to produce
   `data/expression/diagnostic_manifest.csv` (~600 rows).
6. Write `09_train_component2.py` — Component-2 training
   classifier. Must decide on NaN-PRAME handling (Open Risk 3)
   and cross-source PRAME normalization (Open Risk 4) before
   the first production run.

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
