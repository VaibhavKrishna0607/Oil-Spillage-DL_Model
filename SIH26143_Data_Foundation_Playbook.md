# SAR Oil-Spill Data Foundation & Preprocessing Playbook (v2)
### SIH26143 — Oil Spill Detection + AIS Vessel Tracking
**Primary owners:** P1 (SAR Data Engineer) · P2 (ML Engineer)
**Reference material for:** P6 (architecture diagram + PPT)
**Replaces:** Section 4 ("Person 1 — SAR Data Engineer") of the original execution plan

---

## 0. Why this exists

The original plan said, essentially: *download SOS, split 80/10/10, augment, hand off to P2*. That's enough to get a model training, but the review you got flagged three gaps — geometric properties, georeferencing, and the 256×256-patch-to-full-scene generalization gap — and all three trace back to decisions made **before** the model ever sees a batch. Bolting them on later means re-touching every mask twice.

This document is the rewritten Day 1–3 data workflow. It fixes the upstream issues now, so P2 spends Day 4–7 training instead of debugging a dataset that was never checked for leakage.

---

## 1. What you're actually working with (verified, not assumed)

Your target dataset is the **Refined Deep-SAR Oil Spill (SOS) dataset** on Zenodo (DOI `10.5281/zenodo.15298010`, Zuenko & Khaidarova, HSE, published April 2025). It ships as two files: `images.zip` (1.1 GB) and `masks.zip` (28 MB) — matches the "~1.2GB" your original plan expected.

It's a cleaned-up version of the original Zhu et al. (2021, IEEE TGRS) SOS dataset. A few facts worth knowing before you write a single line of loading code:

- **8,070 labeled 256×256 patches**, cropped from just **21 raw SAR scenes**, covering two distinct sub-domains: **SOS-ALOS** (Gulf of Mexico, PALSAR L-band, ~3,101 train / 776 test) and **SOS-Sentinel** (Persian Gulf, Sentinel-1A C-band, ~3,354 train / 839 test).
- Masks are **binary** (oil / not-oil) — not the 5-class Krestenitis scheme. Good: matches your U-Net binary segmentation plan, no relabeling needed.
- The "refined" part matters: roughly **38% of training masks and 50% of validation masks had significant annotation errors manually corrected** in this release. That's the *improved* version — assume residual noise still exists and budget a QC pass rather than trusting labels blindly.
- ⚠️ Different papers that cite this dataset report the patch size as either 256×256 or 416×416 depending on which distribution they used. **Confirm the actual pixel dimensions with a 5-line script on Day 1** before you hardcode an input size anywhere in the pipeline or the model config.

---

## 2. The one fact that changes your split strategy

8,070 patches cut from only **21 source scenes** means roughly 380 patches per scene on average. If you do a naive random 80/10/10 split over patches, you will almost certainly put crops from the *same* source scene into both train and test — the model partially memorizes that scene's texture/noise pattern in training and then "validates" on a near-duplicate. Your IoU number will look better than it is, and it'll fall apart on genuinely unseen data (including your demo image).

**Fix: split by source scene, not by patch.** Two ways to get a scene ID:

1. **Check filenames first.** Zenodo crops often keep a scene/frame prefix in the filename. Inspect a sample before assuming you need anything fancier.
2. **If filenames are anonymized**, cluster by similarity instead — a cheap perceptual hash or downsampled-pixel correlation on each image will group crops from the same source scene together even without metadata. (Code for both paths is in `data_pipeline_starter.py`, Stage 4.)

Then use a **group-aware** split (`GroupShuffleSplit`, grouping on scene/cluster ID) so no scene crosses a split boundary. This single change is the highest-leverage fix in this whole document — it costs you an afternoon and it's the difference between a defensible IoU number and one a judge can poke a hole in during Q&A.

---

## 3. Two-track dataset strategy (this is also your fix for review point #7)

Don't just use SOS. Use SOS plus one more dataset that closes the "256×256 patches never generalize to a 10,000×10,000 real scene" gap the review called out — instead of discovering it on Day 10.

| Track | Dataset | Scale | Role |
|---|---|---|---|
| **A — core training** | SOS Refined (Zenodo `15298010`) | 256×256 (verify) | Bulk of training signal. High volume, fast iteration. |
| **B — scale realism** | *Sentinel-1 SAR Oil-spill dataset for train/validate/test* (3-part Zenodo release: `8346860`, `8253899`, `13761290`) | **2,048×2,048**, Sigma0 dB, VV+VH | Fine-tune / stress-test at near-deployment tile size. Part I = 1,200 oil train/val patches, Part II = 685 no-oil train/val, Part III = held-out test set (150 each of oil / no-oil / look-alike). |
| **C — optional robustness check** | DARTIS (Eastern Mediterranean, DLR) | 640×640, **JPEG** | Out-of-distribution sanity check only. Do **not** blend into core training — it's 8-bit lossy JPEG, not dB TIFF, so mixing it in unnormalized will quietly corrupt your radiometric statistics. |

**Sequencing:** Days 1–3, P1 stands up Track A only (fastest way to unblock P2). Track B gets ingested Days 4–6 in parallel, once the Track A pipeline is proven — this becomes your sliding-window/tiling test before demo prep, not a last-minute discovery. Track C only if time allows, used purely as a held-out check, never blended into the training loss.

**One thing worth 10 minutes on Day 1:** check whether Track B's TIFFs carry an embedded GeoTransform (`rasterio.open(path).transform`, or `gdalinfo` if you have GDAL). If they do, you may have partially solved the review's georeferencing problem (#4) for free, without needing to source separate full Sentinel-1 GRD scenes. Don't assume it either way — verify.

---

## 4. Data cleaning pipeline, step by step

This is the part that actually answers "how do I convert what I have into something trainable." Run these in order; each one feeds the next.

**Step 1 — Integrity & pairing check.** Every image needs exactly one mask with a matching stem name and matching spatial dimensions. Log anything that fails (corrupt file, size mismatch, orphaned image/mask) to a CSV instead of silently dropping it — you want a paper trail for the PPT's "data cleaning" slide, and it's the fastest way to catch a bad unzip.

**Step 2 — Mask QC pass.** Given that even the refined release still needed correction on roughly half the validation masks, treat annotation noise as expected, not paranoia:
- Flag masks that are entirely empty in files whose name/metadata implies an oil scene.
- Flag masks where the oil-labeled area exceeds a suspicious threshold (e.g. >60–70% of the frame) — usually a land/look-alike mislabel, not a real spill.
- Run a light morphological opening to strip isolated 1–4 pixel specks (near-certain label noise, not real slick).
- At the end, render a grid of ~30 random image/mask overlays and eyeball them yourself. Five minutes of looking at real overlays catches things no script will.

**Step 3 — Radiometric normalization.** SOS is already Sigma0 in dB, so you're not converting from raw amplitude — good, one less step. What you should still do:
- Clip outlier backscatter values before scaling. A published KSAT oil-spill dataset paper found the backscatter distribution for oil pixels sits well below non-oil pixels, and clipped their VV values at roughly the 98th percentile before normalizing, since values above that are essentially never oil. Apply the same logic here — compute your clip bound from the training split only, then min-max or z-score scale.
- **Compute normalization statistics (clip bound, mean, std) on the training split only**, then apply those fixed values to val/test. Computing stats over the full dataset before splitting is a classic, easy-to-miss leakage bug.

**Step 4 — Deduplication / scene grouping.** Covered in Section 2 — do this as part of the same pass since you're already iterating over every image.

**Step 5 — Class-imbalance quantification.** Report two numbers, not one: (a) % of *images* that contain any oil at all vs. background-only, and (b) within oil-containing images, the % of *pixels* that are oil. Oil spill segmentation is imbalanced at both levels. Your original plan already picked Dice/BCE+Dice — this step gives you the real percentages to justify that choice with a number in the PPT instead of "it's imbalanced, trust us."

**Step 6 — Format standardization.** Before packing: every image is `float32`, every mask is `uint8`/bool, same channel count across every source you use (Track A is single-channel dB; if Track B's VV+VH gets used, decide once whether P2's model takes 1 or 2 channels — don't let this get decided implicitly at load time).

---

## 5. Split strategy (concrete)

- `GroupShuffleSplit` grouped by scene/cluster ID (Section 2), target ~80/10/10.
- Within that, keep the ALOS (Gulf of Mexico) and Sentinel (Persian Gulf) sub-domains **proportionally represented** in val and test, and **report metrics per sub-domain**, not just one blended number. It's a small extra column in your results table and it's a much stronger answer under judge questioning than a single IoU figure — "0.81 IoU on C-band Sentinel data, 0.74 on L-band ALOS" reads as a team that understands its own data, not one that ran one script and stopped.

---

## 6. Augmentation — what's physically valid for SAR (and what isn't)

**Do:** horizontal/vertical flip, 90° rotations, small-angle rotation, brightness/gamma jitter within a realistic dB range, multiplicative speckle-noise injection (this one specifically helps Track A → Track B generalization, since it simulates sensor/scene variability). SAR has no "correct orientation" bias the way natural photos do, so flips are free, unbiased extra data.

**Avoid or use sparingly:** hue/saturation jitter (meaningless on single-channel dB data), aggressive cutout/erasing (can delete thin spill boundaries — your hardest class to learn — rather than teaching robustness), heavy elastic deformation (can distort real spill geometry in ways that fight against the geometric-property extraction in Section 7).

Use `albumentations` — it works on arbitrary-channel arrays, not just RGB, and lets you apply the identical transform to image + mask in one call.

---

## 7. Precompute geometric-property ground truth now (fixes review point #3 for free)

The review flagged that your pipeline never converts a predicted mask into area/perimeter/eccentricity/centroid — measurable properties judges specifically asked for in the PS. You're already looping over every mask for QC in Step 2 — do this in the same pass instead of as a separate task later:

```python
from skimage import measure

def geometric_properties(mask):
    labeled = measure.label(mask > 0)
    props = measure.regionprops(labeled)
    if not props:
        return None
    largest = max(props, key=lambda p: p.area)
    return {
        "area_px": largest.area,
        "perimeter_px": largest.perimeter,
        "eccentricity": largest.eccentricity,
        "centroid_row": largest.centroid[0],
        "centroid_col": largest.centroid[1],
        "bbox": largest.bbox,
        "n_components": len(props),  # multiple disjoint slicks in one frame
    }
```

Save this as `geometry.csv` alongside your tensors. Three payoffs for zero extra dataset passes: P2 gets a ground truth to validate the post-processing step against before ONNX export; P6 gets real, defensible numbers for the PPT ("mean spill area across the validation set: X px² / Y km²") instead of a single demo screenshot; and it's the exact table structure the review asked for, just computed from ground truth instead of only from predictions.

---

## 8. Storage format — don't leave 8,000 loose files

Thousands of small TIFFs mean thousands of small random-access reads every epoch — painful on a shared Kaggle/Colab session, and it's an easy place to silently lose time you don't have in a 10–15 day sprint. Pack each split into a single shard after cleaning:

```
dataset_ready/
├── train_images.pt      # float32 tensor, normalized
├── train_masks.pt       # uint8 tensor
├── val_images.pt
├── val_masks.pt
├── test_images.pt
├── test_masks.pt
├── geometry.csv          # per-patch: area, perimeter, eccentricity, centroid, split, group_id
├── norm_stats.json       # clip bound, mean, std — computed on train only
├── splits.csv            # filename → group_id → split assignment (for audit/reproducibility)
└── qc_report.md          # auto-generated: pair counts, dropped files, imbalance %, sample overlay grid
```

At 8,070 patches × 256×256×1 float32, the whole packed dataset is roughly ~2 GB — fits comfortably in a Kaggle session's RAM, loads once instead of 8,070 times per epoch.

`qc_report.md` is worth generating even if nobody outside the team reads it during the hackathon — it's your evidence trail for "yes, we actually cleaned the data" instead of an unverifiable claim in the PPT.

---

## 9. Where this leaves the georeferencing problem

Section 3 already points you at checking Track B's TIFFs for an embedded GeoTransform. If they're not georeferenced, you still need what the review recommended: 3–4 full Sentinel-1 GRD GeoTIFFs of known historical spills, read via `rasterio`, with your model run on tiled/overlapping windows and predictions stitched back using the scene's transform matrix. That's a P1 Day-8-ish task once Track A/B are stable — flagging it here so it isn't forgotten, but the drift-model and AIS-side fixes from your review stay owned by P3/P4 as already reassigned.

---

## 10. Revised Day 1–3 task list for P1

| Day | Task |
|---|---|
| **1** | Download SOS refined (`images.zip` + `masks.zip`). Verify actual patch dimensions and filename pattern (Section 1). Run Stage 1 (integrity check) end to end on the raw download. |
| **2** | Run Stages 2–3 (mask QC + normalization stats). Generate the first `qc_report.md` and manually eyeball the overlay grid. Start Track B download in parallel (Section 3). |
| **3** | Run Stage 4 (scene grouping) and Stage 6 (split). Pack Stage 8 tensors. Hand `dataset_ready/` + `qc_report.md` to P2 — this is your Day-3 deliverable, matching the original plan's deadline, just with a dataset that won't quietly leak into its own validation set. |

## 11. Data-specific risk table (add to your existing risk list)

| Risk | Why it matters | Mitigation |
|---|---|---|
| Random split leaks scenes across train/val/test | Inflated IoU that collapses on real unseen data / your demo image | Group-aware split by scene ID (Section 2) |
| Trusting refined masks as ground truth | ~50% of validation masks in this exact release needed manual correction | QC pass + manual overlay review (Step 2) before training starts |
| Model never tested above 256×256 until Day 10 | Matches the review's #7 — discovered too late to fix | Track B ingestion by Day 6, sliding-window test before Day 8 |
| Normalization stats computed on full dataset | Silent val/test leakage, inflated metrics | Compute clip/mean/std on train split only (Step 3) |
| DARTIS/JPEG blended into training set unnormalized | Corrupts radiometric statistics team-wide | Keep Track C as a held-out check only, never blended (Section 3) |

---

## Sources referenced while grounding this document
- Refined Deep-SAR Oil Spill (SOS) dataset, Zenodo, DOI 10.5281/zenodo.15298010
- Zhu et al., "Oil Spill Contextual and Boundary-Supervised Detection Network Based on Marine SAR Images," IEEE TGRS, 2021
- OilSAM2 (arXiv:2603.10231) and SAM2-ELNet (arXiv:2503.12404) — both report SOS dataset composition figures
- "Sentinel-1 SAR Oil spill image dataset for train, validate, and test deep learning models" — Zenodo records 8346860, 8253899, 13761290
- DARTIS dataset description, Frontiers in Environmental Science (2026) and DARTIS project (Tandfonline, 2022)
- Large-scale oil spill SAR dataset paper (arXiv:2006.13575) — backscatter clipping rationale

*A full runnable implementation of Stages 1–8 above is in `data_pipeline_starter.py`, provided alongside this document.*
