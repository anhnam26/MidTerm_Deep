# DataNew cleanup — partial result

Both resolutions now share **22,303 identities: 22,053 train + 250 test**.

| Class | Train | Test | Train shortage to 5,000 |
|---|---:|---:|---:|
| apple | 5,000 | 50 | 0 |
| banana | 4,480 | 50 | 520 |
| orange | 5,000 | 50 | 0 |
| strawberry | 4,888 | 50 | 112 |
| pineapple | 2,685 | 50 | 2,315 |
| **Total** | **22,053** | **250** | **2,947** |

- Removed 3,019 manifest-augmented identities and one additional exact duplicate.
- Every removal applies to both 224x224 AVIF and 32x32 WebP.
- Session15 promoted zero replacements. Session16 then appended **73 strawberry originals**, one per unreserved author-indexed specimen from `yihfeng/strawberry-maturity`. Author metadata declares MIT, iPhone14 capture and three views per specimen. All 11 contact sheets were visually reviewed; repeated views were not counted as independent samples. Five baseline suspects and nine additional native/224 cross-reference suspects were excluded. Review evidence is bound to source hashes; this is **agent review, not user confirmation**. Acquisition metadata, approvals, retirements and reservations remain unchanged.
- Retained images were copied byte-for-byte; all 250 test files per resolution remain unchanged.
- Full staging and post-promotion decode verified RGB, geometry, formats and pairing for **44,606 files**; zero exact file/pixel duplicate groups. Image bytes: **20,425,389 / 25,000,000** for Mid224 and **13,071,104 / 20,000,000** for Mid32. New-source encoded224 similarity crosscheck: zero flags among73 or against retained references, not exhaustive proof of all possible transformations.
- 16,000 retained identities have conservative dHash similarity flags. They are **not 16,000 confirmed duplicates**. These require actual content/group review; dHash alone cannot establish independent captures or leakage.
- Legacy source names do not prove semantic/rights approval or independent physical groups. **Quota, near-duplicate review and source verification are incomplete. Not training-ready.**

## Artifacts (absolute paths)

- Manifest: `D:\DeepLearning\MidTerm\results\dataset_manifest.csv`
- Audit: `D:\DeepLearning\MidTerm\DataNew\cleanup_report.json`
- Provenance: `D:\DeepLearning\MidTerm\DataNew\source_provenance.json`
- Verified backup: `D:\DeepLearning\MidTerm\outputs\checkpoints\datanew_cleanup_15\backup`
- Original directory: `D:\DeepLearning\MidTerm\outputs\checkpoints\datanew_cleanup_15\previous_DataNew`
- Removal decisions, candidate exclusions, unresolved similarity flags and hash inventories: `D:\DeepLearning\MidTerm\outputs\checkpoints\datanew_cleanup_15`
- Latest verified backup and previous directory: `D:\DeepLearning\MidTerm\outputs\checkpoints\datanew_fill_16\reviewed_replenish_final`
- Bound new-source review: `D:\DeepLearning\MidTerm\DataNew\replenishment_review.json`
- New-source discovery/rejection checkpoints: `D:\DeepLearning\MidTerm\outputs\checkpoints\datanew_fill_16`

Abandoned intermediate staging directories contain **rejected, never-promoted trials**, not eligible dataset sources. They are retained for traceability and must not be mistaken for final replacements.

## Checks

Latest **140 automated tests passed** with process-local OMP/MKL and explicit PyTorch intra/inter-op thread limits of1; no package, interpreter or persistent environment changes. A full-suite attempt with only OMP/MKL limits hit the previously observed native convolution access violation; explicit PyTorch thread limits passed. Tests cover native/224 screening, aliases, reservations, bounded acquisition, no autoapproval, resume integrity, paired cleanup and failed promotion. Post-promotion full decode of all44,606files matched staging; all previous22,230identities, test files and acquisition metadata were verified unchanged. Backup44,462files verified; previous directory retained.

## Remaining acquisition blockers

Session16 downloaded **35 original files for review** across three source families:12strawberry Zenodo samples,6banana AgroVisionBD samples,17MiniFruit target images. None were quota-approved. MiniGrocery contains the same17targetmembers (name/size/CRC), excluded without repeat download; visual review also exposed same-fruit multi-view series missed by dHash. Zenodo20286203 has3100PNG1008x756images matching StrawDI's published structure and appearance: likely mirror, not conclusively pixel-proven; original StrawDI prohibits image redistribution, so uploader CC BY alone is insufficient. AgroVisionBD has463original banana-fruit files and primary CC BY evidence, but its README expressly warns specimen identifiers may be unavailable. Commons metadata yielded497candidates thenHTTP429; first image download also429, stopped without bypass. PQC500 remains unlicensed at primary repository; other sources are unknown-license, augmented/mirrors, leaf-only, inaccessible, or paper figures rather than original datasets.

**Still partial: banana520, strawberry112, pineapple2315.** No background replenishment worker remains after verified promotion; next steps require normal permitted access after rate-limit recovery or original-author rights/specimen evidence. Do not treat discovery candidates or abandoned staging as accepted images.