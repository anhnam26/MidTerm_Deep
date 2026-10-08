# DataNew cleanup — partial result

Both resolutions now share **22,230 identities: 21,980 train + 250 test**.

| Class | Train | Test | Train shortage to 5,000 |
|---|---:|---:|---:|
| apple | 5,000 | 50 | 0 |
| banana | 4,480 | 50 | 520 |
| orange | 5,000 | 50 | 0 |
| strawberry | 4,815 | 50 | 185 |
| pineapple | 2,685 | 50 | 2,315 |
| **Total** | **21,980** | **250** | **3,020** |

- Removed 3,019 manifest-augmented identities and one additional exact duplicate.
- Every removal applies to both 224x224 AVIF and 32x32 WebP.
- No new replacements were promoted: sources were reserved/development-only or insufficiently verified. Acquisition metadata, approvals, retirements and reservations remain unchanged.
- Retained images were copied byte-for-byte; all 250 test files per resolution remain unchanged.
- Full staging decode verified RGB, geometry, formats and pairing; zero exact file/pixel duplicate groups. Image bytes: **20,369,509 / 25,000,000** for Mid224 and **13,032,886 / 20,000,000** for Mid32.
- 16,000 retained identities have conservative dHash similarity flags. They are **not 16,000 confirmed duplicates**. These require actual content/group review; dHash alone cannot establish independent captures or leakage.
- Legacy source names do not prove semantic/rights approval or independent physical groups. **Quota, near-duplicate review and source verification are incomplete. Not training-ready.**

## Artifacts (absolute paths)

- Manifest: `D:\DeepLearning\MidTerm\results\dataset_manifest.csv`
- Audit: `D:\DeepLearning\MidTerm\DataNew\cleanup_report.json`
- Provenance: `D:\DeepLearning\MidTerm\DataNew\source_provenance.json`
- Verified backup: `D:\DeepLearning\MidTerm\outputs\checkpoints\datanew_cleanup_15\backup`
- Original directory: `D:\DeepLearning\MidTerm\outputs\checkpoints\datanew_cleanup_15\previous_DataNew`
- Removal decisions, candidate exclusions, unresolved similarity flags and hash inventories: `D:\DeepLearning\MidTerm\outputs\checkpoints\datanew_cleanup_15`

Abandoned intermediate staging directories contain **rejected, never-promoted trials**, not eligible dataset sources. They are retained for traceability and must not be mistaken for final replacements.

## Checks

130 automated tests passed with process-local `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1`; no package, interpreter or persistent environment changes. One earlier full-suite run hit a native PyTorch convolution access violation; retry with bounded threads passed. Cleanup-specific tests include paired removal, test priority, similarity-vs-proof, source eligibility/reservation and failed-promotion protection. Post-promotion full decode of all 44,460 images matched the audited staging inventory; every retained file, all test files and acquisition metadata were verified unchanged.