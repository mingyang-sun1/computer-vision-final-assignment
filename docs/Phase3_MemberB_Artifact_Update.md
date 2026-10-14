# Phase 3 – Member B original artifact handoff (13 October 2026)

**Owner:** Member A / augmentation evaluation. **Source:** Original files sent by Member B in `phase3.zip`. This is a dated supplement to `docs/Phase3_Verification.md` and **supersedes only its 12 October artifact-availability statements**. It does not replace the earlier CSV audit.

## 1. Files received and integrity checks

| File | Evidence status | Size (bytes) |
|---|---|---:|
| `phase3_E2_finetune.json` | Readable original training log | 1,306 |
| `phase3_E4_augment.json` | Readable original training log | 1,362 |
| `finetuned_E2_finetune.keras` | Valid Keras-format ZIP, model metadata/configuration/weights present | 221,150,770 |
| `finetuned_E4_augment.keras` | Valid Keras-format ZIP, model metadata/configuration/weights present | 221,150,770 |

The outer ZIP CRC and both nested Keras ZIP CRC checks passed. SHA-256 identifiers and Keras metadata are stored in `docs/evidence/phase3_memberb_original_artifacts_20261013.json`. **No model has been loaded or executed in this handoff.** ZIP integrity does not prove successful TensorFlow loading or prove that a saved model generated the committed prediction CSV.

## 2. Recorded training results

| Measure | E2: no augmentation | E4: augmentation |
|---|---:|---:|
| Validation MSE | 102745.6903 | 108784.4670 |
| Validation MAE | 235.6500 | 246.3192 |
| Validation R² | 0.334655 | 0.295550 |
| Epochs executed | 15 | 15 |
| Best epoch from logged validation loss | 12 (derived) | 14 (matches recorded field) |
| Lowest validation loss | 0.654432 | 0.693139 |
| Training loss at epoch 10 | 0.079882 | 0.197189 |

The original logs report E4 MSE **5.8774% higher** than E2 (difference **6038.7767**). This agrees in direction and approximate magnitude with the independent October 12 audit of the rounded committed CSV exports; slight numerical differences are expected because the CSV predictions are rounded. This is not a new training experiment.

## 3. Matching conditions and missing original fields

Both JSON files explicitly record `seed=42`, image size 224, learning rate 0.0001, batch size 32, 15 executed epochs, a trainable last CNN stage plus regression head, 15,500,801 trainable parameters and 24,112,513 total parameters. Their recorded validation losses have 15 entries each.

The original files do **not** use the full schema of the currently committed `src/exp_finetune.py`:

- E2 uses `experiment=E2_partial_finetuning`, rather than the current run tag `E2_finetune`, and omits `augmentation`, `epochs_cap`, `patience` and `best_epoch`. Its best epoch **12** was calculated from the original `history.val_loss` array; it was not present as a log field.
- E4 explicitly records `augmentation=true`, `epochs_cap=15` and `best_epoch=14`, but omits `patience`.

Do not edit the originals or fill missing values from current code defaults. It is reasonable to identify E2 as the no-augmentation comparison using the established experiment naming and code history, but its legacy JSON does not independently store the false augmentation flag. Equality of historical patience settings, and equality of the configured epoch cap, cannot be fully established from the two files alone. The two **executed** epoch counts do match.

## 4. Follow-up on the local Windows repository

Place the originals in these paths:

```text
outputs/logs/phase3_E2_finetune.json
outputs/logs/phase3_E4_augment.json
outputs/models/finetuned_E2_finetune.keras
outputs/models/finetuned_E4_augment.keras
```

**Replace the old checker and its Phase 3 tests** with the updated files in this handoff. The October 12 checker required newer metadata fields and would have marked these real legacy logs as FAIL. This update supports the actual schema, clearly labels absent original fields as PENDING, and does not silently supply inferred defaults.

From the repository root in PowerShell (use `py` if the `python` alias does not work):

```powershell
py -3 tools/verify_phase3_augmentation.py --repo-root . --out docs/evidence/phase3_with_original_artifacts_20261013.json
py -3 -m unittest discover -s tests -p "test_verify_phase3_augmentation.py" -v
```

**Output files must be new names**: if `phase3_with_original_artifacts_20261013.json` already exists, use a different name. The checker is read-only and loads no models. The package's synthetic Phase 3 tests were run here: **22 tests passed**; the actual full-repository combined run must still be performed on the user's computer.

The patched checker may still mark metadata checks PENDING because the historical fields were not saved. By default an audit with no FAIL checks exits **0**, while retaining PENDING records. Optionally add `--require-artifacts` to make pending evidence exit **2**. Inspect `PASS`, `FAIL` and `PENDING` in the JSON report rather than treating exit 0 as proof of full training reproduction. Actual results depend on the local split/prediction files being unchanged and accessible.

The course-provided data and large Keras checkpoints should not be committed to GitHub. Keep the original models in a team-controlled shared location or local backup. Their presence is enough for inventory, not model execution verification.

## 5. Report-ready paragraph

The E2 and E4 experiments compare partial ResNet-50 fine-tuning without and with moderate augmentation. Both recorded runs used 224-pixel input images, seed 42, learning rate 0.0001, batch size 32 and 15 training epochs. Original JSON logs report validation MSE of 102,745.69 for E2 and 108,784.47 for E4, so augmentation had 5.88% higher MSE in these runs. The best recorded validation-loss epochs were 12 for E2 (derived from its history) and 14 for E4. E4 also showed a higher training loss at epoch 10, although augmented and unchanged inputs make the two training-loss curves not directly comparable as evidence of generalisation. This comparison tests a combined set of augmentations in one run per condition. It does not show that augmentation is generally harmful or that a longer training schedule would necessarily improve E4. Some configuration metadata was not saved in the older logs, so the comparison is supported but not fully reproducible from these files alone.

**Remaining handoff:** Run the full verification against local repository files; retain original checkpoints and their hashes in shared storage; record Member B's acknowledgement/review; optionally load saved models and check prediction provenance separately if required for final reproducibility.
