# Phase 4 — Member B Review and Handoff

The revised plan assigns Member A the experiment comparison, figures and
selection write-up, with both members deciding the final model. This is not
a request to restart Phase 3 or to rerun all experiments.

## Message for Member B

I have prepared the Phase 4 comparison and model-selection record against
main commit `90959c4`. The current recommendation is to retain the existing
equal-weight E2 + E3 + E4 ensemble, based on its reported validation MSE.
I will attach the new local CSV audit before final approval.

Please confirm the original E3 run/checkpoint location and the identity of
the two stored V3 prediction columns, together with any additional completed
experiment that could change the decision. Please review the comparison
conditions and confirm whether you agree with E5, then record your actual
review date and the audit directory in `Phase4_Model_Selection.md`.

Your E2/E4 original logs and models were already recorded in the Phase 3
handoff. They do not need to be sent again solely because the large files
are excluded from GitHub. We should retain the existing copies and their
provenance for Phase 7 reproducibility.

If E5 is retained, Phase 5 should analyse the audit's ensemble validation
predictions. The current E3 error analysis remains a single-model reference,
not the final ensemble's error analysis.

## Review evidence

Use `Phase4_Model_Selection.md`, `tables/Phase4_Comparison.md`, and the newly
created `docs/evidence/phase4_run_<timestamp>/audit.json`. Inspect FAIL and
PENDING separately: synthetic tests and file-presence checks do not prove
original model execution or team approval. Do not fill missing legacy log
fields from current code defaults or backdate the decision.
