Case 52 (53/53) ───────────────────────────────────────────────────────────

  ┌─ [Case-52] STEP 1/8 · LOAD case data
  │   n_response                     = 312
  │   n_dm_rows                      = 312
  │   n_selected_params              = 11
  │   n_full_params                  = 22
  └──────────────────────────────────────────────────────────
  ✗ ERROR: [Case-52] PRS fitting FAILED: IndexError: too many indices for array: array is 0-dimensional, but 2 were indexed
Traceback (most recent call last):
  File "/home/krithika/mechanism-optimization-suite/src/MUQ-SAC/phase10_prs.py", line 1218, in run
    out = self._run_case(case)
  File "/home/krithika/mechanism-optimization-suite/src/MUQ-SAC/phase10_prs.py", line 824, in _run_case
    blocks = self.IPRS.build_layout_blocks(
  File "/home/krithika/mechanism-optimization-suite/src/MUQ-SAC/IdentifiablePRS.py", line 188, in build_layout_blocks
    _, L_r = reduced_cholesky(Sigma, indices)
  File "/home/krithika/mechanism-optimization-suite/src/MUQ-SAC/IdentifiablePRS.py", line 86, in reduced_cholesky
    Sigma_r = Sigma[np.ix_(idx, idx)]
IndexError: too many indices for array: array is 0-dimensional, but 2 were indexed


This is the bug that is created. Code the module so that it always generats the plots as well as it always do the 10 fold cross validation each time
