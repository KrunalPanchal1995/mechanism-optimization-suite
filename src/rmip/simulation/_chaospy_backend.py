"""
Standalone chaospy gPC / active-subspace backend for PCESurrogateBuilder
(Phase 5, Future.md pt. 7 — surrogate modelling), invoked as a subprocess under
a dedicated venv -- NOT importable from the main rmip environment.

Why a separate venv/subprocess instead of a normal import: chaospy pulls in
numpoly, which needs a numpy the main anaconda env cannot provide without an ABI
break. chaospy 4.3.21 defaults to numpy 2.0.2, but numpoly 1.3.9 then calls
``numpy.reshape(shape=...)`` (a numpy>=2.1 kwarg) and crashes; and this repo's
shared env is pinned to numpy 1.24.4 for scikit-learn/numba/cantera. The venv
``/home/krithika/.venvs/chaospy_gpc`` pins a self-consistent
numpy==1.26.4 / numpoly==1.2.12 / chaospy==4.3.21 stack. Talking to it only via
subprocess + JSON follows the same D7 "subprocess + file contract, never import"
principle used for SALib (``_salib_backend.py``), Zero-RK, CHEMKIN_PRO, etc.

Usage:
    <venv>/bin/python3 _chaospy_backend.py sample <request.json> <response.json>
    <venv>/bin/python3 _chaospy_backend.py fit    <request.json> <response.json>

Distribution spec (per parameter, used by both commands):
    {"type": "uniform", "lower": a, "upper": b}
    {"type": "normal",  "mu": m, "sigma": s}

request.json (sample):
    {"distributions": [spec, ...], "n_samples": int,
     "rule": "latin_hypercube"|"sobol"|"halton"|"random", "seed": int|null}
response.json (sample):
    {"X": [[...], ...]}    -- (n_samples, d), natural parameter space

request.json (fit):
    {"distributions": [spec, ...], "X": [[...], ...], "Y": [...],
     "order": int, "n_as_samples": int|null, "seed": int|null}
response.json (fit):
    {"n_terms": int, "mean": float, "variance": float, "r2_insample": float,
     "sobol_main": [...], "sobol_total": [...],
     "active_subspace": {"eigenvalues": [...], "eigenvectors": [[...], ...],
                         "activity_scores": [...], "n_active": int}}
"""
import json
import sys


def _build_joint(dist_specs):
    import chaospy as cp
    marginals = []
    for spec in dist_specs:
        t = spec["type"].lower()
        if t == "uniform":
            marginals.append(cp.Uniform(spec["lower"], spec["upper"]))
        elif t == "normal":
            marginals.append(cp.Normal(spec["mu"], spec["sigma"]))
        else:
            raise ValueError(f"unsupported distribution type: {t!r}")
    return cp.J(*marginals)


def cmd_sample(req: dict) -> dict:
    J = _build_joint(req["distributions"])
    rule = req.get("rule", "latin_hypercube")
    seed = req.get("seed")
    kwargs = {}
    if seed is not None:
        kwargs["seed"] = seed
    X = J.sample(req["n_samples"], rule=rule, **kwargs)   # (d, N)
    X = X.reshape(1, -1) if X.ndim == 1 else X
    return {"X": X.T.tolist()}                             # (N, d)


def _active_subspace(surrogate, J, d, n_as, seed, fd_rel=1e-4):
    """
    Gradient-based active subspace of the fitted PCE surrogate.

    Works in the standardised input space z_i = (x_i - mu_i)/sigma_i so the
    activity scores are scale-invariant. Gradients of the (cheap) polynomial
    surrogate are taken by central finite differences; C = E[g_z g_z^T] is
    estimated over a fresh Monte-Carlo sample from the input distribution,
    then symmetric-eigendecomposed (eigenvalues descending). The number of
    "active" directions is chosen at the largest relative eigenvalue gap.
    """
    import numpy as np

    kwargs = {"seed": seed} if seed is not None else {}
    Z = J.sample(n_as, rule="latin_hypercube", **kwargs)   # (d, n_as)
    Z = Z.reshape(1, -1) if Z.ndim == 1 else Z
    mu = Z.mean(axis=1)
    sigma = Z.std(axis=1)
    sigma[sigma <= 0] = 1.0

    n = Z.shape[1]
    C = np.zeros((d, d))
    for k in range(n):
        x = Z[:, k].copy()
        g = np.zeros(d)
        for i in range(d):
            h = fd_rel * max(abs(x[i]), 1.0)
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            fp = float(np.asarray(surrogate(*xp.reshape(d, 1))).ravel()[0])
            fm = float(np.asarray(surrogate(*xm.reshape(d, 1))).ravel()[0])
            g[i] = sigma[i] * (fp - fm) / (2.0 * h)         # d f / d z_i
        C += np.outer(g, g)
    C /= n

    eigvals, eigvecs = np.linalg.eigh(C)        # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.clip(eigvals, 0.0, None)

    # Activity score of parameter i: sum_j lambda_j * W[i,j]^2
    activity = (eigvecs**2 * eigvals[np.newaxis, :]).sum(axis=1)

    # n_active from the largest multiplicative gap between consecutive eigenvalues
    n_active = 1
    if d > 1 and eigvals[0] > 0:
        ratios = eigvals[:-1] / np.clip(eigvals[1:], 1e-300, None)
        n_active = int(np.argmax(ratios) + 1)

    return {
        "eigenvalues": eigvals.tolist(),
        "eigenvectors": eigvecs.tolist(),      # columns are directions
        "activity_scores": activity.tolist(),
        "n_active": n_active,
    }


def cmd_fit(req: dict) -> dict:
    import numpy as np
    import chaospy as cp

    J = _build_joint(req["distributions"])
    d = len(req["distributions"])
    X = np.asarray(req["X"], dtype=float)       # (N, d)
    Y = np.asarray(req["Y"], dtype=float)       # (N,)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xt = X.T                                     # (d, N) for chaospy
    order = int(req.get("order", 3))

    expansion = cp.generate_expansion(order, J)
    n_terms = len(expansion)
    surrogate = cp.fit_regression(expansion, Xt, Y)

    # In-sample fit quality
    Y_pred = np.asarray(surrogate(*Xt)).ravel()
    ss_res = float(np.sum((Y - Y_pred) ** 2))
    ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mean = float(cp.E(surrogate, J))
    variance = float(cp.Var(surrogate, J))
    sobol_main = [float(v) for v in np.atleast_1d(cp.Sens_m(surrogate, J))]
    sobol_total = [float(v) for v in np.atleast_1d(cp.Sens_t(surrogate, J))]

    n_as = req.get("n_as_samples") or max(500, 50 * d)
    active = _active_subspace(surrogate, J, d, n_as, req.get("seed"))

    # Portable monomial form so the surrogate can be evaluated OUTSIDE this
    # venv (pure numpy): f(x) = sum_k coeff[k] * prod_i x_i^exponents[k,i].
    exponents = np.asarray(surrogate.exponents, dtype=int).tolist()
    coefficients = [float(np.asarray(c).ravel()[0]) for c in surrogate.coefficients]

    return {
        "n_terms": int(n_terms),
        "mean": mean,
        "variance": variance,
        "r2_insample": float(r2),
        "sobol_main": sobol_main,
        "sobol_total": sobol_total,
        "active_subspace": active,
        "pce_exponents": exponents,
        "pce_coefficients": coefficients,
    }


def cmd_mcmc(req: dict) -> dict:
    """
    Full-posterior sampling for MUM-PCE calibration (Future.md pt. 6) with emcee.

    Log-probability (up to a constant):
        -0.5 [ Σ_k ((PCE_k(x) − y_k)/σ_k)²  +  Σ_i (x_i/σ_prior_i)² ]
    with a hard box prior at ±bounds_sigma·σ_prior (the PCE trust region).

    Returns posterior mean/sigma/covariance and the acceptance fraction. The
    PCE forward models are evaluated in pure numpy from their monomial form, so
    this does not depend on chaospy — only emcee (+ numpy) in the venv.
    """
    import numpy as np
    import emcee

    targets = req["targets"]
    exps = [np.asarray(t["exponents"], dtype=float) for t in targets]
    coefs = [np.asarray(t["coefficients"], dtype=float) for t in targets]
    obs = np.array([t["observed"] for t in targets], dtype=float)
    sig = np.array([t["sigma"] for t in targets], dtype=float)
    prior_sigma = np.asarray(req["prior_sigma"], dtype=float)
    d = prior_sigma.size
    half = float(req.get("bounds_sigma", 2.0)) * prior_sigma

    def pce_eval(exp, coef, x):
        return float(coef @ np.prod(x[np.newaxis, :] ** exp, axis=1))

    def log_prob(x):
        if np.any(x < -half) or np.any(x > half):
            return -np.inf
        pred = np.array([pce_eval(exps[k], coefs[k], x) for k in range(len(targets))])
        chi = np.sum(((pred - obs) / sig) ** 2)
        prior = np.sum((x / prior_sigma) ** 2)
        return -0.5 * (chi + prior)

    n_walkers = int(req.get("n_walkers") or max(2 * d + 2, 2 * (d + 1)))
    if n_walkers % 2 == 1:
        n_walkers += 1
    n_walkers = max(n_walkers, 2 * d + 2)
    n_steps = int(req.get("n_steps", 4000))
    n_burn = int(req.get("n_burn", 1000))
    seed = req.get("seed")
    rng = np.random.default_rng(seed)

    # Initialise walkers in a small ball around the prior mean (0), inside bounds.
    p0 = rng.normal(0.0, 0.1, size=(n_walkers, d)) * prior_sigma[np.newaxis, :]
    p0 = np.clip(p0, -0.99 * half, 0.99 * half)

    sampler = emcee.EnsembleSampler(n_walkers, d, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=False)

    chain = sampler.get_chain(discard=n_burn, flat=True)
    mean = chain.mean(axis=0)
    sigma = chain.std(axis=0)
    cov = np.cov(chain, rowvar=False)
    return {
        "mean": mean.tolist(),
        "sigma": np.atleast_1d(sigma).tolist(),
        "cov": np.atleast_2d(cov).tolist(),
        "n_samples": int(chain.shape[0]),
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
    }


_COMMANDS = {"sample": cmd_sample, "fit": cmd_fit, "mcmc": cmd_mcmc}


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in _COMMANDS:
        print("Usage: _chaospy_backend.py sample|fit|mcmc <request.json> <response.json>",
              file=sys.stderr)
        sys.exit(2)
    cmd, req_path, resp_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(req_path) as fh:
        req = json.load(fh)
    result = _COMMANDS[cmd](req)
    with open(resp_path, "w") as fh:
        json.dump(result, fh)


if __name__ == "__main__":
    main()
