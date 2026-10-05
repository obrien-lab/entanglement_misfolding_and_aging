"""Published analysis: survival fit (Eq. 10), apparent unfolding rate, super-Arrhenius fit (Eq. 11)."""
import numpy as np
from scipy.optimize import curve_fit


def survival_curve(fpt_ns, censored, cap_ns, grid_ns):
    """S_U(t) on a time grid up to the cap; censored replicas count as folded at every t <= cap."""
    fpt = np.where(np.asarray(censored, bool), np.inf, np.asarray(fpt_ns, float))
    t = np.arange(0.0, cap_ns + 0.5 * grid_ns, grid_ns)
    return t, (fpt[None, :] > t[:, None]).mean(axis=1)


def fit_window_end(fpt_ns, censored, cap_ns, pad_ns):
    """End of the Eq. 10 fit window: last unfolding time + pad_ns, never beyond the cap.
    pad_ns=None, or no replica unfolded, gives the cap (fit over t <= cap)."""
    events = np.asarray(fpt_ns, float)[~np.asarray(censored, bool)]
    if pad_ns is None or events.size == 0:
        return float(cap_ns)
    return min(float(events.max()) + float(pad_ns), float(cap_ns))


def fit_survival(fpt_ns, censored, cap_ns, grid_ns, pad_ns):
    """S_U(t) restricted to the fit window, and its Eq. 10 fit. Returns (t, S, fit or None, window_end_ns)."""
    t, S = survival_curve(fpt_ns, censored, cap_ns, grid_ns)
    t_end = fit_window_end(fpt_ns, censored, cap_ns, pad_ns)
    keep = t <= t_end + 1e-9                     # grid and unfolding times are both multiples of the save interval
    t, S = t[keep], S[keep]
    return t, S, fit_eq10(t, S, fpt_ns, censored), t_end


def eq10(t, t0, k):
    return np.where(t < t0, 1.0, np.exp(-k * (t - t0)))


def fit_eq10(t, S, fpt_ns, censored):
    """Least-squares fit of Eq. 10 with a few starting points; returns None if there is nothing to fit."""
    events = np.sort(np.asarray(fpt_ns, float)[~np.asarray(censored, bool)])
    sst = float(np.sum((S - S.mean()) ** 2))
    if events.size == 0 or sst == 0.0:
        return None
    best = None
    for t0_init in sorted({float(events[0]), 0.5 * float(events[0]), float(np.percentile(events, 25))}):
        k_init = 1.0 / max(float(events.mean()) - t0_init, 1e-3)
        try:
            (t0, k), _ = curve_fit(eq10, t, S, p0=[t0_init, k_init], bounds=([0.0, 1e-9], [float(t[-1]), 1e4]),
                                   method="trf", maxfev=5000)
        except Exception:
            continue
        sse = float(np.sum((S - eq10(t, t0, k)) ** 2))
        if best is None or sse < best[2]:
            best = (float(t0), float(k), sse)
    if best is None:
        return None
    t0, k, sse = best
    return dict(t0_ns=t0, k_per_ns=k, k_app_per_ns=1.0 / (t0 + 1.0 / k), R2=1.0 - sse / sst)


def fit_eq11(temps_K, k_app):
    """ln k_app = a/T^2 + b/T + c  (a quadratic in 1/T)."""
    x = 1.0 / np.asarray(temps_K, float)
    y = np.log(np.asarray(k_app, float))
    a, b, cc = np.polyfit(x, y, 2)
    pred = a * x ** 2 + b * x + cc
    ss = float(np.sum((y - y.mean()) ** 2))
    return dict(a=float(a), b=float(b), c=float(cc), R2=1.0 - float(np.sum((y - pred) ** 2)) / ss if ss > 0 else float("nan"))


def k_app_at(fit, temp_K):
    x = 1.0 / float(temp_K)
    return float(np.exp(fit["a"] * x ** 2 + fit["b"] * x + fit["c"]))
