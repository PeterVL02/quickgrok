"""Check that the regression recovers known functions."""

import math

import torch

from quickgrok.regress import fit_neurons, make_grid


def test_fits() -> None:
    """Trig basis finds an exact cos(w(a+b)) at the right frequency; poly basis fits a quadratic."""
    p = 31
    a, b = make_grid(p)
    w = 2 * math.pi * 5 / p
    trig_target = torch.cos(w * (a + b).double()) + 0.5
    x, y = 2 * a.double() / (p - 1) - 1, 2 * b.double() / (p - 1) - 1
    poly_target = x**2 - 3 * x * y + 1
    dead = torch.zeros(p * p, dtype=torch.float64)
    acts = torch.stack([trig_target, poly_target, dead], dim=1)

    trig = fit_neurons(acts, a, b, p, "trig", 0, [0, 1, 2])
    assert trig[0]["r2"] > 0.999 and trig[0]["top_freq"] == 5
    assert trig[0]["top_freq_share"] > 0.999
    assert trig[1]["r2"] < 0.99  # a smooth non-periodic function is not exactly trig
    assert math.isnan(trig[2]["r2"]) and trig[2]["top_freq"] is None

    poly = fit_neurons(acts, a, b, p, "poly", 2, [0, 1, 2])
    assert poly[1]["r2"] > 0.999
    # a periodic function is badly fit by a low-degree polynomial
    assert poly[0]["r2"] < 0.5
