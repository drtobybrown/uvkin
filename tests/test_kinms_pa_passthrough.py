from __future__ import annotations

import types

import numpy as np
import pytest

KinMSModel = pytest.importorskip("uvfit.forward_model").KinMSModel


def test_kinms_posang_receives_pa_init_passthrough(monkeypatch):
    class FakeKinMS:
        last_kwargs = None

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs

        def model_cube(self, **kwargs):
            FakeKinMS.last_kwargs = kwargs
            # KinMS returns (x, y, v)
            return np.zeros((8, 8, 9), dtype=np.float32)

    monkeypatch.setitem(__import__("sys").modules, "kinms", types.SimpleNamespace(KinMS=FakeKinMS))

    radius = np.linspace(0.1, 5.0, 16)
    model = KinMSModel(
        xs=8,
        ys=8,
        vs=9,
        cell_size_arcsec=0.2,
        channel_width_kms=5.0,
        sbprof=np.exp(-radius),
        velprof=np.linspace(10.0, 200.0, radius.size),
        sbrad=radius,
        velrad=radius,
    )
    cube = model.generate_cube(
        {"inc": 55.0, "pa": 205.0, "flux": 50.0, "vsys": 8300.0, "gas_sigma": 12.0}
    )
    assert cube.shape == (9, 8, 8)
    assert FakeKinMS.last_kwargs is not None
    assert FakeKinMS.last_kwargs["posAng"] == 205.0
