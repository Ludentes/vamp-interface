"""Extract FLAME neutral mesh + faces into a chumpy-free npz.

flame2023.pkl stores chumpy arrays; this loads it behind a numpy-alias shim and
a minimal chumpy stub so the pickle resolves, then writes plain numpy. Run once:
    PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.prep_flame_assets
"""
import pickle
import shutil
import sys
import types
from pathlib import Path

import numpy as np

LAM = Path("/home/newub/w/LAM/model_zoo/human_parametric_models")
FLAME_PKL = LAM / "flame_vhap" / "flame2023.pkl"
BASIS_SRC = LAM / "flame_assets" / "flame_arkit_bs.npy"
OUT_DIR = Path("output/flame_assets")


def _install_chumpy_shim() -> None:
    """Make `import chumpy` resolve to a stub whose Ch objects unpickle as
    plain ndarrays. flame2023.pkl only needs the array values, not chumpy's
    autodiff graph."""
    for alias, real in [("bool", np.bool_), ("int", np.int_),
                         ("float", np.float64), ("object", np.object_),
                         ("str", np.str_), ("complex", np.complex128)]:
        if not hasattr(np, alias):
            setattr(np, alias, real)

    class Ch:
        """A chumpy.Ch stand-in: unpickles by absorbing its dict state and
        exposing the raw ndarray value via `.r`. chumpy.Ch is pickled as an
        ordinary object (GLOBAL + __reduce_ex__), so a plain class with
        __setstate__ is enough — we never need the autodiff graph."""

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

        @property
        def r(self):
            """chumpy stores the materialised value under `x` (the dterms
            input). Walk the few known value attributes; fall back to dict."""
            for attr in ("x", "_result", "_r"):
                v = self.__dict__.get(attr)
                if v is not None:
                    return np.asarray(getattr(v, "r", v))
            raise AttributeError("no value attribute on Ch stub")

    chumpy = types.ModuleType("chumpy")
    chumpy.Ch = Ch
    ch_mod = types.ModuleType("chumpy.ch")
    ch_mod.Ch = Ch
    chumpy.ch = ch_mod
    sys.modules.setdefault("chumpy", chumpy)
    sys.modules.setdefault("chumpy.ch", ch_mod)


def _to_array(x) -> np.ndarray:
    """chumpy arrays expose `.r` for their raw ndarray value; plain arrays don't."""
    return np.asarray(getattr(x, "r", x))


def build() -> None:
    if not FLAME_PKL.exists():
        raise FileNotFoundError(
            f"{FLAME_PKL} missing — is the FLAME model zoo in place?")
    _install_chumpy_shim()
    with open(FLAME_PKL, "rb") as f:
        model = pickle.load(f, encoding="latin1")

    v_template = _to_array(model["v_template"]).astype(np.float32).reshape(-1, 3)
    faces = np.asarray(model["f"]).astype(np.int32)
    assert v_template.shape == (5023, 3), v_template.shape
    assert faces.ndim == 2 and faces.shape[1] == 3, faces.shape

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "flame_base.npz", v_template=v_template, faces=faces)
    shutil.copy(BASIS_SRC, OUT_DIR / "flame_arkit_bs.npy")
    print(f"wrote {OUT_DIR / 'flame_base.npz'} — {len(v_template)} verts, "
          f"{len(faces)} faces; copied basis")


if __name__ == "__main__":
    build()
