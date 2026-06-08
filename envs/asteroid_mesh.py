"""Procedurally generate irregular ("potato") asteroid meshes as OBJ files.

Pure NumPy (no trimesh): build an icosphere by subdividing an icosahedron, then push
each vertex radially by a sum of random Gaussian bumps/craters so the surface is lumpy
and visually reads as a rock rather than a sphere. MuJoCo loads the OBJ as a mesh geom
(collision uses the convex hull, which is fine; the visible surface keeps its bumps).

Run once to (re)build the library:
    conda run -n asteroid-belt-runner python envs/asteroid_mesh.py
Writes assets/asteroids/asteroid_0.obj ... asteroid_{N-1}.obj (unit-ish radius).
"""
import os

import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "asteroids")
N_MESHES = 12
SUBDIV = 2  # icosahedron subdivisions: 2 -> 320 faces (enough for visible bumps, cheap hull)


def _icosahedron():
    t = (1.0 + 5 ** 0.5) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=int)
    return verts, faces


def _subdivide(verts, faces):
    verts = list(map(tuple, verts))
    index = {v: i for i, v in enumerate(verts)}
    cache = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        m = (np.array(verts[a]) + np.array(verts[b])) / 2.0
        m = m / np.linalg.norm(m)
        mt = tuple(m)
        idx = index.get(mt)
        if idx is None:
            idx = len(verts)
            verts.append(mt)
            index[mt] = idx
        cache[key] = idx
        return idx

    new_faces = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.array(verts, dtype=float), np.array(new_faces, dtype=int)


def _lumpy(verts, rng):
    """Displace unit-sphere vertices radially by random Gaussian bumps and craters."""
    dirs = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    r = np.ones(len(verts))
    # a few broad lobes for overall irregular silhouette + sharper bumps/craters
    n_features = rng.integers(8, 16)
    for _ in range(n_features):
        c = rng.normal(size=3)
        c /= np.linalg.norm(c)
        sharp = rng.uniform(2.0, 14.0)           # higher -> more localized
        amp = rng.uniform(-0.28, 0.38)           # negative -> crater
        r += amp * np.exp(sharp * (dirs @ c - 1.0))
    r = np.clip(r, 0.55, 1.6)
    return dirs * r[:, None]


def _write_obj(path, verts, faces):
    with open(path, "w") as f:
        f.write("# procedurally generated asteroid\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def build_library(n_meshes=N_MESHES, subdiv=SUBDIV, seed=12345):
    os.makedirs(ASSET_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    base_v, base_f = _icosahedron()
    for _ in range(subdiv):
        base_v, base_f = _subdivide(base_v, base_f)
    paths = []
    for i in range(n_meshes):
        v = _lumpy(base_v, rng)
        # recentre so the mesh COM is near the origin (cleaner free-body dynamics)
        v -= v.mean(axis=0)
        path = os.path.join(ASSET_DIR, f"asteroid_{i}.obj")
        _write_obj(path, v, base_f)
        paths.append(path)
    print(f"wrote {n_meshes} asteroid meshes ({len(base_f)} faces each) to {ASSET_DIR}")
    return paths


if __name__ == "__main__":
    build_library()
