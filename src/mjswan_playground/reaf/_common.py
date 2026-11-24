"""What the scenes share: files from GitHub at pinned commits, cached under ``.cache/``,
and Playground's floor. Playground builds its scenes with MuJoCo 4 and PyTorch, while
mjswan runs MuJoCo 3.11, so the scenes are restated rather than imported."""

from __future__ import annotations

import os
import posixpath
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco

from mjswan_playground._deps import CACHE_DIR

#: Every file comes from one of these, at the commit given.
MENAGERIE = (
    "google-deepmind/mujoco_menagerie",
    "370c43965323c0944b433b202aac5058b5f907a9",
)


def fetch(repo: tuple[str, str], path: str) -> Path:
    """One file of a GitHub repository at its pinned commit, downloaded once."""
    name, commit = repo
    target = CACHE_DIR / "reaf" / name.replace("/", "_") / commit / path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{name}/{commit}/{path}"
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(_download(url))
        partial.replace(target)
    return target


def _download(url: str, attempts: int = 5) -> bytes:
    """`url`'s body, waiting out a 429 or 503 for its Retry-After, else backing off."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (429, 503) or attempt == attempts - 1:
                raise
            time.sleep(float(error.headers.get("Retry-After") or 2**attempt))
    raise AssertionError("unreachable")


def _fetch_mjcf(repo: tuple[str, str], path: str) -> Path:
    """An MJCF and every mesh and texture file it names."""
    xml_path = fetch(repo, path)
    root = ET.parse(xml_path).getroot()
    compiler = {k: v for c in root.iter("compiler") for k, v in c.attrib.items()}
    asset_dir = compiler.get("assetdir", "")
    dirs = {
        "mesh": compiler.get("meshdir", asset_dir),
        "texture": compiler.get("texturedir", asset_dir),
    }
    files = [
        posixpath.normpath(posixpath.join(posixpath.dirname(path), directory, file))
        for tag, directory in dirs.items()
        for element in root.iter(tag)
        if (file := element.get("file"))
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda file: fetch(repo, file), files))
    return xml_path


def load(
    repo: tuple[str, str], path: str, edit: Callable[[str], str] | None = None
) -> mujoco.MjSpec:
    """An MJCF with every mesh and texture path made absolute: once attached to another
    spec, an asset keeps only its file name, and the build could not find the file.
    `edit` rewrites the XML first, into a copy beside it."""
    xml_path = _fetch_mjcf(repo, path)
    if edit is not None:
        edited = xml_path.with_name(f"{xml_path.stem}.edited{xml_path.suffix}")
        edited.write_text(edit(xml_path.read_text()))
        xml_path = edited
    spec = mujoco.MjSpec.from_file(str(xml_path))
    base = spec.modelfiledir or ""
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = os.path.abspath(os.path.join(base, spec.meshdir, mesh.file))
    for texture in spec.textures:
        if texture.file:
            texture.file = os.path.abspath(
                os.path.join(base, spec.texturedir, texture.file)
            )
    return spec


def name_unnamed_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Name every geom, so that attaching gives each the prefix."""
    for body in spec.bodies:
        for i, geom in enumerate(body.geoms):
            if not geom.name:
                geom.name = f"{body.name}_geom_{i}"
    return spec


#: Playground's `_src/objects/assets/primitive/plane.xml`.
PLANE_XML = """
<mujoco model="plane">
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10">
    <flag eulerdamp="disable"/>
  </option>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
      height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.2 0.3" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
      reflectance="0.2"/>
  </asset>
  <worldbody>
    <light name="light" pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <light name="front_fill" pos="0 -1.5 1.5" dir="0 1 -1" directional="true"
      diffuse="0.25 0.25 0.25" specular="0 0 0" castshadow="false"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1"/>
  </worldbody>
</mujoco>
"""


def add_floor(spec: mujoco.MjSpec) -> None:
    """Playground's plane around an mjlab scene."""
    plane = mujoco.MjSpec.from_string(PLANE_XML)
    # mjlab sets the options from `SimulationCfg`; the plane's would only conflict.
    plane.option = spec.option
    spec.attach(plane, prefix="", frame=spec.worldbody.add_frame())
