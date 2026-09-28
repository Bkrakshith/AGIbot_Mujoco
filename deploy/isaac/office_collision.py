"""Static collision for the Isaac Sim Office environment.

  from office_collision import add_office_collision
  counts = add_office_collision(stage, "/World/Office", material=friction_mat)

Makes sure the walls, doors, furniture and larger props of the referenced
office are static colliders (no rigid bodies), removes collision from the
clutter on desks and floor, and sets the friction material, without editing
the asset files.

Almost every object in the office is an instanceable reference to a prop
file, so its meshes are instance proxies and cannot be edited directly (a
plain traversal finds only the ground plane). The prop files already carry a
triangle-mesh CollisionAPI on every mesh, 3645 of them, and PhysX does create
them all. To change an object we author one "over" per instance prototype
under an abstract class prim, holding the physics opinions at the meshes'
relative paths, and add an internal reference to it on the instances
concerned. The reference is part of the instancing key, so instances with the
same treatment still share one prototype and PhysX cooks each mesh once.

Static shapes cost next to nothing per physics step (the office with no
colliders at all is no faster), while every overridden instance adds a
little per-step overhead in Kit (~0.3 ms per 1000 instances). So only what
matters is changed:

  none (triangle mesh)  walls, doors, columns, stairs, tables, desks, counters,
                        chairs, sofas, shelves, cabinets and anything > 1.5 m:
                        as shipped, left alone
  convexHull            other props between 0.3 m and 1.5 m (boxes, plants,
                        bins, printers, monitors, ...): overridden
  off                   clutter smaller than 0.3 m (pens, cups, books,
                        phones) and flat items (paper, keyboards, screens),
                        which would tip objects put on a desk: collision
                        removed
  left as shipped       wall decoration, ceilings, lights, ducts, anything
                        above 1.9 m, the storeys below and above and the city
                        outside: never touched by the robot
The floor is the ground plane. The friction material is bound to the physics
scene as its default material, which covers every office shape without a
material of its own.
"""

import re
from collections import Counter, defaultdict

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdSkel

CLASS_ROOT = "/World/OfficeColliders"
OVERHEAD_Z = 1.9        # m, objects starting above this cannot touch the robot
FLOOR_Z = 0.05          # m, objects ending below this are on/under the floor
SMALL = 0.3             # m, largest extent of clutter without collision
BIG = 1.5               # m, larger objects always get a triangle mesh

OFF_NAMES = re.compile(
    r"Building|SkySphere|Ceiling|SM_Floor|Carpet|SM_Mat|Sticker|Picture|SM_Frame\d|Sign|Plate|Clock|"
    r"Smoke|Sprinkler|Deluge|Ventilation|AirShaft|Pipe|Valve|CCTV|Blinds|Billboard|Advertising|"
    r"FireEscapePlan|FireButton|Calendar|Mirror|TVDisplay|Magnit|Pushpin|SM_A4|/Cube\d", re.I)
MESH_NAMES = re.compile(
    r"(?<!for_)Wall|Column|Door|Glass|Window|Partition|Stairs|Elevator|Reception|Desk|(?<!Por)Table(?!t)|"
    r"Counter|Chair|Sofa|Armchair|Recliner|Bookcase|Cabinet|Cupboard|Closet|Rack|Shelf|Toilet(?!Paper)|Urinal|"
    r"Washbasin|Personenleitsystem|DrinksMachine|WaterCooler|ElectricRoom", re.I)


def classify(path, names, box):
    """(approximation, reason) for the object at path with the given mesh
    names: approximation is 'none', 'convexHull' or 'off'; reason says why an
    object gets no collision."""
    lo, hi = box.GetMin(), box.GetMax()
    size = sorted(hi[i] - lo[i] for i in range(3))
    if OFF_NAMES.search(path + " " + names):
        return "off", "decor"
    if hi[2] < FLOOR_Z or lo[2] > OVERHEAD_Z:
        return "off", "out_of_reach"
    if size[2] < SMALL:
        return "off", "small"
    arch = bool(MESH_NAMES.search(path.rsplit("/", 1)[-1] + " " + names))
    if size[0] < 0.03 and size[2] < 1.2 and not arch:
        return "off", "flat"
    return ("none" if arch or size[2] > BIG else "convexHull"), None


def _author(prim, approx, material):
    if approx == "off":
        # removing the API (not collisionEnabled = false) keeps PhysX from
        # creating the shapes at all
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        return
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approx)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, "physics")


def _shipped(meshes):
    """Approximation all meshes already collide with, or None."""
    approx = set()
    for m in meshes:
        if not (m.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(m).GetCollisionEnabledAttr().Get()):
            return None
        attr = UsdPhysics.MeshCollisionAPI(m).GetApproximationAttr()
        approx.add(attr.Get() if attr and attr.HasValue() else "none")
    return approx.pop() if len(approx) == 1 else None


def _skinned(prim):
    while prim and not prim.IsPseudoRoot():
        if prim.IsA(UsdSkel.Root):
            return True
        prim = prim.GetParent()
    return False


def add_office_collision(stage, root="/World/Office", material=None):
    """Static collision for the office under root (see the module docstring).
    material: optional UsdShade.Material with physics properties; bound to
    every physics scene of the stage as default material and to the
    overridden colliders. Returns a Counter: objects per approximation
    ('none', 'convexHull', 'off'), 'left_decor' / 'left_out_of_reach'
    objects left as shipped, 'overridden' instances and 'classes' authored."""
    counts = Counter()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    groups = defaultdict(list)       # (prototype, mesh paths, approximation) -> instances
    direct = []                      # (mesh path, approximation) outside instances
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):   # does not enter instances
        if prim.IsInstance():
            proto = prim.GetPrototype()
            meshes = [m for m in Usd.PrimRange(proto) if m.IsA(UsdGeom.Mesh)]
            if not meshes:
                continue
        elif prim.IsA(UsdGeom.Mesh):
            if _skinned(prim) or prim.GetPath().pathString.startswith(root + "/GroundPlane"):
                continue
            meshes = [prim]
        else:
            continue
        approx, reason = classify(prim.GetPath().pathString, " ".join(m.GetName() for m in meshes),
                                  cache.ComputeWorldBound(prim).ComputeAlignedRange())
        if reason in ("decor", "out_of_reach"):
            counts["left_" + reason] += 1
            continue
        counts[approx] += 1
        shipped = _shipped(meshes)
        if shipped == approx or (approx == "off" and not any(m.HasAPI(UsdPhysics.CollisionAPI) for m in meshes)):
            continue                         # already as wanted
        if prim.IsInstance():
            rel = tuple(m.GetPath().MakeRelativePath(proto.GetPath()) for m in meshes)
            groups[(proto.GetPath(), rel, approx)].append(prim.GetPath())
        else:
            direct.append((prim.GetPath(), approx))

    # prototypes are rebuilt as soon as an instance changes, so everything is
    # collected first. The class root is abstract (never simulated); the prims
    # under it are overs so the referencing instances stay defined.
    if material is not None:
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, UsdShade.Tokens.strongerThanDescendants, "physics")
    for path, approx in direct:
        _author(stage.GetPrimAtPath(path), approx, material)
        counts["overridden"] += 1
    layer = stage.GetEditTarget().GetLayer()
    if groups:
        Sdf.CreatePrimInLayer(layer, CLASS_ROOT).specifier = Sdf.SpecifierClass
    refs = []
    for i, ((_, rel, approx), insts) in enumerate(sorted(groups.items())):
        cls = Sdf.Path(CLASS_ROOT).AppendChild(f"C{i:03d}_{approx}")
        for r in rel:
            _author(stage.OverridePrim(cls.AppendPath(r)), approx, material)
        refs.append((cls, insts))
        counts["classes"] += 1
        counts["overridden"] += len(insts)
    with Sdf.ChangeBlock():
        for cls, insts in refs:
            for path in insts:
                Sdf.CreatePrimInLayer(layer, path).referenceList.Prepend(Sdf.Reference(primPath=cls))
    return counts


def office_colliders(stage, root="/World/Office"):
    """{mesh path: approximation} of every enabled collider under root,
    instance proxies included; for checks and reports."""
    out = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
            out[prim.GetPath().pathString] = attr.Get() if attr and attr.HasValue() else prim.GetTypeName()
    return out
