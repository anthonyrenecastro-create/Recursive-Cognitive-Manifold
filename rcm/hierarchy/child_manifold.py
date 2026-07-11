from rcm.hierarchy.abstraction import abstract_regions


def refresh_child_manifolds(manifold):
    stable_regions = [
        signature
        for signature, region in manifold.regions.items()
        if len(signature) > 1 and region.stability > 0.85
    ]
    if stable_regions:
        abstracted = abstract_regions(manifold, stable_regions)
        if manifold.child_manifolds:
            manifold.child_manifolds[0] = abstracted
        else:
            manifold.child_manifolds.append(abstracted)
    else:
        manifold.child_manifolds.clear()