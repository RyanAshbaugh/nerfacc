"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
from typing import Optional, Tuple

import torch
from torch import Tensor

from . import cuda as _C
from .data_specs import RayIntervals, RaySamples


@torch.no_grad()
def ray_aabb_intersect(
    rays_o: Tensor,
    rays_d: Tensor,
    aabbs: Tensor,
    near_plane: float = -float("inf"),
    far_plane: float = float("inf"),
    miss_value: float = float("inf"),
) -> Tuple[Tensor, Tensor, Tensor]:
    """Ray-AABB intersection.

    Args:
        rays_o: (n_rays, 3) Ray origins.
        rays_d: (n_rays, 3) Normalized ray directions.
        aabbs: (m, 6) Axis-aligned bounding boxes {xmin, ymin, zmin, xmax, ymax, zmax}.
        near_plane: Optional. Near plane. Default to -infinity.
        far_plane: Optional. Far plane. Default to infinity.
        miss_value: Optional. Value to use for tmin and tmax when there is no intersection.
            Default to infinity.

    Returns:
        A tuple of {Tensor, Tensor, BoolTensor}:

        - **t_mins**: (n_rays, m) tmin for each ray-AABB pair.
        - **t_maxs**: (n_rays, m) tmax for each ray-AABB pair.
        - **hits**: (n_rays, m) whether each ray-AABB pair intersects.
    """
    assert rays_o.ndim == 2 and rays_o.shape[-1] == 3
    assert rays_d.ndim == 2 and rays_d.shape[-1] == 3
    assert aabbs.ndim == 2 and aabbs.shape[-1] == 6

    if torch.cuda.is_available():
        ray_aabb_intersect_func = _C.ray_aabb_intersect
    else:
        ray_aabb_intersect_func = _ray_aabb_intersect

    t_mins, t_maxs, hits = ray_aabb_intersect_func(
        rays_o.contiguous(),
        rays_d.contiguous(),
        aabbs.contiguous(),
        near_plane,
        far_plane,
        miss_value,
    )
    return t_mins, t_maxs, hits


def _ray_aabb_intersect(
    rays_o: Tensor,
    rays_d: Tensor,
    aabbs: Tensor,
    near_plane: float = -float("inf"),
    far_plane: float = float("inf"),
    miss_value: float = float("inf"),
) -> Tuple[Tensor, Tensor, Tensor]:
    """Ray-AABB intersection.

    Functionally the same with `ray_aabb_intersect()`, but slower with pure Pytorch.
    """

    # Compute the minimum and maximum bounds of the AABBs
    aabb_min = aabbs[:, :3]
    aabb_max = aabbs[:, 3:]

    # Compute the intersection distances between the ray and each of the six AABB planes
    t1 = (aabb_min[None, :, :] - rays_o[:, None, :]) / rays_d[:, None, :]
    t2 = (aabb_max[None, :, :] - rays_o[:, None, :]) / rays_d[:, None, :]

    # Compute the maximum tmin and minimum tmax for each AABB
    t_mins = torch.max(torch.min(t1, t2), dim=-1)[0]
    t_maxs = torch.min(torch.max(t1, t2), dim=-1)[0]

    # Compute whether each ray-AABB pair intersects
    hits = (t_maxs > t_mins) & (t_maxs > 0)

    # Clip the tmin and tmax values to the near and far planes
    t_mins = torch.clamp(t_mins, min=near_plane, max=far_plane)
    t_maxs = torch.clamp(t_maxs, min=near_plane, max=far_plane)

    # Set the tmin and tmax values to miss_value if there is no intersection
    t_mins = torch.where(hits, t_mins, miss_value)
    t_maxs = torch.where(hits, t_maxs, miss_value)

    return t_mins, t_maxs, hits


@torch.no_grad()
def traverse_grids(
    # rays
    rays_o: Tensor,  # [n_rays, 3]
    rays_d: Tensor,  # [n_rays, 3]
    # grids
    binaries: Tensor,  # [m, resx, resy, resz]
    aabbs: Tensor,  # [m, 6]
    # options
    near_planes: Optional[Tensor] = None,  # [n_rays]
    far_planes: Optional[Tensor] = None,  # [n_rays]
    step_size: Optional[float] = 1e-3,
    cone_angle: Optional[float] = 0.0,
    traverse_steps_limit: Optional[int] = None,
    over_allocate: Optional[bool] = False,
    rays_mask: Optional[Tensor] = None,  # [n_rays]
    # pre-compute intersections
    t_sorted: Optional[Tensor] = None,  # [n_rays, n_grids * 2]
    t_indices: Optional[Tensor] = None,  # [n_rays, n_grids * 2]
    hits: Optional[Tensor] = None,  # [n_rays, n_grids]
) -> Tuple[RayIntervals, RaySamples, Tensor]:
    """Ray Traversal within Multiple Grids.

    Note:
        This function is not differentiable to any inputs.

    Args:
        rays_o: (n_rays, 3) Ray origins.
        rays_d: (n_rays, 3) Normalized ray directions.
        binary_grids: (m, resx, resy, resz) Multiple binary grids with the same resolution.
        aabbs: (m, 6) Axis-aligned bounding boxes {xmin, ymin, zmin, xmax, ymax, zmax}.
        near_planes: Optional. (n_rays,) Near planes for the traversal to start. Default to 0.
        far_planes: Optional. (n_rays,) Far planes for the traversal to end. Default to infinity.
        step_size: Optional. Step size for ray traversal. Default to 1e-3.
        cone_angle: Optional. Cone angle for linearly-increased step size. 0. means
            constant step size. Default: 0.0.
        traverse_steps_limit: Optional. Maximum number of samples per ray.
        over_allocate: Optional. Whether to over-allocate the memory for the outputs.
        rays_mask: Optional. (n_rays,) Skip some rays if given.
        t_sorted: Optional. (n_rays, n_grids * 2) Pre-computed sorted t values for each ray-grid pair. Default to None.
        t_indices: Optional. (n_rays, n_grids * 2) Pre-computed sorted t indices for each ray-grid pair. Default to None.
        hits: Optional. (n_rays, n_grids) Pre-computed hit flags for each ray-grid pair. Default to None.

    Returns:
        A :class:`RayIntervals` object containing the intervals of the ray traversal, and
        a :class:`RaySamples` object containing the samples within each interval.
        t :class:`Tensor` of shape (n_rays,) containing the terminated t values for each ray.
    """

    if near_planes is None:
        near_planes = torch.zeros_like(rays_o[:, 0])
    if far_planes is None:
        far_planes = torch.full_like(rays_o[:, 0], float("inf"))

    if rays_mask is None:
        rays_mask = torch.ones_like(rays_o[:, 0], dtype=torch.bool)
    if traverse_steps_limit is None:
        traverse_steps_limit = -1
    if over_allocate:
        assert (
            traverse_steps_limit > 0
        ), "traverse_steps_limit must be set if over_allocate is True."

    if t_sorted is None or t_indices is None or hits is None:
        # Compute ray aabb intersection for all levels of grid. [n_rays, m]
        t_mins, t_maxs, hits = ray_aabb_intersect(rays_o, rays_d, aabbs)
        # Sort the t values for each ray. [n_rays, m]
        t_sorted, t_indices = torch.sort(
            torch.cat([t_mins, t_maxs], dim=-1), dim=-1
        )

    if torch.cuda.is_available():
        traverse_grids_func = _C.traverse_grids
    else:
        traverse_grids_func = _traverse_grids

    # Traverse the grids.
    intervals, samples, termination_planes = traverse_grids_func(
        # rays
        rays_o.contiguous(),  # [n_rays, 3]
        rays_d.contiguous(),  # [n_rays, 3]
        rays_mask.contiguous(),  # [n_rays]
        # grids
        binaries.contiguous(),  # [m, resx, resy, resz]
        aabbs.contiguous(),  # [m, 6]
        # intersections
        t_sorted.contiguous(),  # [n_rays, m * 2]
        t_indices.contiguous(),  # [n_rays, m * 2]
        hits.contiguous(),  # [n_rays, m]
        # options
        near_planes.contiguous(),  # [n_rays]
        far_planes.contiguous(),  # [n_rays]
        step_size,
        cone_angle,
        True,
        True,
        True,
        traverse_steps_limit,
        over_allocate,
    )

    intervals_cpu, samples_cpu, termination_planes = _traverse_grids(
        # rays
        rays_o.contiguous().cpu(),  # [n_rays, 3]
        rays_d.contiguous().cpu(),  # [n_rays, 3]
        rays_mask.contiguous().cpu(),  # [n_rays]
        # grids
        binaries.contiguous().cpu(),  # [m, resx, resy, resz]
        aabbs.contiguous().cpu(),  # [m, 6]
        # intersections
        t_sorted.contiguous().cpu(),  # [n_rays, m * 2]
        t_indices.contiguous().cpu(),  # [n_rays, m * 2]
        hits.contiguous().cpu(),  # [n_rays, m]
        # options
        near_planes.contiguous().cpu(),  # [n_rays]
        far_planes.contiguous().cpu(),  # [n_rays]
        step_size,
        cone_angle,
        True,
        True,
        True,
        traverse_steps_limit,
        over_allocate,
    )

    if traverse_grids_func is _C.traverse_grids:
        intervals = RayIntervals._from_cpp(intervals)
        samples = RaySamples._from_cpp(samples)

    return (
        intervals,
        samples,
        termination_planes,
    )

def _traverse_grids(
    rays_o,         # [n_rays, 3]
    rays_d,         # [n_rays, 3]
    rays_mask,      # [n_rays]
    binaries,       # [n_grids, resx, resy, resz]
    aabbs,          # [n_grids, 6]
    t_sorted,       # [n_rays, n_grids * 2]
    t_indices,      # [n_rays, n_grids * 2]
    hits,           # [n_rays, n_grids]
    near_planes,    # [n_rays]
    far_planes,     # [n_rays]
    step_size,
    cone_angle,
    compute_intervals,
    compute_samples,
    compute_terminate_planes,
    traverse_steps_limit, # <= 0 means no limit
    over_allocate,
):
    rays_o, rays_d, rays_mask, binaries, aabbs, t_sorted, t_indices, hits, near_planes, far_planes = [
        xx.cpu() for xx in [rays_o, rays_d, rays_mask, binaries, aabbs, t_sorted, t_indices, hits, near_planes, far_planes]
    ]

    # Initialize output tensors
    n_rays = rays_o.shape[0]
    n_grids = binaries.shape[0]
    resx, resy, resz = binaries.shape[1:4]

    all_intervals = []
    all_samples = []
    all_is_left = []
    all_is_right = []
    all_ray_indices = []

    packed_info_intervals = []
    packed_info_samples = []

    terminate_planes = torch.zeros(n_rays, dtype=torch.float32) if compute_terminate_planes else None

    packed_info_intervals = torch.zeros((n_rays, 2), dtype=torch.long)
    packed_info_samples = torch.zeros((n_rays, 2), dtype=torch.long)

    for ray_idx in range(n_rays):
        # if rays_mask is not None and not rays_mask[ray_idx]:
        #     continue

        ray_origin = rays_o[ray_idx]
        ray_dir = rays_d[ray_idx]
        near_plane = near_planes[ray_idx]
        far_plane = far_planes[ray_idx]

        t_last = near_plane
        n_intervals = 0
        n_samples = 0
        continuous = False
        ray_intervals = []
        ray_samples = []
        ray_is_left = []
        ray_is_right = []

        for grid_idx in range(n_grids):
            # if not hits[ray_idx, grid_idx]:
            #     continue

            aabb_min = aabbs[grid_idx, :3]
            aabb_max = aabbs[grid_idx, 3:]

            t_min = max(near_plane, torch.dot((aabb_min - ray_origin), ray_dir))
            t_max = min(far_plane, torch.dot((aabb_max - ray_origin), ray_dir))
            if t_min >= t_max:
                continue

            while t_last < t_max:
                t_next = t_last + step_size
                if t_next > t_max:
                    t_next = t_max

                mid_t = (t_last + t_next) / 2
                voxel_idx = ((mid_t - aabb_min) / (aabb_max - aabb_min) * torch.tensor([resx, resy, resz])).long()

                if 0 <= voxel_idx[0] < resx and 0 <= voxel_idx[1] < resy and 0 <= voxel_idx[2] < resz:
                    binary_idx = grid_idx * resx * resy * resz + voxel_idx[0] * resy * resz + voxel_idx[1] * resz + voxel_idx[2]
                    if binaries.view(-1)[binary_idx]:
                        # Record interval
                        ray_intervals.append(t_last)
                        ray_intervals.append(t_next)
                        ray_is_left.append(True)
                        ray_is_right.append(False)

                        # Record sample
                        ray_samples.append(mid_t)

                        n_intervals += 1
                        n_samples += 1

                t_last = t_next

        # Store packed information for this ray
        if ray_intervals:
            packed_info_intervals[ray_idx] = torch.tensor([len(all_intervals), len(ray_intervals)])
            all_intervals.extend(ray_intervals)
            all_is_left.extend(ray_is_left)
            all_is_right.extend(ray_is_right)
            all_ray_indices.extend([ray_idx] * len(ray_intervals))

        if ray_samples:
            packed_info_samples[ray_idx] = torch.tensor([len(all_samples), len(ray_samples)])
            all_samples.extend(ray_samples)
            all_ray_indices.extend([ray_idx] * len(ray_samples))

        # if compute_terminate_planes:
        #     terminate_planes[ray_idx] = t_last

    # Convert results to tensors
    intervals = RayIntervals(
        vals=torch.tensor(all_intervals, dtype=torch.float32),
        packed_info=packed_info_intervals,
        ray_indices=torch.tensor(all_ray_indices, dtype=torch.long),
        is_left=torch.tensor(all_is_left, dtype=torch.bool),
        is_right=torch.tensor(all_is_right, dtype=torch.bool),
    )
    samples = RaySamples(
        vals=torch.tensor(all_samples, dtype=torch.float32),
        packed_info=packed_info_samples,
        ray_indices=torch.tensor(all_ray_indices, dtype=torch.long),
    )

    return intervals, samples, terminate_planes

def _enlarge_aabb(aabb, factor: float) -> Tensor:
    center = (aabb[:3] + aabb[3:]) / 2
    extent = (aabb[3:] - aabb[:3]) / 2
    return torch.cat([center - extent * factor, center + extent * factor])


def _query(x: Tensor, data: Tensor, base_aabb: Tensor) -> Tensor:
    """
    Query the grid values at the given points.

    This function assumes the aabbs of multiple grids are 2x scaled.

    Args:
        x: (N, 3) tensor of points to query.
        data: (m, resx, resy, resz) tensor of grid values
        base_aabb: (6,) aabb of base level grid.
    """
    # normalize so that the base_aabb is [0, 1]^3
    aabb_min, aabb_max = torch.split(base_aabb, 3, dim=0)
    x_norm = (x - aabb_min) / (aabb_max - aabb_min)

    # if maxval is almost zero, it will trigger frexpf to output 0
    # for exponent, which is not what we want.
    maxval = (x_norm - 0.5).abs().max(dim=-1).values
    maxval = torch.clamp(maxval, min=0.1)

    # compute the mip level
    exponent = torch.frexp(maxval)[1].long()
    mip = torch.clamp(exponent + 1, min=0)
    selector = mip < data.shape[0]

    # use the mip to re-normalize all points to [0, 1].
    scale = 2**mip
    x_unit = (x_norm - 0.5) / scale[:, None] + 0.5

    # map to the grid index
    resolution = torch.tensor(data.shape[1:], device=x.device)
    ix = (x_unit * resolution).long()

    ix = torch.clamp(ix, max=resolution - 1)
    mip = torch.clamp(mip, max=data.shape[0] - 1)

    return data[mip, ix[:, 0], ix[:, 1], ix[:, 2]] * selector, selector
