"""Reference-paired errors in relative linear HDR units, with explicit support."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from gainweave.training.data import luminance, validate_mask, validate_rgb

METRICS = ("MAE", "MSE", "signed", "over", "under", "gradient")
DEFINITION = dict(
    domain="float64 log1p of nonnegative relative linear RGB; no peak/nit conversion or scale fitting",
    color="RGB channels averaged equally; luminance=.2126R+.7152G+.0722B in float32",
    signed="mean(log1p(prediction)-log1p(reference))",
    over="mean(max(signed pixel/channel error,0)); positive magnitude",
    under="mean(max(-signed pixel/channel error,0)); positive magnitude",
    continuous_bright="b=(Y_reference/(1+Y_reference))^2; sum(b*RGB error)/(3*sum(b))",
    gradient="mean absolute difference of adjacent log1p RGB gradients, both endpoints in support",
    bright_gradient="same adjacent gradient error weighted by mean endpoint reference b",
    masks="0=outside,1=inside,255=ignore; ignore is never foreground",
    aggregation="mean defined views/EVs within source, then equal mean of defined sources; retain empty sources",
    empty="null/None for zero support; no zero imputation",
)


def _weighted_metrics(error, domain, weights=None):
    weights = domain.astype(np.float64) if weights is None else np.where(domain, weights, 0.).astype(np.float64)
    weight_sum = float(weights.sum(dtype=np.float64))
    denominator = 3 * weight_sum
    result = dict(support_pixels=int(domain.sum()), weight_sum=weight_sum, RGB_support=denominator,
                  gradient_edges=0, gradient_weight_sum=0.)
    values = dict(MAE=np.abs(error), MSE=error * error, signed=error,
                  over=np.maximum(error, 0), under=np.maximum(-error, 0))
    for name, value in values.items():
        result[name] = float((value * weights[..., None]).sum(dtype=np.float64) / denominator) if denominator else None
    gradient_sum, edge_weights = 0., 0.
    for axis in (0, 1):
        first = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        second = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        valid_edge = domain[first] & domain[second]
        # Full/region weights are 1; continuous-bright weights are endpoint means.
        edge_weight = np.where(valid_edge, .5 * (weights[first] + weights[second]), 0.)
        difference = np.abs(error[second] - error[first])
        gradient_sum += float((difference * edge_weight[..., None]).sum(dtype=np.float64))
        edge_weights += float(edge_weight.sum(dtype=np.float64))
        result["gradient_edges"] += int(valid_edge.sum())
    result["gradient_weight_sum"] = edge_weights
    result["gradient"] = gradient_sum / (3 * edge_weights) if edge_weights else None
    return result


def evaluate_case(prediction, reference, valid=None, region=None):
    """Score matching HWC RGB arrays without resizing or per-image alignment.

    ``valid`` and ``region`` accept bool or 0/1/255 arrays. Region support is
    intersected with valid support. Empty support remains explicit, including
    all-black references whose continuous bright weight is zero.
    """
    prediction, reference = validate_rgb(prediction), validate_rgb(reference)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction/reference geometry differs; no implicit resize is applied")
    shape = reference.shape[:2]
    valid = np.ones(shape, dtype=np.bool_) if valid is None else validate_mask(valid, shape)
    error = np.log1p(prediction.astype(np.float64)) - np.log1p(reference.astype(np.float64))
    y = luminance(reference)
    bright = np.square(y / (np.float32(1) + y)).astype(np.float64)
    result = dict(full=_weighted_metrics(error, valid), continuous_bright=_weighted_metrics(error, valid, bright))
    if region is not None:
        selected = valid & validate_mask(region, shape)
        result["region"] = _weighted_metrics(error, selected)
        result["region_continuous_bright"] = _weighted_metrics(error, selected, bright)
    return result


def aggregate_cases(cases):
    """Source macro summaries for each EV and ALL; undefined views stay counted."""
    groups = defaultdict(list)
    for case in cases:
        for region, values in case["regions"].items():
            for exposure in (str(case["EV"]), "ALL"):
                groups[(case["method"], region, exposure)].append((case["source_id"], values))
    aggregates = []
    for (method, region, exposure), records in sorted(groups.items()):
        sources = defaultdict(list)
        for source_id, values in records:
            sources[source_id].append(values)
        source_rows = []
        for source_id, views in sorted(sources.items()):
            row = dict(source_id=source_id, total_views=len(views),
                       support_pixels=sum(view["support_pixels"] for view in views), metrics={}, defined_views={})
            for name in METRICS:
                defined = [view[name] for view in views if view[name] is not None]
                row["metrics"][name] = float(np.mean(defined, dtype=np.float64)) if defined else None
                row["defined_views"][name] = len(defined)
            source_rows.append(row)
        metrics, defined_sources = {}, {}
        for name in METRICS:
            defined = [row["metrics"][name] for row in source_rows if row["metrics"][name] is not None]
            metrics[name] = float(np.mean(defined, dtype=np.float64)) if defined else None
            defined_sources[name] = len(defined)
        aggregates.append(dict(method=method, region=region, EV=exposure, total_views=len(records),
                               total_sources=len(source_rows), equal_source=metrics,
                               defined_sources=defined_sources, sources=source_rows))
    return aggregates
