"""The five-term smooth-highlight recipe, with effective-batch denominators."""
from __future__ import annotations

import math

import torch

LOSS_COEFFICIENTS = dict(HDR=1., bright=.25, grad=.10, smooth=.05, over=1.)
EFFECTIVE_BATCH = 8


def luminance(rgb):
    return rgb[:, :1] * .2126 + rgb[:, 1:2] * .7152 + rgb[:, 2:3] * .0722


def tensors_for_sample(sample, device):
    return {key: torch.from_numpy(sample[key][None]).to(device=device)
            for key in ("s_srgb", "base", "target", "valid")}


def edge_terms(base, target, valid):
    ls, lt = torch.log1p(luminance(base)), torch.log1p(luminance(target))
    for axis in (2, 3):
        first = [slice(None)] * 4
        second = first.copy()
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        mask = valid[first] & valid[second]
        weight = torch.exp(-10 * torch.maximum((ls[first] - ls[second]).abs(), (lt[first] - lt[second]).abs()))
        yield first, second, mask, weight


def loss_denominators(samples, device):
    if len(samples) != EFFECTIVE_BATCH:
        raise ValueError("The recipe uses microbatch 1, accumulation 8")
    result = dict(valid_pixels=0, bright_weight_sum=0., grad_RGB_edges=0, smooth_edge_weight_sum=0.)
    with torch.no_grad():
        for sample in samples:
            tensors = tensors_for_sample(sample, device)
            valid, target = tensors["valid"], tensors["target"]
            y = luminance(target)
            bright = (y / (1 + y)).square()
            result["valid_pixels"] += int(valid.sum().item())
            result["bright_weight_sum"] += float(torch.where(valid, bright, 0.).sum(dtype=torch.float64).item())
            for _, _, mask, weight in edge_terms(tensors["base"], target, valid):
                result["grad_RGB_edges"] += 3 * int(mask.sum().item())
                result["smooth_edge_weight_sum"] += float(torch.where(mask, weight, 0.).sum(dtype=torch.float64).item())
    return result


def smooth_loss(result, tensors, denominators):
    """One microbatch's contribution to the globally normalized eight-view loss."""
    pred, target, valid = result["H_new"], tensors["target"], tensors["valid"]
    if pred.dtype != torch.float32 or target.dtype != torch.float32 or pred.shape != target.shape:
        raise ValueError("Loss expects matching float32 NCHW RGB")
    if not bool(torch.isfinite(pred).all()) or not bool((pred >= 0).all()):
        raise ValueError("Prediction must be finite nonnegative HDR")
    lp, lt = torch.log1p(pred), torch.log1p(target)
    error = lp - lt
    zero = result["z"].sum() * 0.
    terms = {key: zero for key in LOSS_COEFFICIENTS}
    undefined = {}
    selected = valid.expand_as(error)
    if denominators["valid_pixels"]:
        terms["HDR"] = torch.where(selected, error.abs(), 0.).sum() / (3 * denominators["valid_pixels"])
        terms["over"] = torch.where(selected, error.relu(), 0.).sum() / (3 * denominators["valid_pixels"])
    else:
        undefined.update(HDR="empty_valid_support", over="empty_valid_support")
    y = luminance(target)
    bright = (y / (1 + y)).square()
    if denominators["bright_weight_sum"] > 0:
        terms["bright"] = torch.where(selected, bright * error.abs(), 0.).sum() / (3 * denominators["bright_weight_sum"])
    else:
        undefined["bright"] = "zero_reference_bright_weight"
    grad, smooth = zero, zero
    for first, second, mask, weight in edge_terms(tensors["base"], target, valid):
        grad = grad + torch.where(mask.expand_as(error[first]),
                                  ((lp[first] - lp[second]) - (lt[first] - lt[second])).abs(), 0.).sum()
        smooth = smooth + torch.where(mask, weight * (result["delta"][first] - result["delta"][second]).abs(), 0.).sum()
    if denominators["grad_RGB_edges"]:
        terms["grad"] = grad / denominators["grad_RGB_edges"]
    else:
        undefined["grad"] = "empty_valid_edge_support"
    if denominators["smooth_edge_weight_sum"] > 0:
        terms["smooth"] = smooth / denominators["smooth_edge_weight_sum"]
    else:
        undefined["smooth"] = "zero_valid_edge_weight"
    total = sum(LOSS_COEFFICIENTS[key] * value for key, value in terms.items())
    if not bool(torch.isfinite(total)) or float(total.detach()) < 0:
        raise ValueError("Nonfinite or negative training loss")
    values = {key: None if key in undefined else float(value.detach()) for key, value in terms.items()}
    return total, values, undefined


def optimizer_for(head):
    modules = dict(head.named_modules())
    decay, no_decay = [], []
    for name, parameter in head.named_parameters():
        if not parameter.requires_grad:
            raise ValueError("Every head parameter must be trainable")
        parent, _, field = name.rpartition(".")
        group = decay if isinstance(modules[parent], torch.nn.Conv2d) and field == "weight" else no_decay
        group.append(parameter)
    if sum(parameter.numel() for parameter in decay + no_decay) != 2769:
        raise ValueError("Expected the 2,769-parameter smooth highlight head")
    return torch.optim.AdamW([dict(params=decay, weight_decay=.01), dict(params=no_decay, weight_decay=0.)],
                             lr=1e-4, betas=(.9, .999), eps=1e-8, foreach=False, fused=False)


class FixedScheduler:
    """128-update linear warmup, then cosine 1e-4 -> 1e-6 through update 2048."""
    def __init__(self, optimizer, total_updates=2048, warmup_updates=128):
        if total_updates < 1 or warmup_updates < 1:
            raise ValueError("Scheduler update counts must be positive")
        self.optimizer, self.total_updates, self.warmup_updates = optimizer, total_updates, warmup_updates
        self.completed_update = 0
        self._apply(self.learning_rate(1))

    def learning_rate(self, update):
        if not 1 <= update <= self.total_updates:
            raise ValueError("Scheduler update is outside the configured range")
        if update <= self.warmup_updates:
            return 1e-4 * update / self.warmup_updates
        return 1e-6 + .5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * (update - self.warmup_updates) / (self.total_updates - self.warmup_updates)))

    def _apply(self, value):
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def begin(self, update):
        if update != self.completed_update + 1:
            raise ValueError("Updates must be contiguous")
        value = self.learning_rate(update)
        self._apply(value)
        return value

    def complete(self, update):
        if update != self.completed_update + 1:
            raise ValueError("Updates must be contiguous")
        self.completed_update = update
        self._apply(self.learning_rate(min(update + 1, self.total_updates)))

    def state_dict(self):
        return dict(total_updates=self.total_updates, warmup_updates=self.warmup_updates,
                    completed_update=self.completed_update, next_lr=self.optimizer.param_groups[0]["lr"])

    def load_state_dict(self, state):
        if state["total_updates"] != self.total_updates or state["warmup_updates"] != self.warmup_updates:
            raise ValueError("Checkpoint scheduler configuration differs")
        if not 0 <= state["completed_update"] <= self.total_updates:
            raise ValueError("Checkpoint scheduler position is invalid")
        self.completed_update = state["completed_update"]
        self._apply(self.learning_rate(min(self.completed_update + 1, self.total_updates)))
        if self.optimizer.param_groups[0]["lr"] != state["next_lr"]:
            raise ValueError("Checkpoint scheduler learning rate differs")


def update_once(model, optimizer, scheduler, update, samples, device):
    """One optimizer update; frozen VAE/bridge, trainable head, eight microbatches."""
    denominators = loss_denominators(samples, device)
    if not denominators["valid_pixels"]:
        raise ValueError("Cannot train an effective batch with zero valid pixels")
    optimizer.zero_grad(set_to_none=True)
    learning_rate = scheduler.begin(update)
    values = {key: [] for key in LOSS_COEFFICIENTS}
    total = 0.
    for sample in samples:
        tensors = tensors_for_sample(sample, device)
        result = model.forward_tensors(tensors["s_srgb"], tensors["base"])
        loss, metrics, _ = smooth_loss(result, tensors, denominators)
        loss.backward()
        total += float(loss.detach())
        for key, value in metrics.items():
            if value is not None:
                values[key].append(value)
    for module in (model.vae, model.bridge):
        if any(parameter.requires_grad or parameter.grad is not None for parameter in module.parameters()):
            raise RuntimeError("VAE and bridge must remain frozen without gradients")
    for name, parameter in model.head.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"Missing or nonfinite head gradient: {name}")
    norms = {name: float(torch.linalg.vector_norm(parameter.grad)) for name, parameter in model.head.named_parameters()}
    norm = float(torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1., foreach=False))
    if not math.isfinite(norm):
        raise RuntimeError("Nonfinite gradient norm")
    optimizer.step()
    scheduler.complete(update)
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.head.parameters()):
        raise RuntimeError("Nonfinite head after optimizer update")
    optimizer.zero_grad(set_to_none=True)
    return dict(update=update, loss=total, terms={key: sum(items) if items else None for key, items in values.items()},
                denominators=denominators, learning_rate=learning_rate, gradient_norm=norm,
                parameter_gradient_norms=norms, samples=8, hard_clip=4, soft_compress=4)
