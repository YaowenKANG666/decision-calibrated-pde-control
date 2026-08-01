"""Notebook-friendly uncertainty experiment on the official NS2D dataset.

This benchmark validates function-valued uncertainty calibration on 2D
Navier--Stokes data.  It is not presented as a controlled-flow experiment:
the public NeuralOperator dataset contains input/output vorticity fields but
no action channel.  Closed-loop control remains evaluated on controlled
Burgers until an action-conditioned 2D dataset is generated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def higher_quantile(values: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((len(values) + 1) * coverage)) - 1
    if rank >= len(values):
        return float("inf")
    return float(values[max(rank, 0)])


def _batch_loss(model, batch, device):
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    prediction = model(x)
    return torch.mean((prediction - y).square())


def train_operator(model, loader, validation_loader, device, epochs, learning_rate=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                validation_losses.append(float(_batch_loss(model, batch, device).cpu()))
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_mse": float(np.mean(train_losses)),
            "validation_mse": float(np.mean(validation_losses)),
        }
        history.append(row)
        print(
            f"epoch={epoch + 1:03d} train_mse={row['train_mse']:.6e} "
            f"validation_mse={row['validation_mse']:.6e}"
        )
    return history


@torch.no_grad()
def predict_with_disagreement(base, perturbed, x, smoothing_window, floor):
    mean = base(x)
    second = perturbed(x)
    disagreement = torch.abs(mean - second)
    if smoothing_window > 1:
        padding = smoothing_window // 2
        disagreement = F.avg_pool2d(
            F.pad(disagreement, (padding, padding, padding, padding), mode="replicate"),
            kernel_size=smoothing_window,
            stride=1,
        )
    return mean, disagreement.clamp_min(floor)


@torch.no_grad()
def estimate_floor(base, perturbed, loader, device, smoothing_window):
    values = []
    for batch in loader:
        x = batch["x"].to(device)
        mean = base(x)
        second = perturbed(x)
        disagreement = torch.abs(mean - second)
        if smoothing_window > 1:
            padding = smoothing_window // 2
            disagreement = F.avg_pool2d(
                F.pad(
                    disagreement,
                    (padding, padding, padding, padding),
                    mode="replicate",
                ),
                kernel_size=smoothing_window,
                stride=1,
            )
        values.append(disagreement.flatten().cpu())
    return max(1e-7, 0.1 * float(torch.median(torch.cat(values))))


@torch.no_grad()
def collect_scores(base, perturbed, loader, device, smoothing_window, floor):
    scores, field_rmses, widths, scale_error_pairs = [], [], [], []
    examples = None
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mean, scale = predict_with_disagreement(
            base, perturbed, x, smoothing_window, floor
        )
        residual = y - mean
        score = (residual.abs() / scale).flatten(1).amax(dim=1)
        scores.append(score.cpu().numpy())
        field_rmses.append(
            torch.sqrt(torch.mean(residual.square(), dim=(1, 2, 3))).cpu().numpy()
        )
        widths.append(scale.mean(dim=(1, 2, 3)).cpu().numpy())
        # Bounded subsample for a readable error-scale scatter plot.
        flat_scale = scale.flatten().cpu().numpy()
        flat_error = residual.abs().flatten().cpu().numpy()
        stride = max(1, flat_scale.size // 5000)
        scale_error_pairs.append(
            np.column_stack((flat_scale[::stride], flat_error[::stride]))[:5000]
        )
        if examples is None:
            examples = {
                "x": x[0, 0].cpu().numpy(),
                "y": y[0, 0].cpu().numpy(),
                "mean": mean[0, 0].cpu().numpy(),
                "scale": scale[0, 0].cpu().numpy(),
                "error": residual[0, 0].abs().cpu().numpy(),
            }
    return {
        "scores": np.concatenate(scores),
        "field_rmses": np.concatenate(field_rmses),
        "mean_scales": np.concatenate(widths),
        "scale_error_pairs": np.concatenate(scale_error_pairs),
        "example": examples,
    }


def _style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure(figure, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def field_figure(
    field,
    title,
    stem,
    cmap="RdBu_r",
    nonnegative=False,
    bound=None,
    upper=None,
):
    figure, axis = plt.subplots(figsize=(3.45, 3.0), constrained_layout=True)
    if nonnegative:
        image = axis.imshow(field, origin="lower", cmap=cmap, vmin=0.0, vmax=upper)
    else:
        bound = float(np.max(np.abs(field))) if bound is None else float(bound)
        image = axis.imshow(field, origin="lower", cmap=cmap, vmin=-bound, vmax=bound)
    axis.set_xlabel("Grid coordinate $x_1$")
    axis.set_ylabel("Grid coordinate $x_2$")
    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.048, pad=0.03)
    colorbar.ax.tick_params(labelsize=6.5)
    save_figure(figure, stem)


def make_figures(history, audit, test, q90, output_dir):
    _style()
    example = test["example"]
    target_prediction_bound = float(
        max(np.max(np.abs(example["y"])), np.max(np.abs(example["mean"])))
    )
    field_figure(example["x"], "Input vorticity", output_dir / "ns2d_01_input")
    field_figure(
        example["y"],
        "Target vorticity",
        output_dir / "ns2d_02_target",
        bound=target_prediction_bound,
    )
    field_figure(
        example["mean"],
        "FNO prediction",
        output_dir / "ns2d_03_prediction",
        bound=target_prediction_bound,
    )
    field_figure(
        example["error"],
        "Absolute prediction error",
        output_dir / "ns2d_04_absolute_error",
        cmap="magma",
        nonnegative=True,
    )
    field_figure(
        q90 * example["scale"],
        "Calibrated uncertainty half-width",
        output_dir / "ns2d_05_uncertainty_halfwidth",
        cmap="viridis",
        nonnegative=True,
    )
    pointwise_utilization = example["error"] / np.maximum(
        q90 * example["scale"], 1e-12
    )
    field_figure(
        pointwise_utilization,
        "Pointwise band utilization",
        output_dir / "ns2d_06_band_utilization",
        cmap="magma",
        nonnegative=True,
        upper=max(1.0, float(np.max(pointwise_utilization))),
    )

    figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    axis.semilogy(epochs, [row["train_mse"] for row in history], color="#3A5A78", label="Train")
    axis.semilogy(
        epochs,
        [row["validation_mse"] for row in history],
        color="#D8843F",
        label="Validation",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.set_title("Base FNO optimization")
    axis.legend()
    axis.grid(alpha=0.18)
    save_figure(figure, output_dir / "ns2d_07_training_curve")

    figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    sorted_scores = np.sort(test["scores"])
    empirical = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axis.plot(sorted_scores, empirical, color="#3A5A78", lw=1.6)
    axis.axvline(q90, color="#D8843F", ls="--", lw=1.2, label="90% audit quantile")
    axis.set_xscale("log")
    axis.set_xlabel("Max standardized field error")
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_title("Simultaneous conformal score distribution")
    axis.legend()
    axis.grid(alpha=0.18)
    save_figure(figure, output_dir / "ns2d_08_score_ecdf")

    coverage_levels = np.asarray([0.80, 0.85, 0.90, 0.925, 0.95, 0.975])
    qs = np.asarray([higher_quantile(audit["scores"], level) for level in coverage_levels])
    empirical_coverage = np.asarray([np.mean(test["scores"] <= q) for q in qs])
    mean_width = 2.0 * qs * float(np.mean(test["mean_scales"]))

    figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    axis.plot(coverage_levels, empirical_coverage, "o-", color="#3A5A78", lw=1.5, ms=4)
    axis.plot([0.78, 0.99], [0.78, 0.99], "--", color="#777777", lw=1)
    axis.set_xlim(0.78, 0.99)
    axis.set_ylim(0.78, 1.005)
    axis.set_xlabel("Nominal simultaneous coverage")
    axis.set_ylabel("Test simultaneous coverage")
    axis.set_title("Coverage reliability")
    axis.grid(alpha=0.18)
    save_figure(figure, output_dir / "ns2d_09_coverage_reliability")

    figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    axis.plot(mean_width, empirical_coverage, "o-", color="#D8843F", lw=1.5, ms=4)
    for width, cov, level in zip(mean_width, empirical_coverage, coverage_levels):
        axis.annotate(
            f"{level:.3g}",
            (width, cov),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=6,
        )
    axis.set_xlabel("Mean full band width")
    axis.set_ylabel("Test simultaneous coverage")
    axis.set_title("Coverage–width trade-off")
    axis.grid(alpha=0.18)
    save_figure(figure, output_dir / "ns2d_10_coverage_width")

    pairs = test["scale_error_pairs"]
    figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    axis.hexbin(
        pairs[:, 0],
        pairs[:, 1],
        gridsize=45,
        bins="log",
        mincnt=1,
        cmap="Blues",
    )
    axis.set_xlabel("Uncalibrated disagreement scale")
    axis.set_ylabel("Absolute pointwise error")
    scale_error_pearson = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
    axis.set_title(f"Error localization by disagreement (Pearson r={scale_error_pearson:.2f})")
    save_figure(figure, output_dir / "ns2d_11_error_scale")

    return [
        {
            "nominal_coverage": float(level),
            "test_coverage": float(cov),
            "quantile": float(q),
            "mean_full_width": float(width),
        }
        for level, cov, q, width in zip(coverage_levels, empirical_coverage, qs, mean_width)
    ]


def run(args):
    try:
        from neuralop.data.datasets import NavierStokesDataset
        from neuralop.data.datasets.tensor_dataset import TensorDataset
        from neuralop.models import FNO
    except ImportError as error:
        raise RuntimeError(
            "Install the official package first: pip install neuraloperator"
        ) from error

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_count = min(max(32, args.n_audit // 4), max(32, args.n_train // 5))
    required_training_examples = args.n_train + validation_count + args.n_audit
    dataset = NavierStokesDataset(
        root_dir=args.data_root,
        n_train=required_training_examples,
        n_tests=[args.n_test],
        batch_size=args.batch_size,
        test_batch_sizes=[args.batch_size],
        train_resolution=128,
        test_resolutions=[128],
        encode_input=False,
        encode_output=False,
        download=not args.no_download,
    )
    proper_indices = list(range(args.n_train))
    validation_indices = list(
        range(args.n_train, args.n_train + validation_count)
    )
    audit_indices = list(
        range(
            args.n_train + validation_count,
            args.n_train + validation_count + args.n_audit,
        )
    )
    train_db = Subset(dataset.train_db, proper_indices)
    validation_db = Subset(dataset.train_db, validation_indices)
    audit_db = Subset(dataset.train_db, audit_indices)
    test_db = dataset.test_dbs[128]

    train_loader = DataLoader(train_db, batch_size=args.batch_size, shuffle=True)
    audit_loader = DataLoader(audit_db, batch_size=args.batch_size, shuffle=False)
    validation_loader = DataLoader(validation_db, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_db, batch_size=args.batch_size, shuffle=False)

    def make_model():
        return FNO(
            n_modes=(args.modes, args.modes),
            hidden_channels=args.hidden_channels,
            in_channels=1,
            out_channels=1,
            n_layers=4,
        ).to(device)

    base = make_model()
    base_history = train_operator(
        base, train_loader, validation_loader, device, args.epochs
    )

    full_train = dataset.train_db
    train_x = full_train.x[proper_indices]
    train_y = full_train.y[proper_indices]
    generator = torch.Generator().manual_seed(args.seed + 1)
    noise_std = args.label_noise * float(train_y.std())
    perturbed_y = train_y + noise_std * torch.randn(
        train_y.shape, generator=generator, dtype=train_y.dtype
    )
    perturbed_db = TensorDataset(train_x, perturbed_y)
    perturbed_loader = DataLoader(
        perturbed_db, batch_size=args.batch_size, shuffle=True
    )
    perturbed = make_model()
    perturbed_history = train_operator(
        perturbed, perturbed_loader, validation_loader, device, args.epochs
    )

    floor = estimate_floor(
        base, perturbed, train_loader, device, args.smoothing_window
    )
    audit = collect_scores(
        base,
        perturbed,
        audit_loader,
        device,
        args.smoothing_window,
        floor,
    )
    test = collect_scores(
        base,
        perturbed,
        test_loader,
        device,
        args.smoothing_window,
        floor,
    )
    q90 = higher_quantile(audit["scores"], 0.90)
    output = args.output_dir
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(parents=True, exist_ok=True)
    curves = make_figures(
        base_history,
        audit,
        test,
        q90,
        output / "figures",
    )
    with (output / "data" / "coverage_width.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    with (output / "data" / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(base_history[0]))
        writer.writeheader()
        writer.writerows(base_history)
    metrics = {
        "device": str(device),
        "n_train": args.n_train,
        "n_validation": validation_count,
        "n_audit": args.n_audit,
        "n_test": args.n_test,
        "resolution": 128,
        "label_noise": args.label_noise,
        "smoothing_window": args.smoothing_window,
        "floor": floor,
        "q90": q90,
        "test_simultaneous_coverage": float(np.mean(test["scores"] <= q90)),
        "mean_rmse": float(np.mean(test["field_rmses"])),
        "rmse_definition": (
            "sqrt(mean((prediction-target)^2)) over all output coordinates, "
            "averaged over test samples"
        ),
        "mean_full_band_width": float(2.0 * q90 * np.mean(test["mean_scales"])),
        "scale_error_pearson": float(
            np.corrcoef(
                test["scale_error_pairs"][:, 0],
                test["scale_error_pairs"][:, 1],
            )[0, 1]
        ),
        "scientific_scope": (
            "2D function-valued uncertainty benchmark; no action channel and "
            "therefore no closed-loop control claim."
        ),
        "split_integrity": (
            "Proper training, validation, conformal audit, and test indices are disjoint."
        ),
    }
    (output / "results" / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "base": base.state_dict(),
            "perturbed": perturbed.state_dict(),
            "args": vars(args),
            "floor": floor,
            "q90": q90,
            "perturbed_final_history": perturbed_history[-1],
        },
        output / "results" / "ns2d_models.pt",
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ns2d"))
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-audit", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--label-noise", type=float, default=0.05)
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
