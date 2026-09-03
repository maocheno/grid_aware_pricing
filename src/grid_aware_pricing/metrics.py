"""Trajectory output, diagnostic plots, and planned York case-study reporting."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config


_GLOBAL_LIMITATIONS = [
    "No observed charging transactions, queues, weather, or background electrical load are available.",
    "Facility and energy parameters are provisional assumptions rather than measured site attributes.",
    "Traffic conditions are extrapolated from 17 sparse counters to unobserved road links.",
    "The case study covers a single six-hour day window.",
    "The centralized coordinate-search reference is neither an exact oracle nor an upper bound.",
]
_OUTPUT_SCHEMA_VERSION = "2.0"
_EXPECTED_METHODS = (
    "fixed_tariff",
    "myopic_local",
    "proposed",
    "mappo_no_inference",
    "ippo",
    "centralized_coordinate_search_reference",
)


def flatten_info(info: dict) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "period": int(info["period"]),
        "reward": float(info["reward"]),
        "welfare": float(info["welfare"]),
    }
    for key in (
        "grid_price", "outside_share", "inference_loss",
        "outside_cost_estimate", "outside_cost_error",
    ):
        row[key] = float(info[key])
    vector_keys = (
        "prices", "expected_demand", "realized_demand", "requested_energy",
        "served_energy", "profit", "wait", "wait_violation", "unmet",
        "pv_to_ev", "pv_to_battery", "pv_curtailed", "grid_to_ev",
        "grid_to_battery", "battery_to_ev", "soc",
    )
    for key in vector_keys:
        for index, value in enumerate(np.asarray(info[key]).reshape(-1)):
            row[f"{key}_hub_{index}"] = float(value)
    return row


def trajectory_frame(infos: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame([flatten_info(info) for info in infos])


def save_trajectory(infos: Iterable[dict], path: str | Path) -> pd.DataFrame:
    frame = trajectory_frame(infos)
    frame.to_csv(path, index=False)
    return frame


def _columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return [column for column in frame.columns if column.startswith(prefix)]


def plot_trajectory(
    frame_or_path: pd.DataFrame | str | Path, output_dir: str | Path
) -> list[Path]:
    frame = pd.read_csv(frame_or_path) if not isinstance(frame_or_path, pd.DataFrame) else frame_or_path
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    groups = {
        "performance": ["reward", "welfare", *_columns(frame, "profit_hub_")],
        "market": [*_columns(frame, "prices_hub_"), *_columns(frame, "realized_demand_hub_"), "outside_share"],
        "constraints": [*_columns(frame, "wait_hub_"), *_columns(frame, "wait_violation_hub_"), *_columns(frame, "unmet_hub_")],
        "energy": [
            *_columns(frame, "pv_to_ev_hub_"), *_columns(frame, "pv_to_battery_hub_"),
            *_columns(frame, "pv_curtailed_hub_"), *_columns(frame, "grid_to_ev_hub_"),
            *_columns(frame, "grid_to_battery_hub_"), *_columns(frame, "battery_to_ev_hub_"),
            *_columns(frame, "soc_hub_"),
        ],
        "inference": ["inference_loss", "outside_cost_estimate", "outside_cost_error"],
    }
    x = np.arange(len(frame))
    for name, candidates in groups.items():
        columns = [column for column in candidates if column in frame.columns]
        if not columns:
            continue
        fig, axis = plt.subplots(figsize=(8, 4))
        for column in columns:
            axis.plot(x, frame[column].to_numpy(), label=column)
        axis.set_xlabel("step")
        axis.set_title(name.replace("_", " ").title())
        axis.grid(alpha=0.25)
        if len(columns) <= 10:
            axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = output / f"{name}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
    return written


def plot_training_metrics(
    frame_or_path: pd.DataFrame | str | Path, output_dir: str | Path
) -> Path | None:
    frame = pd.read_csv(frame_or_path) if not isinstance(frame_or_path, pd.DataFrame) else frame_or_path
    columns = [
        column for column in (
            "mean_reward", "return", "actor_loss", "critic_loss", "entropy",
            "approx_kl", "clip_fraction", "explained_variance",
        ) if column in frame.columns
    ]
    if not columns:
        return None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(columns), 1, figsize=(8, max(3, 2.2 * len(columns))), squeeze=False)
    for axis, column in zip(axes[:, 0], columns):
        axis.plot(frame[column].to_numpy())
        axis.set_ylabel(column)
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("update")
    fig.tight_layout()
    path = output / "ppo_diagnostics.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _read_first(paths: list[Path]) -> tuple[pd.DataFrame, Path | None]:
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if not frame.empty:
            return frame, path
    return pd.DataFrame(), None


def _discover(result_dir: Path, names: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for name in names:
        direct = result_dir / name
        if direct.exists():
            found.append(direct)
        found.extend(path for path in result_dir.glob(f"**/{name}") if path != direct)
    return list(dict.fromkeys(found))


def _watermark(fig: plt.Figure, status: str, missing: list[str]) -> None:
    if status == "complete":
        return
    detail = "; ".join(missing[:3])
    label = status.upper() + (f": {detail}" if detail else "")
    fig.text(
        0.5, 0.5, label, ha="center", va="center", rotation=24,
        fontsize=18, color="crimson", alpha=0.22, weight="bold",
        transform=fig.transFigure,
    )


def _placeholder(path: Path, title: str, status: str, missing: list[str]) -> None:
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.axis("off")
    axis.set_title(title, fontsize=14, weight="bold")
    axis.text(
        0.5, 0.5, "\n".join(missing) if missing else "No compatible data",
        ha="center", va="center", wrap=True,
    )
    _watermark(fig, status, missing)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _entry(
    number: int,
    section: str,
    title: str,
    files: list[Path],
    sources: list[Path],
    status: str,
    missing: list[str],
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "section": section,
        "section_status": "planned_extension_not_present_in_source_pdf",
        "figure_number": str(number),
        "title": title,
        "files": [str(path.resolve()) for path in files],
        "sources": [str(path.resolve()) for path in sources],
        "status": status,
        "missing_items": missing,
        "limitations": limitations or [],
    }


def _fig3(config: dict[str, Any], path: Path) -> tuple[str, list[str]]:
    scenario = config.get("_york_scenario")
    if scenario is None:
        missing = ["York map data are unavailable"]
        _placeholder(path, "Fig. 3 Study area and infrastructure", "blocked", missing)
        return "blocked", missing
    nodes = scenario.road_nodes.set_index("node_id")
    fig, axis = plt.subplots(figsize=(10, 9))
    edges = scenario.road_edges
    stride = max(len(edges) // 4500, 1)
    for row in edges.iloc[::stride].itertuples(index=False):
        geometry = getattr(row, "geometry", None)
        if geometry:
            points = np.asarray(geometry, dtype=float)
            axis.plot(points[:, 0], points[:, 1], color="0.55", alpha=0.14, linewidth=0.35)
        elif row.u in nodes.index and row.v in nodes.index:
            axis.plot(
                [nodes.loc[row.u, "longitude"], nodes.loc[row.v, "longitude"]],
                [nodes.loc[row.u, "latitude"], nodes.loc[row.v, "latitude"]],
                color="0.55", alpha=0.14, linewidth=0.35,
            )
    counters = scenario.traffic_flow.drop_duplicates("site_number")
    axis.scatter(counters["longitude"], counters["latitude"], marker="^", s=28, c="#333333", label=f"ATC counters ({len(counters)})", zorder=3)
    demand = scenario.od_demand.drop_duplicates("od_id")
    for index, row in enumerate(demand.itertuples(index=False)):
        origin = nodes.loc[str(row.origin_node_id)]
        destination = nodes.loc[str(row.destination_node_id)]
        axis.plot([origin.longitude, destination.longitude], [origin.latitude, destination.latitude], color="#7b3294", alpha=0.42, linewidth=1.1, zorder=2)
        axis.scatter([origin.longitude, destination.longitude], [origin.latitude, destination.latitude], marker="o" if index == 0 else "o", facecolors="white", edgecolors="#7b3294", s=24, zorder=4, label="OD endpoints/lines (6)" if index == 0 else None)
    hubs = scenario.charging_hubs.sort_values("hub_id")
    sizes = 35.0 + 15.0 * hubs["num_chargers"].to_numpy(dtype=float)
    scatter = axis.scatter(hubs["longitude"], hubs["latitude"], s=sizes, c=hubs["grid_import_limit_kw"], cmap="viridis", edgecolors="black", linewidths=0.6, zorder=5, label="Charging hubs (8)")
    for row in hubs.itertuples(index=False):
        axis.annotate(row.hub_id, (row.longitude, row.latitude), xytext=(4, 4), textcoords="offset points", fontsize=8)
    colorbar = fig.colorbar(scatter, ax=axis, shrink=0.72)
    colorbar.set_label("Grid import capacity (kW); marker size = chargers")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title("Fig. 3. York study area, traffic counters, OD pairs, and charging hubs")
    axis.legend(loc="best", fontsize=8)
    axis.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return "complete", []


def _training_frames(result_dir: Path) -> tuple[list[pd.DataFrame], list[Path]]:
    discovered = _discover(result_dir, ("training_metrics.csv", "train_metrics.csv"))
    proposed = [
        path for path in discovered
        if "proposed" in path.relative_to(result_dir).parts
    ]
    paths = proposed or [path for path in discovered if path.parent == result_dir]
    frames: list[pd.DataFrame] = []
    used: list[Path] = []
    seen: set[tuple[str, int]] = set()
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if frame.empty:
            continue
        seed = int(frame["training_seed"].iloc[0]) if "training_seed" in frame else None
        if seed is None:
            manifest = path.parent / "run_manifest.json"
            if manifest.exists():
                try:
                    seed = int(json.loads(manifest.read_text(encoding="utf-8")).get("seed"))
                except (ValueError, TypeError, json.JSONDecodeError):
                    seed = None
        seed = seed if seed is not None else len(frames)
        identity = (str(path.parent.resolve()), seed)
        if identity in seen:
            continue
        seen.add(identity)
        frame = frame.copy()
        frame["training_seed"] = seed
        frames.append(frame)
        used.append(path)
    return frames, used


def _fig4(result_dir: Path, path: Path) -> tuple[str, list[str], list[Path]]:
    frames, sources = _training_frames(result_dir)
    if not frames:
        missing = ["No training_metrics.csv files"]
        _placeholder(path, "Fig. 4 Learning convergence", "blocked", missing)
        return "blocked", missing, sources
    data = pd.concat(frames, ignore_index=True)
    seed_count = data["training_seed"].nunique()
    episode_column = "episode" if "episode" in data else None
    max_length = max(len(frame) for frame in frames)
    missing = []
    if seed_count < 3:
        missing.append(f"Only {seed_count} training seed(s); at least 3 required")
    if max_length < 100:
        missing.append(f"Training history has only {max_length} rows per longest seed")
    status = "partial" if missing else "complete"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    metric_specs = (
        ((0, 0), "return", "Episode return"),
        ((0, 1), "outside_cost_estimate", "Outside estimate (GBP)"),
        ((1, 0), "outside_nll", "Outside-option NLL"),
        ((1, 1), "wait_violation", "Wait/unmet violations"),
    )
    for (row, col), metric, label in metric_specs:
        axis = axes[row, col]
        if metric in data:
            for seed, group in data.groupby("training_seed"):
                group = group.sort_values(episode_column) if episode_column else group
                values = pd.to_numeric(group[metric], errors="coerce")
                window = max(1, min(50, len(group) // 10 or 1))
                mean = values.rolling(window, min_periods=1).mean()
                std = values.rolling(window, min_periods=min(2, window)).std().fillna(0.0)
                x = group[episode_column].to_numpy() if episode_column else np.arange(len(group))
                axis.plot(x, mean, label=f"seed {seed}")
                axis.fill_between(x, mean - std, mean + std, alpha=0.12)
            if metric == "outside_cost_estimate":
                axis.axhline(16.5, color="black", linestyle="--", linewidth=1.0, label="true 16.5")
        else:
            axis.text(0.5, 0.5, f"missing {metric}", ha="center", va="center", transform=axis.transAxes)
        if metric == "wait_violation" and "unmet" in data:
            grouped = data.groupby("training_seed", sort=False)
            for seed, group in grouped:
                x = group[episode_column].to_numpy() if episode_column else np.arange(len(group))
                axis.plot(x, pd.to_numeric(group["unmet"], errors="coerce"), linestyle=":", label=f"unmet seed {seed}")
        axis.set_title(label)
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
        if row == 0 and col == 0:
            axis.legend(fontsize=7)
    fig.suptitle("Fig. 4. Learning convergence and lower-layer estimation")
    _watermark(fig, status, missing)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return status, missing, sources


def _benchmark_metadata(result_dir: Path) -> tuple[dict[str, Any], Path | None]:
    manifests = _discover(result_dir, ("benchmark_manifest.json",))
    for path in manifests:
        try:
            return json.loads(path.read_text(encoding="utf-8")), path
        except (OSError, json.JSONDecodeError):
            continue
    return {}, None


def _benchmark_protocol_missing(metadata: dict[str, Any]) -> list[str]:
    if metadata.get("stochastic_evaluation") is False:
        return [
            "Benchmark uses deterministic expected demand; stochastic multi-scenario evaluation is missing"
        ]
    return []


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benchmark_bundle(
    runtime: dict[str, Any], result_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[Path], list[str]]:
    metadata, manifest_path = _benchmark_metadata(result_dir)
    if manifest_path is None:
        episodes, period, sources = _benchmark_data(result_dir)
        sources.extend(path for path in (manifest_path,) if path is not None)
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            metadata,
            sources,
            ["Compatible benchmark_manifest.json is missing"],
        )

    episode_path = manifest_path.parent / "benchmark_episodes.csv"
    period_path = manifest_path.parent / "benchmark_period_hub.csv"
    sources = [manifest_path, episode_path, period_path]
    missing: list[str] = []
    expected_metadata = {
        "output_schema_version": _OUTPUT_SCHEMA_VERSION,
        "environment_schema_version": runtime.get("environment_schema_version"),
        "queue_semantics": runtime.get("queue_semantics"),
        "zip_sha256": runtime.get("data", {}).get("zip_sha256"),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            missing.append(f"Benchmark {key} is incompatible with the current config")
    artifacts = metadata.get("artifacts", {})
    frames: dict[str, pd.DataFrame] = {}
    for name, path in (
        ("benchmark_episodes.csv", episode_path),
        ("benchmark_period_hub.csv", period_path),
    ):
        specification = artifacts.get(name, {})
        if not path.exists() or not isinstance(specification, dict):
            missing.append(f"Benchmark artifact is missing from manifest bundle: {name}")
            continue
        if specification.get("sha256") != _file_sha256(path):
            missing.append(f"Benchmark artifact hash mismatch: {name}")
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            missing.append(f"Benchmark artifact cannot be read: {name}")
            continue
        if int(specification.get("rows", -1)) != len(frame):
            missing.append(f"Benchmark artifact row-count mismatch: {name}")
        expected_columns = specification.get("columns")
        if expected_columns is not None and list(frame.columns) != list(expected_columns):
            missing.append(f"Benchmark artifact column mismatch: {name}")
        frames[name] = frame
    if missing:
        return pd.DataFrame(), pd.DataFrame(), metadata, sources, missing
    episodes = frames["benchmark_episodes.csv"].copy()
    legacy_aliases = {
        "profit": "profit_gbp",
        "weighted_hub_profit_welfare": "welfare_gbp",
        "welfare_gbp": "weighted_hub_profit_welfare",
        "unmet": "unmet_energy_kwh",
        "centralized_reference_gap": "centralized_reference_difference",
    }
    for old, new in legacy_aliases.items():
        if new not in episodes and old in episodes:
            episodes[new] = episodes[old]
    return episodes, frames["benchmark_period_hub.csv"], metadata, sources, []


def _benchmark_data(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    episode, episode_path = _read_first(_discover(result_dir, ("benchmark_episodes.csv", "evaluation_episodes.csv", "episodes.csv")))
    period, period_path = _read_first(_discover(result_dir, ("benchmark_period_hub.csv", "evaluation_period_hub.csv", "period_hub.csv")))
    episode = episode.copy()
    legacy_aliases = {
        "profit": "profit_gbp",
        "weighted_hub_profit_welfare": "welfare_gbp",
        "welfare_gbp": "weighted_hub_profit_welfare",
        "unmet": "unmet_energy_kwh",
        "centralized_reference_gap": "centralized_reference_difference",
    }
    for old, new in legacy_aliases.items():
        if new not in episode and old in episode:
            episode[new] = episode[old]
    return episode, period, [path for path in (episode_path, period_path) if path]


def _mechanism_data(result_dir: Path, benchmark_period: pd.DataFrame) -> tuple[pd.DataFrame, list[Path], bool]:
    frame, path = _read_first(_discover(result_dir, ("proposed_mechanism_period_hub.csv",)))
    if not frame.empty:
        return frame, [path] if path else [], True
    if not benchmark_period.empty and "method" in benchmark_period:
        diagnostic = benchmark_period[benchmark_period["method"] == "proposed"].copy()
        return diagnostic, [], False
    return pd.DataFrame(), [], False


def _fig5(
    result_dir: Path,
    benchmark_period: pd.DataFrame,
    path: Path,
    bundle_missing: list[str] | None = None,
) -> tuple[str, list[str], list[Path]]:
    frame, sources, mechanism = _mechanism_data(result_dir, benchmark_period)
    missing: list[str] = list(bundle_missing or [])
    if not mechanism:
        missing.append("Dedicated deterministic online proposed mechanism replay is missing")
    proposed_manifests = list(result_dir.glob("**/training/proposed/seed_*/run_manifest.json"))
    if not proposed_manifests:
        proposed_manifests = list(result_dir.glob("training/proposed/seed_*/run_manifest.json"))
    episode_counts: list[int] = []
    for manifest in proposed_manifests:
        try:
            episodes = json.loads(manifest.read_text(encoding="utf-8")).get("episodes")
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(episodes, int):
            episode_counts.append(episodes)
    if episode_counts and max(episode_counts) < 100:
        missing.append(
            f"Proposed checkpoint is an undertrained smoke model ({max(episode_counts)} episodes); mechanism illustration only"
        )
    sources.extend(path for path in proposed_manifests if path not in sources)
    if frame.empty:
        missing.append("No proposed period-hub rows")
        _placeholder(path, "Fig. 5 Dynamic prices and allocation", "partial", missing)
        return "partial", missing, sources
    hub_col = "hub_id" if "hub_id" in frame else "hub"
    period_col = "timestamp" if "timestamp" in frame and frame["timestamp"].notna().any() else "period"
    demand_col = "arrivals" if "arrivals" in frame else "realized_demand"
    if frame[hub_col].nunique() < 8:
        missing.append(f"Only {frame[hub_col].nunique()} hubs are present")
    status = "partial" if missing else "complete"
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), squeeze=False)
    pivot = frame.pivot_table(index=period_col, columns=hub_col, values="price", aggfunc="mean")
    for column in pivot:
        axes[0, 0].plot(np.arange(len(pivot)), pivot[column], marker="o", label=str(column))
    axes[0, 0].set_title("Hub prices")
    axes[0, 0].set_ylabel("GBP/kWh")
    axes[0, 0].legend(ncol=4, fontsize=7)
    if demand_col in frame:
        heat = frame.pivot_table(index=hub_col, columns=period_col, values=demand_col, aggfunc="mean")
        image = axes[1, 0].imshow(heat.to_numpy(), aspect="auto", cmap="magma")
        axes[1, 0].set_yticks(np.arange(len(heat.index)), labels=[str(value) for value in heat.index])
        axes[1, 0].set_title("Charging-hub demand allocation")
        fig.colorbar(image, ax=axes[1, 0], label="requests")
    else:
        axes[1, 0].text(0.5, 0.5, "Demand allocation unavailable", ha="center", va="center")
    if "outside_share" in frame:
        outside = frame.groupby(period_col)["outside_share"].mean()
        axes[2, 0].plot(np.arange(len(outside)), 1.0 - outside, label="public hubs")
        axes[2, 0].plot(np.arange(len(outside)), outside, label="outside option")
        axes[2, 0].set_ylim(0.0, 1.0)
        axes[2, 0].legend()
    axes[2, 0].set_title("Public-hub and outside-option shares")
    for axis in axes[:, 0]:
        axis.grid(alpha=0.2)
        axis.set_xlabel("Period")
    fig.suptitle("Fig. 5. Dynamic pricing and demand allocation")
    _watermark(fig, status, missing)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return status, missing, sources


def _method_seed_missing(episodes: pd.DataFrame) -> list[str]:
    if episodes.empty or "method" not in episodes:
        return ["Benchmark episode data are missing"]
    methods = set(episodes["method"].astype(str))
    missing = [f"Missing method: {method}" for method in _EXPECTED_METHODS if method not in methods]
    trained_methods = {"proposed", "mappo_no_inference", "ippo"}
    for method in _EXPECTED_METHODS:
        group = episodes[episodes["method"] == method]
        if group.empty:
            continue
        seed_col = "training_seed" if method in trained_methods else "scenario_seed"
        seed_count = group[seed_col].nunique() if seed_col in group else 0
        if seed_count < 3:
            seed_label = "training seeds" if method in trained_methods else "scenario seeds"
            missing.append(f"{method} has fewer than 3 {seed_label}")
    return missing


def _bar_with_error(axis: plt.Axes, frame: pd.DataFrame, metric: str, title: str) -> None:
    if metric not in frame:
        axis.text(0.5, 0.5, f"missing {metric}", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        return
    stats = frame.groupby("method", sort=False)[metric].agg(["mean", "std"])
    axis.bar(np.arange(len(stats)), stats["mean"], yerr=stats["std"].fillna(0.0), capsize=3)
    axis.set_xticks(np.arange(len(stats)), labels=stats.index, rotation=28, ha="right", fontsize=8)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)


def _fig6(
    episodes: pd.DataFrame,
    period: pd.DataFrame,
    path: Path,
    protocol_missing: list[str] | None = None,
) -> tuple[str, list[str]]:
    missing = [*_method_seed_missing(episodes), *(protocol_missing or [])]
    required_episode = {
        "mean_wait_min", "p95_wait_min", "max_wait_min", "wait_violation_rate",
        "peak_queued_energy_kwh", "minimum_admission_ratio",
        "peak_admission_pressure", "queue_cleared_by_end",
    }
    required_period = {
        "period", "method", "queued_energy_next_kwh", "admission_ratio",
        "admission_pressure", "wait_min",
    }
    absent_episode = sorted(required_episode - set(episodes.columns))
    absent_period = sorted(required_period - set(period.columns))
    if absent_episode:
        missing.append("Missing queue episode metrics: " + ", ".join(absent_episode))
    if absent_period:
        missing.append("Missing queue trajectory fields: " + ", ".join(absent_period))
    if episodes.empty or period.empty or absent_episode or absent_period:
        _placeholder(path, "Fig. 6 Queue and service quality", "blocked", missing)
        return "blocked", missing
    status = "partial" if missing else "complete"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for axis, (metric, title) in zip(axes[0], (
        ("mean_wait_min", "Mean wait (min)"),
        ("p95_wait_min", "P95 wait (min)"),
        ("max_wait_min", "Maximum wait (min)"),
    )):
        _bar_with_error(axis, episodes, metric, title)
        axis.axhline(15.0, color="crimson", linestyle="--", linewidth=0.9)

    queue = period.groupby(["method", "episode", "period"], sort=False)["queued_energy_next_kwh"].sum().reset_index()
    queue_mean = queue.groupby(["method", "period"], sort=False)["queued_energy_next_kwh"].mean().reset_index()
    for method, group in queue_mean.groupby("method", sort=False):
        axes[1, 0].plot(group["period"], group["queued_energy_next_kwh"], marker="o", label=method)
    axes[1, 0].set_title("System queued energy carryover")
    axes[1, 0].set_ylabel("kWh")
    axes[1, 0].legend(fontsize=6)

    _bar_with_error(axes[1, 1], episodes, "wait_violation_rate", "15-min violation rate")
    admission = episodes.groupby("method", sort=False)[
        ["minimum_admission_ratio", "peak_admission_pressure"]
    ].mean()
    x = np.arange(len(admission))
    axes[1, 2].bar(x, admission["minimum_admission_ratio"], label="minimum admission ratio")
    pressure_axis = axes[1, 2].twinx()
    pressure_axis.plot(x, admission["peak_admission_pressure"], color="tab:red", marker="o", label="peak pressure")
    axes[1, 2].set_xticks(x, labels=admission.index, rotation=28, ha="right", fontsize=8)
    axes[1, 2].set_ylim(0.0, 1.05)
    axes[1, 2].set_title("Admission and queue clearance pressure")
    pressure_axis.set_ylabel("Pending / service capacity", color="tab:red")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xlabel("Period" if axis is axes[1, 0] else "")
    fig.suptitle("Fig. 6. Cross-period queue carryover, 15-minute violations, and admission")
    _watermark(fig, status, missing)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return status, missing


def _fig7(
    episodes: pd.DataFrame,
    period: pd.DataFrame,
    path: Path,
    protocol_missing: list[str] | None = None,
) -> tuple[str, list[str]]:
    missing = [*_method_seed_missing(episodes), *(protocol_missing or [])]
    proposed = period[period["method"] == "proposed"].copy() if not period.empty and "method" in period else pd.DataFrame()
    needed = {
        "pv_used_kwh", "battery_charge_kwh", "battery_discharge_kwh",
        "grid_import_kwh", "soc", "pending_energy_kwh", "admitted_energy_kwh",
        "served_energy_kwh", "unmet_energy_kwh", "grid_utilization",
    }
    absent = sorted(needed - set(proposed.columns))
    if proposed.empty:
        missing.append("Proposed period-hub data are missing")
    if absent:
        missing.append("Missing proposed energy fields: " + ", ".join(absent))
    if proposed.empty or absent:
        _placeholder(path, "Fig. 7 Energy and grid operation", "partial", missing)
        return "partial", missing
    status = "partial" if missing else "complete"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
    by_period = proposed.groupby("period")[["pv_used_kwh", "battery_discharge_kwh", "grid_import_kwh"]].sum()
    axes[0, 0].stackplot(by_period.index, *(by_period[column] for column in by_period), labels=["PV used", "Battery discharge", "Grid import"], alpha=0.8)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Proposed system supply stack")
    stages = proposed.groupby("period")[[
        "pending_energy_kwh", "admitted_energy_kwh", "served_energy_kwh", "unmet_energy_kwh"
    ]].sum()
    for column in stages:
        axes[0, 1].plot(stages.index, stages[column], marker="o", label=column.replace("_kwh", ""))
    axes[0, 1].set_title("Pending, admitted, served, and unmet energy")
    axes[0, 1].set_ylabel("kWh")
    axes[0, 1].legend(fontsize=7)

    hub_col = "hub_id" if "hub_id" in proposed else "hub"
    selected = [hub for hub in ("H01", "H02", "H05", "H06") if hub in set(proposed[hub_col].astype(str))]
    if not selected:
        selected = [str(value) for value in proposed[hub_col].drop_duplicates().iloc[:4]]
    for hub in selected:
        group = proposed[proposed[hub_col].astype(str) == hub]
        axes[1, 0].plot(group["period"], group["soc"], label=hub)
        axes[1, 1].plot(group["period"], group["grid_utilization"], label=hub)
    axes[1, 0].set_title("H01/H02/H05/H06 state of charge")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_title("H01/H02/H05/H06 grid-limit utilization")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("Fig. 7. Renewable, battery, and grid operation")
    _watermark(fig, status, missing)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return status, missing


def _fig8(
    episodes: pd.DataFrame,
    path: Path,
    protocol_missing: list[str] | None = None,
) -> tuple[str, list[str]]:
    missing = [*_method_seed_missing(episodes), *(protocol_missing or [])]
    requested = {
        "mean_detour_min", "mean_access_min", "profit_gbp", "wait_violation_rate",
        "peak_queued_energy_kwh", "centralized_reference_difference",
    }
    absent = sorted(requested - set(episodes.columns))
    if absent:
        missing.append("Missing metrics: " + ", ".join(absent))
    if episodes.empty or absent:
        _placeholder(path, "Fig. 8 Benchmark performance", "blocked", missing)
        return "blocked", missing
    status = "partial" if missing else "complete"
    stats = episodes.groupby("method", sort=False)[list(requested)].mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
    scatter_axis = axes[0, 0]
    sizes = 60.0 + 300.0 * stats["peak_queued_energy_kwh"] / max(
        float(stats["peak_queued_energy_kwh"].max()), 1e-12
    )
    scatter = scatter_axis.scatter(
        stats["mean_detour_min"], stats["profit_gbp"], s=sizes,
        c=stats["wait_violation_rate"], cmap="viridis_r", edgecolors="black", alpha=0.82,
    )
    for method, row in stats.iterrows():
        scatter_axis.annotate(str(method), (row["mean_detour_min"], row["profit_gbp"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    scatter_axis.set_xlabel("Mean detour (min)")
    scatter_axis.set_ylabel("Aggregate hub profit (GBP)")
    scatter_axis.set_title("Access/detour–profit trade-off\nsize = peak queued energy")
    fig.colorbar(scatter, ax=scatter_axis, label="15-min wait violation rate")

    reference_axis = axes[0, 1]
    reference_axis.bar(np.arange(len(stats)), stats["centralized_reference_difference"])
    reference_axis.axhline(0.0, color="black", linewidth=0.8)
    reference_axis.set_xticks(np.arange(len(stats)), labels=stats.index, rotation=35, ha="right", fontsize=8)
    reference_axis.set_ylabel("Reference return - method return")
    reference_axis.set_title("Signed coordinate-search reference difference\nnot an oracle gap or upper bound")
    for axis in axes[0]:
        axis.grid(alpha=0.22)
    fig.suptitle("Fig. 8. Benchmark trade-off and non-oracle centralized reference")
    _watermark(fig, status, missing)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return status, missing


def _table1(
    episodes: pd.DataFrame,
    path: Path,
    protocol_missing: list[str] | None = None,
) -> tuple[str, list[str]]:
    metrics = {
        "return": "scaled_reward",
        "profit_gbp": "GBP",
        "weighted_hub_profit_welfare": "GBP_weighted_hub_profit",
        "served_requests": "requests",
        "outside_share": "fraction",
        "pending_requests": "requests",
        "admitted_requests": "requests",
        "admission_ratio": "fraction",
        "minimum_admission_ratio": "fraction",
        "peak_admission_pressure": "pending_over_capacity",
        "admitted_full_service_ratio": "fraction",
        "pending_full_service_ratio": "fraction",
        "mean_wait_min": "min",
        "p95_wait_min": "min",
        "max_wait_min": "min",
        "wait_violation_rate": "fraction",
        "mean_wait_excess_min": "min",
        "pending_energy_kwh": "kWh",
        "admitted_energy_kwh": "kWh",
        "unmet_energy_kwh": "kWh",
        "peak_queued_energy_kwh": "kWh",
        "mean_queued_energy_kwh": "kWh",
        "final_queued_energy_kwh": "kWh",
        "queue_cleared_by_end": "boolean",
        "queue_clearance_period": "period_index",
        "max_queue_vehicle_conservation_error": "hours",
        "max_queue_energy_conservation_error_kwh": "kWh",
        "grid_energy_kwh": "kWh",
        "peak_grid_import_kwh": "kWh_per_period",
        "pv_utilization": "fraction",
        "battery_throughput_kwh": "kWh",
        "energy_cost_gbp": "GBP",
        "mean_access_min": "min",
        "p95_access_min": "min",
        "mean_detour_min": "min",
        "p95_detour_min": "min",
        "outside_mae": "GBP",
        "outside_nll": "nll",
        "approx_unilateral_gain": "scaled_local_reward",
        "centralized_reference_difference": "scaled_reward",
        "exact_oracle_gap": "scaled_reward",
    }
    observed = list(dict.fromkeys(episodes["method"].astype(str))) if not episodes.empty and "method" in episodes else []
    methods = list(dict.fromkeys([*_EXPECTED_METHODS, *observed]))
    rows: list[dict[str, Any]] = []
    for method in methods:
        group = episodes[episodes["method"] == method] if not episodes.empty and "method" in episodes else pd.DataFrame()
        for metric, unit in metrics.items():
            values = pd.to_numeric(group[metric], errors="coerce").dropna() if metric in group else pd.Series(dtype=float)
            reason = "" if len(values) else ("metric_not_recorded" if metric not in group else "no_finite_values")
            rows.append({
                "method": method, "metric": metric,
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else np.nan),
                "mean_plus_minus_std": f"{values.mean():.6g} ± {values.std(ddof=1) if len(values) > 1 else 0.0:.6g}" if len(values) else "NaN",
                "n": int(len(values)), "unit": unit,
                "availability": "available" if len(values) else "missing",
                "missing_reason": reason,
            })
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    missing = [*_method_seed_missing(episodes), *(protocol_missing or [])]
    if table["availability"].eq("missing").any():
        missing.append("Some requested Table I metrics are unavailable; see missing_reason")
    return ("partial" if missing else "complete"), missing


def generate_case_study_report(
    config: str | Path | dict[str, Any],
    result_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate planned Section V Fig. 3-8, Table I, and a provenance manifest."""
    runtime = load_config(config) if not isinstance(config, dict) else config
    results = Path(result_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, Any]] = []

    fig3 = output / "fig3_study_area.png"
    status, missing = _fig3(runtime, fig3)
    figures.append(_entry(3, "Planned Section V-A", "York study area and infrastructure", [fig3], [Path(runtime.get("_york_zip_path", runtime.get("_config_path", "")))], status, missing, ["Counter observations are spatially sparse; facility capacities are provisional."]))

    fig4 = output / "fig4_learning_convergence.png"
    status, missing, sources = _fig4(results, fig4)
    figures.append(_entry(4, "Planned Section V-B", "Learning convergence and preference estimation", [fig4], sources, status, missing, ["Convergence claims require at least three sufficiently long independent training seeds."]))

    episodes, period, benchmark_metadata, benchmark_sources, bundle_missing = (
        _benchmark_bundle(runtime, results)
    )
    protocol_missing = [
        *bundle_missing,
        *_benchmark_protocol_missing(benchmark_metadata),
    ]
    fig5 = output / "fig5_dynamic_prices_allocation.png"
    status, missing, sources = _fig5(results, period, fig5, bundle_missing)
    figures.append(_entry(5, "Planned Section V-C", "Dynamic prices and demand allocation", [fig5], sources or benchmark_sources, status, missing, ["Only a deterministic single-episode online replay is a mechanism illustration; it is not the frozen statistical evaluation."]))

    fig6 = output / "fig6_queue_service.png"
    status, missing = _fig6(episodes, period, fig6, protocol_missing)
    figures.append(_entry(6, "Planned Section V-C/V-D", "Queue congestion and service quality", [fig6], benchmark_sources, status, missing, ["Queue outcomes are simulator outputs, not observed queue measurements."]))

    fig7 = output / "fig7_energy_grid.png"
    mechanism_period, mechanism_sources, _ = _mechanism_data(results, period)
    energy_period = period
    if not mechanism_period.empty:
        non_proposed = period[period["method"] != "proposed"] if not period.empty and "method" in period else pd.DataFrame()
        energy_period = pd.concat([non_proposed, mechanism_period], ignore_index=True)
    status, missing = _fig7(episodes, energy_period, fig7, protocol_missing)
    figures.append(_entry(7, "Planned Section V-D", "Energy dispatch and grid interaction", [fig7], benchmark_sources + mechanism_sources, status, missing, ["PV, battery, grid-limit, and background-load assumptions are not site measurements."]))

    fig8 = output / "fig8_benchmark_performance.png"
    status, missing = _fig8(episodes, fig8, protocol_missing)
    figures.append(_entry(8, "Planned Section V-E", "Benchmark performance and signed centralized-reference difference", [fig8], benchmark_sources, status, missing, ["centralized_reference_difference is signed reference return minus method return; it is not an oracle gap."]))

    table_path = output / "table1_main_results.csv"
    table_status, table_missing = _table1(episodes, table_path, protocol_missing)
    table_entry = {
        "section": "Planned Section V-E",
        "section_status": "planned_extension_not_present_in_source_pdf",
        "table_number": "I",
        "title": "Main benchmark results",
        "files": [str(table_path.resolve())],
        "sources": [str(path.resolve()) for path in benchmark_sources],
        "status": table_status,
        "missing_items": table_missing,
        "limitations": ["NaN values are retained explicitly with missing_reason; no unavailable metric is imputed."],
    }
    manifest = {
        "schema_version": _OUTPUT_SCHEMA_VERSION,
        "output_schema_version": _OUTPUT_SCHEMA_VERSION,
        "environment_schema_version": runtime.get("environment_schema_version"),
        "queue_semantics": runtime.get("queue_semantics"),
        "zip_sha256": runtime.get("data", {}).get("zip_sha256"),
        "generated_from": str(results.resolve()),
        "config": str(runtime.get("_config_path", config if not isinstance(config, dict) else "runtime_dict")),
        "section_status": "planned_extension_not_present_in_source_pdf",
        "global_limitations": _GLOBAL_LIMITATIONS,
        "reference_semantics": {
            "centralized_reference_difference": "reference return - method return (signed)",
            "centralized_reference_gap": "deprecated non-oracle alias",
            "exact_oracle_gap": None,
            "reference_is_upper_bound": False,
        },
        "benchmark_bundle": {
            "status": "compatible" if not bundle_missing else "incompatible",
            "validation_errors": bundle_missing,
            "manifest": next(
                (
                    str(path.resolve())
                    for path in benchmark_sources
                    if path.name == "benchmark_manifest.json"
                ),
                None,
            ),
        },
        "benchmark_evaluation": {
            "mode": benchmark_metadata.get("evaluation_mode", "unknown"),
            "stochastic": benchmark_metadata.get("stochastic_evaluation"),
            "scenario_seeds": benchmark_metadata.get("scenario_seeds", []),
            "common_scenario_seeds_across_methods": benchmark_metadata.get(
                "common_scenario_seeds_across_methods"
            ),
        },
        "figures": figures,
        "tables": [table_entry],
    }
    (output / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    return manifest
