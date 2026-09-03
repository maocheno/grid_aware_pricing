"""Seeded synthetic daily traffic, demand, PV, and grid-price profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticProfiles:
    link_flows: np.ndarray
    od_counts: np.ndarray
    pv_kwh: np.ndarray
    grid_prices: np.ndarray


def _daily_wave(
    hours: np.ndarray,
    peak_hours: np.ndarray,
    peak_widths: np.ndarray,
    phase: float = 0.0,
) -> np.ndarray:
    wave = np.sum(
        np.exp(-0.5 * ((hours[:, None] - peak_hours[None, :] - phase) / peak_widths[None, :]) ** 2),
        axis=1,
    )
    return wave / np.max(wave)


def generate_profiles(config: dict, rng: np.random.Generator | None = None) -> SyntheticProfiles:
    rng = rng or np.random.default_rng(int(config["seed"]))
    periods = int(config["system"]["periods"])
    dt_hours = float(config["system"]["dt_hours"])
    hours = np.arange(periods, dtype=float) * dt_hours
    n_hubs = int(config["system"]["n_hubs"])
    traffic = config["traffic"]
    demand = config["demand"]
    profiles = config["profiles"]
    peak_hours = np.asarray(profiles["traffic_peak_hours"], dtype=float)
    peak_widths = np.asarray(profiles["traffic_peak_width_hours"], dtype=float)

    traffic_wave = _daily_wave(hours, peak_hours, peak_widths)
    base_flows = np.asarray(traffic["base_link_flows"], dtype=float)
    link_noise = rng.normal(0.0, float(traffic["noise_fraction"]), (periods, base_flows.size))
    link_flows = base_flows[None, :] * (
        1.0 + float(traffic["flow_amplitude"]) * traffic_wave[:, None] + link_noise
    )
    link_flows = np.maximum(link_flows, 0.0)

    demand_wave = _daily_wave(
        hours, peak_hours, peak_widths, phase=float(profiles["demand_peak_shift_hours"])
    )
    base_demand = np.asarray(demand["base_od_counts"], dtype=float)
    demand_noise = rng.normal(0.0, float(demand["noise_fraction"]), (periods, base_demand.size))
    od_counts = base_demand[None, :] * (
        1.0 + float(demand["daily_amplitude"]) * demand_wave[:, None] + demand_noise
    )
    od_counts = np.maximum(np.rint(od_counts), 0.0).astype(int)

    solar_start = float(profiles["solar_start_hour"])
    solar_duration = float(profiles["solar_end_hour"]) - solar_start
    solar = np.maximum(np.sin(np.pi * (hours - solar_start) / solar_duration), 0.0)
    pv_peaks = np.asarray(config["hubs"]["pv_peak_kwh"], dtype=float)
    pv_noise = rng.normal(0.0, float(profiles["pv_noise_fraction"]), (periods, n_hubs))
    pv_kwh = np.maximum(pv_peaks[None, :] * solar[:, None] * (1.0 + pv_noise), 0.0)

    shifts = np.asarray(profiles["grid_peak_shift_hours"], dtype=float)
    weights = np.asarray(profiles["grid_wave_weights"], dtype=float)
    weights = weights / weights.sum()
    price_wave = sum(
        weight * _daily_wave(hours, peak_hours, peak_widths, phase=shift)
        for weight, shift in zip(weights, shifts)
    )
    grid_prices = float(profiles["grid_price_base"]) + float(profiles["grid_price_amplitude"]) * price_wave
    return SyntheticProfiles(link_flows, od_counts, pv_kwh, grid_prices)
