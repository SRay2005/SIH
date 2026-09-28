#!/usr/bin/env python3
"""
Smart Scan Strategy submission evaluator.

This evaluator is a deterministic, organizer-controlled benchmark for the SIH
Smart Scan Strategy problem. It runs all submissions against the same seeded
scenarios and computes the SIH-facing metrics named in the problem statement,
plus diagnostics useful for improving strategies.

Usage:
    python evaluate_submissions.py
    python evaluate_submissions.py --submissions ./submissions
    python evaluate_submissions.py --scripts scheduler_alice.py scheduler_bob.py --no-baseline
    python evaluate_submissions.py --script scheduler_alice.py --script scheduler_bob.py
    python evaluate_submissions.py --scripts alice bob --no-baseline
    python evaluate_submissions.py --slots 2000 --seed 123
    python evaluate_submissions.py --no-baseline
    python evaluate_submissions.py --match linucb
    python evaluate_submissions.py --list-scripts

Default directory layout:
    evaluate_submissions.py
    submissions/
        scheduler_alice.py
        scheduler_bob.py

Outputs are written to evaluation_output/.

Important benchmark note:
The SIH problem statement names nine figures of merit but does not specify a
complete mathematical scoring formula for every metric. This evaluator uses
the explicit conventions documented in submission_interface_spec.md and applies
them identically to all submissions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.util
import math
import os
import random
import statistics
import sys
import time
import types
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# =============================================================================
# Public submission interface
# =============================================================================


@dataclass
class Observation:
    slot: int
    action: int
    window: List[int]
    snr: List[float]
    hit: List[bool]
    blind: bool


@dataclass
class Prediction:
    band: int
    predicted_slot: int
    confidence: float = 1.0


class BaseScheduler:
    def __init__(self, num_bands: int, window_size: int, config: dict):
        self.num_bands = num_bands
        self.window_size = window_size
        self.config = config

    def select_action(self, slot: int) -> int:
        raise NotImplementedError

    def update(self, obs: Observation) -> None:
        raise NotImplementedError

    def predict(self, slot: int) -> List[Prediction]:
        """Optional. Return predictions made before the current slot executes."""
        return []

    def reset(self) -> None:
        pass


# Dynamic submission modules do: from base_scheduler import ...
base_scheduler_module = types.ModuleType("base_scheduler")
base_scheduler_module.Observation = Observation
base_scheduler_module.Prediction = Prediction
base_scheduler_module.BaseScheduler = BaseScheduler
sys.modules["base_scheduler"] = base_scheduler_module


# =============================================================================
# RF simulator
# =============================================================================


@dataclass
class Emitter:
    name: str
    kind: str  # fixed, periodic, hopping, agile
    bands: List[int]
    start: int = 0
    end: Optional[int] = None
    interval: int = 1
    phase: int = 0
    duration: int = 1
    snr: float = 10.0

    # Hopping
    hop_sequence: List[int] = field(default_factory=list)
    hop_probability: Optional[float] = None

    # Optional mid-run mode change
    mode_change_slot: Optional[int] = None
    bands_after: Optional[List[int]] = None
    interval_after: Optional[int] = None
    duration_after: Optional[int] = None

    def parameters_at(self, slot: int) -> Tuple[List[int], int, int]:
        if self.mode_change_slot is not None and slot >= self.mode_change_slot:
            return (
                list(self.bands_after if self.bands_after is not None else self.bands),
                self.interval_after if self.interval_after is not None else self.interval,
                self.duration_after if self.duration_after is not None else self.duration,
            )
        return list(self.bands), self.interval, self.duration


@dataclass
class Scenario:
    name: str
    seed: int
    num_bands: int
    window_size: int
    slots: int
    emitters: List[Emitter]
    retune_blind_slots: int = 1
    detection_threshold: float = 5.0
    measurement_noise_std: float = 1.25
    noise_floor: float = 0.0
    agile_duty_cycle: float = 0.25
    prediction_time_tolerance: int = 1
    max_predictions_per_slot: int = 4
    urgency_delta_snr: float = 3.0


class RfSimulator:
    """Private RF environment. Hidden truth is never passed to submissions."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self._last_action: Optional[int] = None
        self._blind_remaining = 0
        self._current_slot = 0
        self._truth: Dict[int, Dict[int, float]] = {
            slot: {} for slot in range(scenario.slots)
        }
        self._events: Dict[int, List[Tuple[str, int]]] = {
            slot: [] for slot in range(scenario.slots)
        }
        self._measurements: Dict[int, Dict[int, float]] = {
            slot: {} for slot in range(scenario.slots)
        }
        self._build_truth()
        self._build_measurements()

    def _rng(self, *parts: int) -> random.Random:
        x = self.scenario.seed & 0xFFFFFFFFFFFFFFFF
        for part in parts:
            x = (
                x * 6364136223846793005
                + int(part)
                + 1442695040888963407
            ) & 0xFFFFFFFFFFFFFFFF
        return random.Random(x)

    def _build_truth(self) -> None:
        for emitter_index, emitter in enumerate(self.scenario.emitters):
            for slot in range(self.scenario.slots):
                if slot < emitter.start:
                    continue
                if emitter.end is not None and slot > emitter.end:
                    continue

                bands, interval, duration = emitter.parameters_at(slot)
                if not bands:
                    continue

                active_bands: List[int] = []

                if emitter.kind in {"fixed", "periodic"}:
                    interval = max(1, interval)
                    duration = max(1, duration)
                    offset = slot - emitter.start - emitter.phase
                    if offset >= 0 and offset % interval < duration:
                        active_bands = bands

                elif emitter.kind == "hopping":
                    # Hopping has randomized timing AND randomized band.
                    probability = (
                        emitter.hop_probability
                        if emitter.hop_probability is not None
                        else 1.0 / max(1, interval)
                    )
                    probability = min(1.0, max(0.0, probability))
                    rng = self._rng(101, emitter_index, slot)
                    if rng.random() < probability:
                        sequence = emitter.hop_sequence or bands
                        active_bands = [sequence[rng.randrange(len(sequence))]]

                elif emitter.kind == "agile":
                    rng = self._rng(103, emitter_index, slot)
                    if rng.random() < self.scenario.agile_duty_cycle:
                        active_bands = [bands[rng.randrange(len(bands))]]

                for band in active_bands:
                    if not 0 <= band < self.scenario.num_bands:
                        continue
                    self._truth[slot][band] = max(
                        self._truth[slot].get(band, -math.inf),
                        emitter.snr,
                    )
                    self._events[slot].append((emitter.name, band))

    def _build_measurements(self) -> None:
        noise_sd = max(0.0, self.scenario.measurement_noise_std)
        for slot in range(self.scenario.slots):
            for band in range(self.scenario.num_bands):
                rng = self._rng(107, slot, band)
                noise = rng.gauss(0.0, noise_sd)
                true_snr = self._truth[slot].get(
                    band,
                    self.scenario.noise_floor,
                )
                self._measurements[slot][band] = float(true_snr + noise)

    def step(self, action: int) -> Observation:
        s = self.scenario
        max_action = s.num_bands - s.window_size

        if not isinstance(action, int) or isinstance(action, bool):
            raise ValueError(
                f"action must be int; got {type(action).__name__}"
            )
        if not 0 <= action <= max_action:
            raise ValueError(
                f"action {action} outside valid range [0, {max_action}]"
            )

        window = list(range(action, action + s.window_size))

        if self._last_action is not None and action != self._last_action:
            self._blind_remaining = max(
                self._blind_remaining,
                s.retune_blind_slots,
            )
        self._last_action = action

        slot = self._current_slot
        if self._blind_remaining > 0:
            self._blind_remaining -= 1
            return Observation(
                slot=slot,
                action=action,
                window=window,
                snr=[],
                hit=[],
                blind=True,
            )

        snr = [self._measurements[slot][band] for band in window]
        hit = [value >= s.detection_threshold for value in snr]
        return Observation(
            slot=slot,
            action=action,
            window=window,
            snr=[float(x) for x in snr],
            hit=[bool(x) for x in hit],
            blind=False,
        )

    def run_slot(self, slot: int, action: int) -> Observation:
        self._current_slot = slot
        return self.step(action)

    @property
    def truth(self) -> Dict[int, Dict[int, float]]:
        return self._truth

    @property
    def events(self) -> Dict[int, List[Tuple[str, int]]]:
        return self._events

    @property
    def measurements(self) -> Dict[int, Dict[int, float]]:
        return self._measurements


# =============================================================================
# Scenario generation
# =============================================================================


def choose_bands(rng: random.Random, n: int, count: int) -> List[int]:
    count = max(1, min(count, n))
    return rng.sample(range(n), count)


def make_scenarios(
    num_bands: int,
    window_size: int,
    slots: int,
    base_seed: int,
) -> List[Scenario]:
    scenarios: List[Scenario] = []

    def add(
        name: str,
        emitters: List[Emitter],
        offset: int,
        *,
        blind: int = 1,
        threshold: float = 5.0,
        noise: float = 1.25,
        agile_duty: float = 0.25,
        prediction_tolerance: int = 1,
        max_predictions: int = 4,
    ) -> None:
        scenarios.append(
            Scenario(
                name=name,
                seed=base_seed + offset,
                num_bands=num_bands,
                window_size=window_size,
                slots=slots,
                emitters=emitters,
                retune_blind_slots=blind,
                detection_threshold=threshold,
                measurement_noise_std=noise,
                agile_duty_cycle=agile_duty,
                prediction_time_tolerance=prediction_tolerance,
                max_predictions_per_slot=max_predictions,
            )
        )

    # Core signal/interval scenarios.
    add(
        "01_fixed_continuous",
        [
            Emitter("fixed-A", "fixed", [2 % num_bands], snr=12),
            Emitter("fixed-B", "fixed", [num_bands // 2], snr=10),
            Emitter("fixed-C", "fixed", [max(0, num_bands - 2)], snr=9),
        ],
        1,
    )
    add(
        "02_fixed_intervals_2_5",
        [
            Emitter("fixed-2", "fixed", [3 % num_bands], interval=2, snr=12),
            Emitter(
                "fixed-5",
                "fixed",
                [max(0, num_bands - 4)],
                interval=5,
                snr=11,
            ),
        ],
        2,
    )
    add(
        "03_periodic_3_10",
        [
            Emitter("periodic-3", "periodic", [5 % num_bands], interval=3, phase=0, snr=13),
            Emitter("periodic-10", "periodic", [15 % num_bands], interval=10, phase=2, snr=13),
        ],
        3,
    )
    add(
        "04_periodic_interval_7",
        [Emitter("periodic-7", "periodic", [num_bands // 3], interval=7, phase=2, snr=13)],
        4,
    )
    add(
        "05_periodic_interval_13",
        [Emitter("periodic-13", "periodic", [2 * num_bands // 3], interval=13, phase=5, snr=14)],
        5,
    )
    add(
        "06_periodic_long_20",
        [Emitter("periodic-20", "periodic", [max(0, num_bands - 6)], interval=20, phase=5, snr=14)],
        6,
    )
    add(
        "07_periodic_long_31",
        [Emitter("periodic-31", "periodic", [num_bands // 2], interval=31, phase=7, snr=14)],
        7,
    )
    add(
        "08_periodic_duration_2",
        [Emitter("periodic-dwell", "periodic", [8 % num_bands], interval=8, phase=1, duration=2, snr=10)],
        8,
    )

    # Hopping/agile families.
    rng = random.Random(base_seed + 9)
    hop = choose_bands(rng, num_bands, min(10, num_bands))
    add(
        "09_hopping_random_timing",
        [Emitter("hopper", "hopping", hop, interval=2, hop_sequence=hop, snr=12)],
        9,
    )

    rng = random.Random(base_seed + 10)
    hop_a = choose_bands(rng, num_bands, min(8, num_bands))
    hop_b = choose_bands(rng, num_bands, min(8, num_bands))
    add(
        "10_two_hoppers",
        [
            Emitter("hopper-A", "hopping", hop_a, interval=2, hop_sequence=hop_a, snr=11),
            Emitter("hopper-B", "hopping", hop_b, interval=3, hop_sequence=hop_b, snr=10),
        ],
        10,
    )

    rng = random.Random(base_seed + 11)
    add(
        "11_agile_low_duty",
        [Emitter("agile-low", "agile", choose_bands(rng, num_bands, min(10, num_bands)), snr=12)],
        11,
        agile_duty=0.15,
    )

    rng = random.Random(base_seed + 12)
    add(
        "12_agile_high_duty",
        [Emitter("agile-high", "agile", choose_bands(rng, num_bands, min(12, num_bands)), snr=11)],
        12,
        agile_duty=0.40,
    )

    # Mixed cases.
    rng = random.Random(base_seed + 13)
    mix_hop = choose_bands(rng, num_bands, min(8, num_bands))
    mix_agile = choose_bands(rng, num_bands, min(8, num_bands))
    add(
        "13_mixed_all_four",
        [
            Emitter("mixed-fixed", "fixed", [1 % num_bands], snr=11),
            Emitter("mixed-periodic", "periodic", [num_bands // 3], interval=7, phase=2, snr=13),
            Emitter("mixed-hopper", "hopping", mix_hop, interval=2, hop_sequence=mix_hop, snr=12),
            Emitter("mixed-agile", "agile", mix_agile, snr=10),
        ],
        13,
    )

    add(
        "14_sparse_long_interval_mix",
        [
            Emitter("sparse-fixed", "fixed", [num_bands // 6], interval=9, snr=12),
            Emitter("sparse-periodic", "periodic", [5 * num_bands // 6], interval=23, phase=4, snr=14),
        ],
        14,
    )

    add(
        "15_mode_change_5_to_11",
        [
            Emitter(
                "changer",
                "periodic",
                [num_bands // 4],
                interval=5,
                phase=1,
                snr=13,
                mode_change_slot=max(1, slots // 2),
                bands_after=[min(num_bands - 1, 3 * num_bands // 4)],
                interval_after=11,
            )
        ],
        15,
    )

    add(
        "16_mode_change_7_to_3_plus_hopper",
        [
            Emitter(
                "period-change",
                "periodic",
                [num_bands // 5],
                interval=7,
                phase=2,
                snr=12,
                mode_change_slot=max(1, slots // 2),
                bands_after=[4 * num_bands // 5],
                interval_after=3,
            ),
            Emitter(
                "background-hopper",
                "hopping",
                [0, max(1, num_bands // 4), max(2, num_bands // 2), max(3, 3 * num_bands // 4)],
                interval=4,
                snr=10,
            ),
        ],
        16,
    )

    add(
        "17_noisy_low_snr",
        [
            Emitter("weak-fixed", "fixed", [num_bands // 4], snr=5.8),
            Emitter("weak-periodic", "periodic", [num_bands // 2], interval=6, phase=1, snr=6.5),
            Emitter("weak-agile", "agile", choose_bands(random.Random(base_seed + 17), num_bands, min(8, num_bands)), snr=5.5),
        ],
        17,
        threshold=5.0,
        noise=2.0,
        agile_duty=0.25,
    )

    # Randomized mixture families. 4 independent seed families x 6 densities.
    density_specs = [
        (1, 1, 1, 1),
        (2, 1, 1, 0),
        (1, 2, 2, 1),
        (2, 2, 2, 2),
        (3, 3, 2, 2),
        (4, 3, 3, 3),
    ]
    scenario_id = 18
    for family in [101, 202, 303, 404]:
        for density_id, (fixed_n, periodic_n, hop_n, agile_n) in enumerate(density_specs, 1):
            rng = random.Random(base_seed + family + density_id * 37)
            used = set()

            def fresh_band() -> int:
                candidates = [b for b in range(num_bands) if b not in used]
                if not candidates:
                    candidates = list(range(num_bands))
                band = rng.choice(candidates)
                used.add(band)
                return band

            intervals = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 17, 23, 29]
            emitters: List[Emitter] = []

            for i in range(fixed_n):
                interval = rng.choice(intervals)
                emitters.append(
                    Emitter(
                        f"rnd-{family}-{density_id}-fixed-{i}",
                        "fixed",
                        [fresh_band()],
                        start=rng.randrange(max(1, min(25, slots))),
                        interval=interval,
                        phase=rng.randrange(interval),
                        duration=rng.choice([1, 1, 2]),
                        snr=rng.uniform(8, 15),
                    )
                )

            for i in range(periodic_n):
                interval = rng.choice(intervals[2:])
                emitters.append(
                    Emitter(
                        f"rnd-{family}-{density_id}-periodic-{i}",
                        "periodic",
                        [fresh_band()],
                        interval=interval,
                        phase=rng.randrange(interval),
                        duration=rng.choice([1, 1, 1, 2]),
                        snr=rng.uniform(8, 16),
                    )
                )

            for i in range(hop_n):
                sequence = choose_bands(
                    rng,
                    num_bands,
                    rng.randint(min(4, num_bands), min(max(6, 4 * window_size), num_bands)),
                )
                emitters.append(
                    Emitter(
                        f"rnd-{family}-{density_id}-hopper-{i}",
                        "hopping",
                        sequence,
                        interval=rng.choice([1, 2, 3, 4, 5]),
                        hop_sequence=sequence,
                        hop_probability=rng.choice([0.15, 0.25, 0.4, 0.6]),
                        snr=rng.uniform(8, 15),
                    )
                )

            for i in range(agile_n):
                active_bands = choose_bands(
                    rng,
                    num_bands,
                    rng.randint(min(5, num_bands), min(max(8, 4 * window_size), num_bands)),
                )
                emitters.append(
                    Emitter(
                        f"rnd-{family}-{density_id}-agile-{i}",
                        "agile",
                        active_bands,
                        snr=rng.uniform(8, 14),
                    )
                )

            add(
                f"{scenario_id:02d}_seed{family}_mix{density_id}",
                emitters,
                scenario_id,
                blind=rng.choice([0, 1, 1, 2]),
                threshold=rng.choice([4.0, 5.0, 6.0]),
                noise=rng.choice([1.0, 1.25, 1.5, 2.0]),
                agile_duty=rng.choice([0.10, 0.20, 0.30, 0.40]),
            )
            scenario_id += 1

    # Hard mixed cases.
    for hard_id in range(1, 7):
        rng = random.Random(base_seed + 5000 + hard_id * 101)
        low = rng.randrange(max(1, num_bands // 3))
        mid = rng.randrange(max(1, num_bands // 3), max(2, 2 * num_bands // 3))
        high = rng.randrange(max(2, 2 * num_bands // 3), num_bands)
        rare = rng.choice([17, 23, 29, 31, 37])
        fast = rng.choice([2, 3, 4, 5])
        hop_sequence = choose_bands(rng, num_bands, min(max(8, 4 * window_size), num_bands))
        agile_bands = choose_bands(rng, num_bands, min(max(8, 3 * window_size), num_bands))

        emitters = [
            Emitter(f"hard-{hard_id}-fixed", "fixed", [low], snr=rng.uniform(10, 14)),
            Emitter(f"hard-{hard_id}-rare", "periodic", [high], interval=rare, phase=rng.randrange(rare), snr=rng.uniform(11, 16)),
            Emitter(f"hard-{hard_id}-fast", "periodic", [mid], interval=fast, phase=rng.randrange(fast), duration=rng.choice([1, 1, 2]), snr=rng.uniform(8, 13)),
            Emitter(f"hard-{hard_id}-hopper", "hopping", hop_sequence, interval=rng.choice([1, 2, 3]), hop_sequence=hop_sequence, hop_probability=rng.choice([0.25, 0.5, 0.75]), snr=rng.uniform(9, 14)),
            Emitter(f"hard-{hard_id}-agile", "agile", agile_bands, snr=rng.uniform(8, 13)),
        ]

        if hard_id % 2 == 0:
            emitters.append(
                Emitter(
                    f"hard-{hard_id}-changer",
                    "periodic",
                    [rng.randrange(num_bands)],
                    interval=rng.choice([5, 7, 11]),
                    phase=rng.randrange(3),
                    snr=rng.uniform(10, 15),
                    mode_change_slot=rng.randint(max(1, slots // 3), max(1, 2 * slots // 3)),
                    bands_after=[rng.randrange(num_bands)],
                    interval_after=rng.choice([3, 6, 10, 13, 17]),
                )
            )

        add(
            f"{scenario_id:02d}_hard_mixed_{hard_id}",
            emitters,
            scenario_id,
            blind=rng.choice([1, 2]),
            threshold=rng.choice([4.0, 5.0, 6.0]),
            noise=rng.choice([1.25, 1.5, 2.0]),
            agile_duty=rng.choice([0.10, 0.20, 0.30, 0.40]),
        )
        scenario_id += 1

    return scenarios


# =============================================================================
# Scoring structures
# =============================================================================


@dataclass
class PredictionRecord:
    made_at_slot: int
    band: int
    predicted_slot: int
    confidence: float
    valid: bool = True
    correct: bool = False
    matched_truth_slot: Optional[int] = None
    time_error: Optional[int] = None


@dataclass
class RunResult:
    submission: str
    scenario: str
    status: str
    error: Optional[str] = None

    # Scenario metadata.
    scenario_seed: Optional[int] = None
    retune_blind_slots: Optional[int] = None
    detection_threshold: Optional[float] = None
    measurement_noise_std: Optional[float] = None
    agile_duty_cycle: Optional[float] = None
    prediction_time_tolerance: Optional[int] = None

    # The 9 SIH-facing metrics.
    probability_detection: float = 0.0
    probability_false_alarm: float = 0.0
    sensitivity_snr: Optional[float] = None
    average_intercept_rate: float = 0.0
    average_reward: float = 0.0
    percentage_correct_predictions: float = 0.0
    interception_ratio: float = 0.0
    average_intercept_time: Optional[float] = None
    average_intercept_time_error: Optional[float] = None

    # Prediction diagnostics.
    prediction_count: int = 0
    correct_prediction_count: int = 0
    prediction_precision: float = 0.0
    prediction_recall: float = 0.0
    prediction_error_median: Optional[float] = None
    prediction_lead_time_mean: Optional[float] = None
    prediction_horizon_mean: Optional[float] = None

    # Detection diagnostics.
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    transmission_opportunities: int = 0
    observed_transmission_band_slots: int = 0
    detected_transmission_band_slots: int = 0
    missed_unobserved_transmission_band_slots: int = 0
    observed_snr_signal_mean: Optional[float] = None
    observed_snr_silence_mean: Optional[float] = None
    sensitivity_threshold_pfa: Optional[float] = None

    # Scheduling diagnostics.
    switches: int = 0
    switch_rate: float = 0.0
    mean_action_jump: float = 0.0
    unique_windows: int = 0
    unique_band_coverage: float = 0.0
    normalized_action_entropy: float = 0.0
    new_window_rate: float = 0.0
    blind_slots: int = 0
    blind_fraction: float = 0.0
    max_blind_streak: int = 0
    mean_revisit_interval: Optional[float] = None
    max_revisit_interval: Optional[float] = None

    # Emitter diagnostics.
    emitter_count: int = 0
    captured_emitter_count: int = 0
    uncaptured_emitter_count: int = 0
    emitter_capture_rate: float = 0.0
    average_intercept_time_censored: Optional[float] = None

    # Reward components.
    reward_hit_component: float = 0.0
    reward_switch_cost: float = 0.0
    reward_urgency_penalty: float = 0.0
    urgency_miss_count: int = 0

    # Timing.
    decisions: int = 0
    decision_time_ms_p50: float = 0.0
    decision_time_ms_p95: float = 0.0
    decision_time_ms_max: float = 0.0
    update_time_ms_p50: float = 0.0
    update_time_ms_p95: float = 0.0
    update_time_ms_max: float = 0.0

    # Detailed logs retained until written to disk.
    prediction_records: List[PredictionRecord] = field(default_factory=list, repr=False)
    slot_records: List[dict] = field(default_factory=list, repr=False)
    emitter_breakdown: List[dict] = field(default_factory=list, repr=False)


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[int(lo)]
    frac = pos - lo
    return xs[int(lo)] * (1 - frac) + xs[int(hi)] * frac


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    clean = list(values)
    return statistics.mean(clean) if clean else default


def sensitivity_at_target_pd(
    observed: Sequence[Tuple[float, bool]],
    target_pd: float = 0.90,
) -> Tuple[Optional[float], Optional[float]]:
    """Return highest SNR threshold reaching target Pd and Pfa there."""
    signal = [snr for snr, is_signal in observed if is_signal]
    silence = [snr for snr, is_signal in observed if not is_signal]
    if not signal:
        return None, None

    candidates = sorted(set(snr for snr, _ in observed), reverse=True)
    for threshold in candidates:
        tp = sum(snr >= threshold for snr in signal)
        pd = tp / len(signal)
        if pd >= target_pd:
            fp = sum(snr >= threshold for snr in silence)
            pfa = fp / len(silence) if silence else 0.0
            return float(threshold), float(pfa)
    return None, None


def truth_band_slots(simulator: RfSimulator) -> Dict[int, set]:
    return {
        slot: set(events.keys())
        for slot, events in (
            (slot, simulator.truth[slot]) for slot in range(simulator.scenario.slots)
        )
    }


def match_predictions(
    predictions: List[PredictionRecord],
    simulator: RfSimulator,
    tolerance: int,
) -> int:
    # Prediction truth is band/time based. Multiple emitters on one band/slot
    # create one observable RF event for this metric.
    truth_events = sorted(
        {
            (slot, band)
            for slot, bands in truth_band_slots(simulator).items()
            for band in bands
        }
    )

    valid = [p for p in predictions if p.valid]
    ordered = sorted(
        valid,
        key=lambda p: (-p.confidence, p.made_at_slot, p.predicted_slot, p.band),
    )
    used = set()
    correct = 0

    for prediction in ordered:
        low = max(
            prediction.made_at_slot,
            prediction.predicted_slot - tolerance,
        )
        high = prediction.predicted_slot + tolerance
        candidates = []
        for truth_slot, truth_band in truth_events:
            if truth_band != prediction.band:
                continue
            if (truth_slot, truth_band) in used:
                continue
            if low <= truth_slot <= high:
                candidates.append(
                    (
                        abs(truth_slot - prediction.predicted_slot),
                        truth_slot,
                        truth_band,
                    )
                )
        if not candidates:
            continue

        _, truth_slot, truth_band = min(candidates)
        used.add((truth_slot, truth_band))
        prediction.correct = True
        prediction.matched_truth_slot = truth_slot
        prediction.time_error = abs(prediction.predicted_slot - truth_slot)
        correct += 1

    return correct


def score_run(
    scenario: Scenario,
    simulator: RfSimulator,
    action_log: List[int],
    observations: List[Observation],
    predictions: List[PredictionRecord],
    decision_times_ms: List[float],
    update_times_ms: List[float],
) -> RunResult:
    result = RunResult(
        submission="",
        scenario=scenario.name,
        status="OK",
        scenario_seed=scenario.seed,
        retune_blind_slots=scenario.retune_blind_slots,
        detection_threshold=scenario.detection_threshold,
        measurement_noise_std=scenario.measurement_noise_std,
        agile_duty_cycle=scenario.agile_duty_cycle,
        prediction_time_tolerance=scenario.prediction_time_tolerance,
        decisions=len(action_log),
    )

    truth = truth_band_slots(simulator)
    total_truth = sum(len(bands) for bands in truth.values())
    result.transmission_opportunities = total_truth

    # Detection / false alarm.
    observed_records: List[Tuple[float, bool]] = []
    signal_snrs: List[float] = []
    silence_snrs: List[float] = []
    prev_snr: Dict[int, float] = {}
    observed_truth = set()
    detected_truth = set()

    for obs in observations:
        if obs.blind:
            continue
        for i, band in enumerate(obs.window):
            snr = float(obs.snr[i])
            hit = bool(obs.hit[i])
            is_signal = band in truth.get(obs.slot, set())
            observed_records.append((snr, is_signal))

            if is_signal:
                observed_truth.add((obs.slot, band))
                signal_snrs.append(snr)
                if hit:
                    result.tp += 1
                    detected_truth.add((obs.slot, band))
                else:
                    result.fn += 1
            else:
                silence_snrs.append(snr)
                if hit:
                    result.fp += 1
                else:
                    result.tn += 1

            old = prev_snr.get(band)
            if (
                old is not None
                and snr - old >= scenario.urgency_delta_snr
                and is_signal
                and not hit
            ):
                result.urgency_miss_count += 1
            prev_snr[band] = snr

    result.observed_transmission_band_slots = len(observed_truth)
    result.detected_transmission_band_slots = len(detected_truth)
    result.missed_unobserved_transmission_band_slots = max(
        0,
        total_truth - len(observed_truth),
    )
    result.probability_detection = (
        result.tp / (result.tp + result.fn)
        if result.tp + result.fn
        else 0.0
    )
    result.probability_false_alarm = (
        result.fp / (result.fp + result.tn)
        if result.fp + result.tn
        else 0.0
    )
    result.observed_snr_signal_mean = mean_or_none(signal_snrs)
    result.observed_snr_silence_mean = mean_or_none(silence_snrs)
    result.sensitivity_snr, result.sensitivity_threshold_pfa = sensitivity_at_target_pd(
        observed_records
    )

    # Interception and emitter capture.
    first_activity: Dict[str, int] = {}
    first_intercept: Dict[str, int] = {}
    intercepted_truth = set()

    for slot, events in simulator.events.items():
        for emitter_name, band in events:
            first_activity.setdefault(emitter_name, slot)

    for obs in observations:
        if obs.blind:
            continue
        selected = set(obs.window)
        for emitter_name, band in simulator.events[obs.slot]:
            if band in selected:
                intercepted_truth.add((obs.slot, band))
                first_intercept.setdefault(emitter_name, obs.slot)

    result.interception_ratio = (
        len(intercepted_truth) / total_truth if total_truth else 0.0
    )
    result.average_intercept_rate = len(intercepted_truth) / max(1, scenario.slots)

    captured_delays: List[float] = []
    censored_delays: List[float] = []
    for emitter_name, activity_slot in first_activity.items():
        if emitter_name in first_intercept:
            delay = first_intercept[emitter_name] - activity_slot
            captured_delays.append(float(delay))
            censored_delays.append(float(delay))
        else:
            censored_delays.append(float(max(0, scenario.slots - 1 - activity_slot)))

    result.average_intercept_time = (
        statistics.mean(captured_delays) if captured_delays else None
    )
    result.average_intercept_time_censored = (
        statistics.mean(censored_delays) if censored_delays else None
    )
    result.emitter_count = len(first_activity)
    result.captured_emitter_count = len(first_intercept)
    result.uncaptured_emitter_count = max(0, result.emitter_count - result.captured_emitter_count)
    result.emitter_capture_rate = (
        result.captured_emitter_count / result.emitter_count
        if result.emitter_count
        else 0.0
    )

    # Prediction metrics.
    correct = match_predictions(
        predictions,
        simulator,
        scenario.prediction_time_tolerance,
    )
    valid_predictions = [p for p in predictions if p.valid]
    errors = [float(p.time_error) for p in predictions if p.correct and p.time_error is not None]
    lead_times = [
        float(p.matched_truth_slot - p.made_at_slot)
        for p in predictions
        if p.correct and p.matched_truth_slot is not None
    ]
    horizons = [float(p.predicted_slot - p.made_at_slot) for p in valid_predictions]

    result.prediction_records = predictions
    result.prediction_count = len(valid_predictions)
    result.correct_prediction_count = correct
    result.percentage_correct_predictions = (
        correct / len(valid_predictions) if valid_predictions else 0.0
    )
    result.prediction_precision = result.percentage_correct_predictions
    result.prediction_recall = (
        correct / total_truth if total_truth else 0.0
    )
    result.average_intercept_time_error = (
        statistics.mean(errors) if errors else None
    )
    result.prediction_error_median = statistics.median(errors) if errors else None
    result.prediction_lead_time_mean = statistics.mean(lead_times) if lead_times else None
    result.prediction_horizon_mean = statistics.mean(horizons) if horizons else None

    # Scheduling diagnostics.
    result.switches = sum(
        action_log[i] != action_log[i - 1]
        for i in range(1, len(action_log))
    )
    result.switch_rate = result.switches / max(1, len(action_log) - 1)
    jumps = [
        abs(action_log[i] - action_log[i - 1])
        for i in range(1, len(action_log))
    ]
    result.mean_action_jump = safe_mean(jumps)
    result.unique_windows = len(set(action_log))
    observed_bands = {
        band
        for action in action_log
        for band in range(action, action + scenario.window_size)
    }
    result.unique_band_coverage = len(observed_bands) / scenario.num_bands

    window_counts = list(
        defaultdict(int, ((action, action_log.count(action)) for action in set(action_log))).values()
    )
    entropy = 0.0
    total_actions = sum(window_counts)
    for count in window_counts:
        if count > 0:
            p = count / total_actions
            entropy -= p * math.log(p)
    max_entropy = math.log(max(1, scenario.num_bands - scenario.window_size + 1))
    result.normalized_action_entropy = entropy / max_entropy if max_entropy else 0.0

    seen_windows = set()
    result.new_window_rate = sum(
        not (action in seen_windows or seen_windows.add(action))
        for action in action_log
    ) / max(1, len(action_log))

    result.blind_slots = sum(obs.blind for obs in observations)
    result.blind_fraction = result.blind_slots / max(1, scenario.slots)
    current_streak = 0
    max_streak = 0
    for obs in observations:
        if obs.blind:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    result.max_blind_streak = max_streak

    per_band_seen: Dict[int, List[int]] = defaultdict(list)
    for obs in observations:
        if not obs.blind:
            for band in obs.window:
                per_band_seen[band].append(obs.slot)
    revisit_intervals = [
        b - a
        for times in per_band_seen.values()
        for a, b in zip(times, times[1:])
    ]
    result.mean_revisit_interval = (
        statistics.mean(revisit_intervals) if revisit_intervals else None
    )
    result.max_revisit_interval = max(revisit_intervals) if revisit_intervals else None

    # Reward: benchmark equivalent of the plan's three-term reward.
    # +1 true positive; -0.05 window switch; -0.25 high-delta observed miss.
    result.reward_hit_component = float(result.tp)
    result.reward_switch_cost = 0.05 * result.switches
    result.reward_urgency_penalty = 0.25 * result.urgency_miss_count
    reward_total = (
        result.reward_hit_component
        - result.reward_switch_cost
        - result.reward_urgency_penalty
    )
    result.average_reward = reward_total / max(1, scenario.slots)

    # Timing.
    result.decision_time_ms_p50 = percentile(decision_times_ms, 0.50)
    result.decision_time_ms_p95 = percentile(decision_times_ms, 0.95)
    result.decision_time_ms_max = max(decision_times_ms) if decision_times_ms else 0.0
    result.update_time_ms_p50 = percentile(update_times_ms, 0.50)
    result.update_time_ms_p95 = percentile(update_times_ms, 0.95)
    result.update_time_ms_max = max(update_times_ms) if update_times_ms else 0.0

    # Slot-level diagnostics.
    for obs in observations:
        truth_bands = truth.get(obs.slot, set())
        selected_truth = len(set(obs.window) & truth_bands) if not obs.blind else 0
        detected_selected = (
            sum(
                band in truth_bands and hit
                for band, hit in zip(obs.window, obs.hit)
            )
            if not obs.blind
            else 0
        )
        made = [p for p in predictions if p.made_at_slot == obs.slot and p.valid]
        matched = [p for p in made if p.correct]
        result.slot_records.append(
            {
                "slot": obs.slot,
                "action": obs.action,
                "window_start": obs.window[0],
                "window_end": obs.window[-1],
                "blind": int(obs.blind),
                "truth_band_count_total": len(truth_bands),
                "selected_truth_band_count": selected_truth,
                "detected_truth_band_count": detected_selected,
                "prediction_count": len(made),
                "correct_prediction_count": len(matched),
                "mean_observed_snr": mean_or_none(obs.snr),
            }
        )

    # Per-emitter diagnostics.
    emitter_map = {e.name: e for e in scenario.emitters}
    for emitter in scenario.emitters:
        events = {
            (slot, band)
            for slot, slot_events in simulator.events.items()
            for name, band in slot_events
            if name == emitter.name
        }
        intercepted = sum(
            (slot, band) in intercepted_truth
            for slot, band in events
        )
        first_activity_slot = min((slot for slot, _ in events), default=None)
        first_intercept_slot = first_intercept.get(emitter.name)
        result.emitter_breakdown.append(
            {
                "emitter_name": emitter.name,
                "emitter_kind": emitter.kind,
                "interval_before": emitter.interval,
                "interval_after": emitter.interval_after,
                "mode_change_slot": emitter.mode_change_slot,
                "truth_band_slots": len(events),
                "intercepted_band_slots": intercepted,
                "interception_ratio": intercepted / len(events) if events else 0.0,
                "first_activity_slot": first_activity_slot,
                "first_intercept_slot": first_intercept_slot,
                "first_intercept_delay": (
                    first_intercept_slot - first_activity_slot
                    if first_intercept_slot is not None and first_activity_slot is not None
                    else None
                ),
            }
        )

    return result


# =============================================================================
# Built-in reference baseline
# =============================================================================


class RoundRobinScheduler(BaseScheduler):
    """Reference sweep that holds each window during its retune/observe cycle."""

    def __init__(self, num_bands: int, window_size: int, config: dict):
        super().__init__(num_bands, window_size, config)
        self.pos = 0
        self.hold = 0
        self.hold_slots = max(1, int(config.get("retune_blind_slots", 0)) + 1)

    def select_action(self, slot: int) -> int:
        return self.pos

    def update(self, obs: Observation) -> None:
        self.hold += 1
        if self.hold >= self.hold_slots:
            self.pos = (
                self.pos + self.window_size
            ) % (self.num_bands - self.window_size + 1)
            self.hold = 0

    def reset(self) -> None:
        self.pos = 0
        self.hold = 0


# =============================================================================
# Submission execution
# =============================================================================


def load_scheduler_class(path: Path):
    module_name = f"_submission_{path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "Scheduler", None)
    if cls is None:
        raise AttributeError("submission must define class Scheduler")
    if not isinstance(cls, type):
        raise TypeError("Scheduler must be a class")
    if not issubclass(cls, BaseScheduler):
        raise TypeError("Scheduler must subclass BaseScheduler")
    return cls


def validate_predictions(
    raw,
    scenario: Scenario,
    slot: int,
) -> List[PredictionRecord]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise TypeError("predict() must return a list or tuple")
    if len(raw) > scenario.max_predictions_per_slot:
        raise ValueError(
            f"predict({slot}) returned {len(raw)} predictions; "
            f"maximum allowed is {scenario.max_predictions_per_slot}"
        )

    result: List[PredictionRecord] = []
    for item in raw:
        if not isinstance(item, Prediction):
            raise TypeError("predict() must return Prediction objects")
        if not 0 <= item.band < scenario.num_bands:
            raise ValueError(f"invalid prediction band {item.band}")
        if item.predicted_slot < slot:
            raise ValueError(
                f"prediction at slot {slot} points to past slot {item.predicted_slot}"
            )
        confidence = float(item.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"prediction confidence must be in [0,1], got {confidence}"
            )
        result.append(
            PredictionRecord(
                made_at_slot=slot,
                band=int(item.band),
                predicted_slot=int(item.predicted_slot),
                confidence=confidence,
            )
        )
    return result


def failed_result(
    submission: str,
    scenario: Scenario,
    status: str,
    error: str,
    decisions: int = 0,
) -> RunResult:
    return RunResult(
        submission=submission,
        scenario=scenario.name,
        status=status,
        error=error,
        scenario_seed=scenario.seed,
        retune_blind_slots=scenario.retune_blind_slots,
        detection_threshold=scenario.detection_threshold,
        measurement_noise_std=scenario.measurement_noise_std,
        agile_duty_cycle=scenario.agile_duty_cycle,
        prediction_time_tolerance=scenario.prediction_time_tolerance,
        decisions=decisions,
    )


def run_scheduler_on_scenario(
    scheduler_cls,
    submission_name: str,
    scenario: Scenario,
    decision_budget_ms: float,
) -> RunResult:
    simulator = RfSimulator(scenario)
    public_config = {
        "retune_blind_slots": scenario.retune_blind_slots,
        "detection_threshold": scenario.detection_threshold,
        "measurement_noise_std": scenario.measurement_noise_std,
        "prediction_time_tolerance": scenario.prediction_time_tolerance,
        "max_predictions_per_slot": scenario.max_predictions_per_slot,
        "urgency_delta_snr": scenario.urgency_delta_snr,
        "slot_duration": 1.0,
    }

    try:
        scheduler = scheduler_cls(
            scenario.num_bands,
            scenario.window_size,
            public_config,
        )
    except Exception as exc:
        return failed_result(
            submission_name,
            scenario,
            "INIT_ERROR",
            f"{type(exc).__name__}: {exc}",
        )

    if not isinstance(scheduler, BaseScheduler):
        return failed_result(
            submission_name,
            scenario,
            "INTERFACE_ERROR",
            "Scheduler instance is not a BaseScheduler",
        )

    try:
        scheduler.reset()
    except Exception as exc:
        return failed_result(
            submission_name,
            scenario,
            "RESET_ERROR",
            f"{type(exc).__name__}: {exc}",
        )

    actions: List[int] = []
    observations: List[Observation] = []
    predictions: List[PredictionRecord] = []
    decision_times: List[float] = []
    update_times: List[float] = []
    max_action = scenario.num_bands - scenario.window_size

    for slot in range(scenario.slots):
        # Prediction and action happen before the environment advances.
        decision_start = time.perf_counter()
        try:
            raw_predictions = scheduler.predict(slot)
            slot_predictions = validate_predictions(
                raw_predictions,
                scenario,
                slot,
            )
            predictions.extend(slot_predictions)
        except Exception as exc:
            return failed_result(
                submission_name,
                scenario,
                "CRASH",
                f"predict({slot}): {type(exc).__name__}: {exc}",
                decisions=len(actions),
            )

        try:
            action = scheduler.select_action(slot)
        except Exception as exc:
            return failed_result(
                submission_name,
                scenario,
                "CRASH",
                f"select_action({slot}): {type(exc).__name__}: {exc}",
                decisions=len(actions),
            )

        elapsed_ms = (time.perf_counter() - decision_start) * 1000.0
        decision_times.append(elapsed_ms)

        if elapsed_ms > decision_budget_ms:
            return failed_result(
                submission_name,
                scenario,
                "TIMEOUT",
                (
                    f"predict()+select_action({slot}) took {elapsed_ms:.3f} ms; "
                    f"limit is {decision_budget_ms:.3f} ms"
                ),
                decisions=len(actions),
            )

        if not isinstance(action, int) or isinstance(action, bool):
            return failed_result(
                submission_name,
                scenario,
                "INVALID_ACTION",
                f"slot {slot}: action must be int, got {type(action).__name__}",
                decisions=len(actions),
            )
        if not 0 <= action <= max_action:
            return failed_result(
                submission_name,
                scenario,
                "INVALID_ACTION",
                f"slot {slot}: action={action}; valid range [0,{max_action}]",
                decisions=len(actions),
            )

        try:
            obs = simulator.run_slot(slot, action)
        except Exception as exc:
            return failed_result(
                submission_name,
                scenario,
                "SIMULATOR_ERROR",
                f"slot {slot}: {type(exc).__name__}: {exc}",
                decisions=len(actions),
            )

        expected_window = list(range(action, action + scenario.window_size))
        if (
            obs.slot != slot
            or obs.action != action
            or obs.window != expected_window
        ):
            return failed_result(
                submission_name,
                scenario,
                "TESTER_ERROR",
                "simulator returned an invalid Observation contract",
                decisions=len(actions),
            )
        if obs.blind:
            if obs.snr != [] or obs.hit != []:
                return failed_result(
                    submission_name,
                    scenario,
                    "TESTER_ERROR",
                    "blind observation must have snr=[] and hit=[]",
                    decisions=len(actions),
                )
        else:
            if len(obs.snr) != scenario.window_size or len(obs.hit) != scenario.window_size:
                return failed_result(
                    submission_name,
                    scenario,
                    "TESTER_ERROR",
                    "non-blind observation has incorrect array lengths",
                    decisions=len(actions),
                )

        update_start = time.perf_counter()
        try:
            scheduler.update(obs)
        except Exception as exc:
            return failed_result(
                submission_name,
                scenario,
                "CRASH",
                f"update(slot={slot}): {type(exc).__name__}: {exc}",
                decisions=len(actions),
            )
        update_times.append((time.perf_counter() - update_start) * 1000.0)

        actions.append(action)
        observations.append(obs)

    result = score_run(
        scenario,
        simulator,
        actions,
        observations,
        predictions,
        decision_times,
        update_times,
    )
    result.submission = submission_name
    return result


def run_scenario_worker(job) -> RunResult:
    """Process-pool worker for one submission/scenario pair.

    The scheduler class itself is not pickled because submission modules are
    dynamically imported. Each worker imports the submission from its path,
    which also keeps submissions isolated from one another.
    """
    submission_path, submission_name, scenario, decision_budget_ms, is_baseline = job
    if is_baseline:
        scheduler_cls = RoundRobinScheduler
    else:
        scheduler_cls = load_scheduler_class(Path(submission_path))
    return run_scheduler_on_scenario(
        scheduler_cls,
        submission_name,
        scenario,
        decision_budget_ms,
    )


# =============================================================================
# Reporting
# =============================================================================


def pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def metric_value(value: Optional[float], width: int = 7) -> str:
    if value is None:
        return "n/a".rjust(width)
    return f"{value:{width}.2f}"


def print_run_result(result: RunResult) -> None:
    if result.status != "OK":
        print(
            f"{result.submission:28s} | {result.scenario:36s} | "
            f"{result.status:14s} | {result.error}"
        )
        return

    print(
        f"{result.submission:28s} | {result.scenario:36s} | "
        f"Pd={pct(result.probability_detection)} | "
        f"Pfa={pct(result.probability_false_alarm)} | "
        f"IR={pct(result.interception_ratio)} | "
        f"AIR={result.average_intercept_rate:6.3f} | "
        f"Pred={pct(result.percentage_correct_predictions)} | "
        f"AIT={metric_value(result.average_intercept_time, 6)} | "
        f"Err={metric_value(result.average_intercept_time_error, 5)} | "
        f"R={result.average_reward:7.3f} | "
        f"P95={result.decision_time_ms_p95:6.2f}ms"
    )


def aggregate_results(results: List[RunResult]) -> List[dict]:
    grouped: Dict[str, List[RunResult]] = defaultdict(list)
    for result in results:
        grouped[result.submission].append(result)

    rows: List[dict] = []
    for submission, runs in grouped.items():
        ok = [r for r in runs if r.status == "OK"]
        row = {
            "submission": submission,
            "status": "PASS" if len(ok) == len(runs) else "FAIL",
            "scenarios": len(runs),
            "successful_scenarios": len(ok),
        }
        if not ok:
            rows.append(row)
            continue

        total_tp = sum(r.tp for r in ok)
        total_fn = sum(r.fn for r in ok)
        total_fp = sum(r.fp for r in ok)
        total_tn = sum(r.tn for r in ok)

        row.update(
            {
                "probability_detection": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
                "probability_false_alarm": total_fp / (total_fp + total_tn) if total_fp + total_tn else 0.0,
                "sensitivity_snr_mean": mean_or_none(r.sensitivity_snr for r in ok),
                "average_intercept_rate": safe_mean(r.average_intercept_rate for r in ok),
                "average_reward": safe_mean(r.average_reward for r in ok),
                "percentage_correct_predictions": safe_mean(r.percentage_correct_predictions for r in ok),
                "interception_ratio": safe_mean(r.interception_ratio for r in ok),
                "average_intercept_time": mean_or_none(r.average_intercept_time for r in ok),
                "average_intercept_time_error": mean_or_none(r.average_intercept_time_error for r in ok),
                "prediction_count": sum(r.prediction_count for r in ok),
                "correct_prediction_count": sum(r.correct_prediction_count for r in ok),
                "prediction_recall_mean": safe_mean(r.prediction_recall for r in ok),
                "emitter_capture_rate": safe_mean(r.emitter_capture_rate for r in ok),
                "switch_rate": safe_mean(r.switch_rate for r in ok),
                "mean_action_jump": safe_mean(r.mean_action_jump for r in ok),
                "unique_band_coverage": safe_mean(r.unique_band_coverage for r in ok),
                "normalized_action_entropy": safe_mean(r.normalized_action_entropy for r in ok),
                "new_window_rate": safe_mean(r.new_window_rate for r in ok),
                "blind_fraction": safe_mean(r.blind_fraction for r in ok),
                "mean_revisit_interval": mean_or_none(r.mean_revisit_interval for r in ok),
                "max_revisit_interval": max(
                    (r.max_revisit_interval or 0.0) for r in ok
                ),
                "prediction_error_median_mean": mean_or_none(r.prediction_error_median for r in ok),
                "prediction_lead_time_mean": mean_or_none(r.prediction_lead_time_mean for r in ok),
                "prediction_horizon_mean": mean_or_none(r.prediction_horizon_mean for r in ok),
                "decision_time_ms_p50": safe_mean(r.decision_time_ms_p50 for r in ok),
                "decision_time_ms_p95": safe_mean(r.decision_time_ms_p95 for r in ok),
                "decision_time_ms_max": max(r.decision_time_ms_max for r in ok),
                "update_time_ms_p95": safe_mean(r.update_time_ms_p95 for r in ok),
                "missed_unobserved_transmission_band_slots": sum(r.missed_unobserved_transmission_band_slots for r in ok),
                "detected_transmission_band_slots": sum(r.detected_transmission_band_slots for r in ok),
                "observed_transmission_band_slots": sum(r.observed_transmission_band_slots for r in ok),
                "transmission_opportunities": sum(r.transmission_opportunities for r in ok),
            }
        )
        rows.append(row)

    rows.sort(key=lambda x: x["submission"])
    return rows


def print_aggregate_table(rows: List[dict]) -> None:
    print("\n" + "=" * 180)
    print("AGGREGATE SIH METRICS")
    print("=" * 180)
    print(
        f"{'Submission':28s} | {'Status':6s} | {'Scen':>4s} | "
        f"{'Pd':>8s} | {'Pfa':>8s} | {'Sens':>8s} | {'AIR':>7s} | "
        f"{'Reward':>8s} | {'Pred%':>8s} | {'IR':>8s} | {'AIT':>7s} | "
        f"{'TimeErr':>8s} | {'P95':>7s}"
    )
    print("-" * 180)
    for row in rows:
        # FAIL means at least one scenario did not complete, not that the
        # successful scenarios are thrown away. Display the partial aggregate
        # whenever at least one scenario succeeded.
        if row["successful_scenarios"] == 0:
            print(
                f"{row['submission']:28s} | {row['status']:6s} | "
                f"{row['successful_scenarios']:4d}/{row['scenarios']:<4d} | "
                f"{'n/a':>8s} | {'n/a':>8s} | {'n/a':>8s} | "
                f"{'n/a':>7s} | {'n/a':>8s} | {'n/a':>8s} | "
                f"{'n/a':>8s} | {'n/a':>7s} | {'n/a':>8s} | {'n/a':>7s}"
            )
            continue
        sens = row["sensitivity_snr_mean"]
        ait = row["average_intercept_time"]
        terr = row["average_intercept_time_error"]
        print(
            f"{row['submission']:28s} | {row['status']:6s} | {row['scenarios']:4d} | "
            f"{pct(row['probability_detection']):>8s} | "
            f"{pct(row['probability_false_alarm']):>8s} | "
            f"{metric_value(sens, 8)} | "
            f"{row['average_intercept_rate']:7.3f} | "
            f"{row['average_reward']:8.3f} | "
            f"{pct(row['percentage_correct_predictions']):>8s} | "
            f"{pct(row['interception_ratio']):>8s} | "
            f"{metric_value(ait, 7)} | "
            f"{metric_value(terr, 8)} | "
            f"{row['decision_time_ms_p95']:7.2f}"
        )


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(results: List[RunResult], path: Path) -> None:
    excluded = {"prediction_records", "slot_records", "emitter_breakdown"}
    fields = [
        name
        for name in RunResult.__dataclass_fields__
        if name not in excluded
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({name: getattr(result, name) for name in fields})


def write_prediction_csv(results: List[RunResult], path: Path) -> None:
    rows = []
    for result in results:
        for prediction in result.prediction_records:
            rows.append(
                {
                    "submission": result.submission,
                    "scenario": result.scenario,
                    "made_at_slot": prediction.made_at_slot,
                    "band": prediction.band,
                    "predicted_slot": prediction.predicted_slot,
                    "confidence": prediction.confidence,
                    "correct": int(prediction.correct),
                    "matched_truth_slot": prediction.matched_truth_slot,
                    "time_error": prediction.time_error,
                }
            )
    write_csv(path, rows)


def write_slot_csv(results: List[RunResult], path: Path) -> None:
    rows = []
    for result in results:
        for slot_row in result.slot_records:
            rows.append({
                "submission": result.submission,
                "scenario": result.scenario,
                **slot_row,
            })
    write_csv(path, rows)


def write_emitter_csv(results: List[RunResult], path: Path) -> None:
    rows = []
    for result in results:
        for emitter_row in result.emitter_breakdown:
            rows.append({
                "submission": result.submission,
                "scenario": result.scenario,
                **emitter_row,
            })
    write_csv(path, rows)


def write_failure_csv(results: List[RunResult], path: Path) -> None:
    rows = []
    for result in results:
        if result.status != "OK":
            rows.append({
                "submission": result.submission,
                "scenario": result.scenario,
                "status": result.status,
                "error": result.error,
                "decisions": result.decisions,
            })
    write_csv(path, rows)


def write_scenario_manifest(scenarios: List[Scenario], path: Path) -> None:
    rows = []
    for scenario in scenarios:
        for emitter in scenario.emitters:
            rows.append(
                {
                    "scenario": scenario.name,
                    "seed": scenario.seed,
                    "num_bands": scenario.num_bands,
                    "window_size": scenario.window_size,
                    "slots": scenario.slots,
                    "retune_blind_slots": scenario.retune_blind_slots,
                    "detection_threshold": scenario.detection_threshold,
                    "measurement_noise_std": scenario.measurement_noise_std,
                    "agile_duty_cycle": scenario.agile_duty_cycle,
                    "prediction_time_tolerance": scenario.prediction_time_tolerance,
                    "max_predictions_per_slot": scenario.max_predictions_per_slot,
                    "emitter_name": emitter.name,
                    "emitter_kind": emitter.kind,
                    "bands": ";".join(map(str, emitter.bands)),
                    "interval": emitter.interval,
                    "phase": emitter.phase,
                    "duration": emitter.duration,
                    "snr": emitter.snr,
                    "mode_change_slot": emitter.mode_change_slot,
                    "bands_after": ";".join(map(str, emitter.bands_after or [])),
                    "interval_after": emitter.interval_after,
                    "hop_sequence": ";".join(map(str, emitter.hop_sequence)),
                    "hop_probability": emitter.hop_probability,
                }
            )
    write_csv(path, rows)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Smart Scan Strategy scheduler submissions."
    )
    parser.add_argument(
        "--submissions",
        type=Path,
        default=Path("submissions"),
        help="Directory containing scheduler_*.py files.",
    )
    parser.add_argument("--slots", type=int, default=1000)
    parser.add_argument("--num-bands", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--timeout-ms",
        type=float,
        default=50.0,
        help="Combined predict()+select_action() wall-clock budget.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help=(
            "Number of worker processes used to evaluate submission/scenario pairs "
            "in parallel. Use --workers 1 for sequential execution."
        ),
    )
    parser.add_argument(
        "--match",
        type=str,
        default="",
        help="Evaluate only discovered filenames containing this text.",
    )
    parser.add_argument(
        "--script",
        dest="scripts",
        action="append",
        default=[],
        metavar="SCRIPT",
        help=(
            "Evaluate one specific script. May be repeated. Accepts a filename, "
            "filename stem, path, or path relative to --submissions."
        ),
    )
    parser.add_argument(
        "--scripts",
        dest="script_list",
        nargs="+",
        default=[],
        metavar="SCRIPT",
        help=(
            "Evaluate a specific set of scripts. Accepts filenames, stems, or paths."
        ),
    )
    parser.add_argument(
        "--list-scripts",
        action="store_true",
        help="List available Python submission scripts and exit without evaluating them.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not include the built-in round-robin reference.",
    )
    parser.add_argument(
        "--with-baseline",
        action="store_true",
        help="Include the built-in round-robin reference. By default, explicit --script/--scripts selection runs only the requested submissions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_output"),
    )
    return parser.parse_args()


def resolve_submission_files(
    submissions_dir: Path,
    explicit_specs: Sequence[str],
) -> List[Path]:
    """Resolve exact submission selectors to Python files.

    A selector may be:
      - an existing path (absolute or relative to the current directory),
      - a filename inside --submissions, or
      - a filename stem inside --submissions.

    Exact selectors are intentionally not substring matches. Use --match when
    substring filtering is desired.
    """
    discovered = sorted(
        p for p in submissions_dir.glob("*.py") if p.is_file()
    )

    resolved: List[Path] = []
    seen = set()

    for spec in explicit_specs:
        raw = Path(spec)
        candidates: List[Path] = []

        if raw.is_file():
            candidates.append(raw.resolve())

        relative = submissions_dir / raw
        if relative.is_file():
            candidates.append(relative.resolve())

        if raw.suffix == "":
            stem_candidate = submissions_dir / f"{spec}.py"
            if stem_candidate.is_file():
                candidates.append(stem_candidate.resolve())

        # Also allow matching an existing discovered filename or stem. This is
        # useful when the filename contains scheduler_ and the caller wants to
        # type only the stem.
        for path in discovered:
            if path.name == spec or path.stem == spec:
                candidates.append(path.resolve())

        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)

        if not unique_candidates:
            raise FileNotFoundError(
                f"could not resolve submission selector '{spec}' in {submissions_dir}"
            )

        for candidate in unique_candidates:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                resolved.append(candidate)

    return sorted(resolved, key=lambda path: path.name.lower())


def main() -> int:
    args = parse_args()

    if args.num_bands <= 0:
        print("ERROR: --num-bands must be > 0")
        return 2
    if not 1 <= args.window_size <= args.num_bands:
        print("ERROR: --window-size must be between 1 and num_bands")
        return 2
    if args.slots <= 0:
        print("ERROR: --slots must be > 0")
        return 2
    if args.timeout_ms <= 0:
        print("ERROR: --timeout-ms must be > 0")
        return 2
    if args.workers <= 0:
        print("ERROR: --workers must be > 0")
        return 2

    if not args.submissions.exists():
        if args.list_scripts:
            print(f"No submissions directory: {args.submissions}")
            return 0
        if args.scripts or args.script_list or args.no_baseline:
            print(f"ERROR: submissions directory does not exist: {args.submissions}")
            return 1

    explicit_specs = [
        *args.scripts,
        *args.script_list,
    ]

    try:
        if explicit_specs:
            submission_files = resolve_submission_files(
                args.submissions,
                explicit_specs,
            )
        else:
            # Preserve the original default behavior: only scheduler_*.py
            # files are treated as submissions when no explicit selector is
            # provided.
            submission_files = sorted(
                p
                for p in args.submissions.glob("scheduler_*.py")
                if p.is_file()
            )
    except (OSError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.match:
        submission_files = [
            p for p in submission_files
            if args.match.lower() in p.name.lower()
        ]

    if args.list_scripts:
        for path in submission_files:
            print(path)
        return 0

    if not submission_files and args.no_baseline:
        print(f"ERROR: no submissions selected from {args.submissions}")
        return 1

    scenarios = make_scenarios(
        args.num_bands,
        args.window_size,
        args.slots,
        args.seed,
    )

    print("=" * 180)
    print("SMART SCAN STRATEGY SUBMISSION EVALUATOR")
    print("=" * 180)
    print(f"Submissions       : {len(submission_files)}")
    if submission_files:
        for path in submission_files:
            print(f"  - {path.name}")
    print(f"Scenarios         : {len(scenarios)}")
    print(f"Bands (N)         : {args.num_bands}")
    print(f"Window (k)        : {args.window_size}")
    print(f"Slots / scenario  : {args.slots}")
    print(f"Base seed         : {args.seed}")
    print(f"Decision budget   : {args.timeout_ms:.3f} ms / predict()+select_action()")
    print(f"Worker processes  : {args.workers}")
    print("SIH metrics       : Pd, Pfa, Sensitivity, AIR, Reward, Prediction%, IR, AIT, InterceptTimeError")
    print()
    print("Scenario suite:")
    for scenario in scenarios:
        print(f"  - {scenario.name} (seed={scenario.seed})")

    all_results: List[RunResult] = []

    # When the user explicitly selects submissions, evaluate only those
    # submissions by default. The baseline is included for the original
    # discovery mode, or when explicitly requested with --with-baseline.
    run_baseline = (not explicit_specs and not args.no_baseline) or args.with_baseline

    jobs = []
    if run_baseline:
        for scenario in scenarios:
            jobs.append(
                (None, "baseline_round_robin", scenario, args.timeout_ms, True)
            )

    submission_classes = {}
    for submission_path in submission_files:
        try:
            # Validate/import once in the parent so a bad submission produces
            # a clean LOAD_ERROR for every scenario instead of killing workers.
            submission_classes[submission_path] = load_scheduler_class(submission_path)
        except Exception as exc:
            for scenario in scenarios:
                result = failed_result(
                    submission_path.stem,
                    scenario,
                    "LOAD_ERROR",
                    f"{type(exc).__name__}: {exc}",
                )
                all_results.append(result)
                print_run_result(result)

    # Only successfully imported submissions are sent to workers. Workers
    # import the module again from its path to keep process state isolated.
    for submission_path in submission_classes:
        for scenario in scenarios:
            jobs.append(
                (
                    str(submission_path),
                    submission_path.stem,
                    scenario,
                    args.timeout_ms,
                    False,
                )
            )

    print(f"\nRunning {len(jobs)} scenario jobs with {min(args.workers, max(1, len(jobs)))} worker process(es)...")

    if jobs:
        max_workers = min(args.workers, len(jobs))
        if max_workers == 1:
            for job in jobs:
                try:
                    result = run_scenario_worker(job)
                except Exception as exc:
                    _, submission_name, scenario, _, _ = job
                    result = failed_result(
                        submission_name,
                        scenario,
                        "WORKER_ERROR",
                        f"{type(exc).__name__}: {exc}",
                    )
                all_results.append(result)
                print_run_result(result)
        else:
            # Scenario runs are independent, so process-level parallelism gives
            # real speedup for Python-heavy schedulers without sharing mutable
            # scheduler state. Results are still aggregated exactly as before.
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_job = {
                    executor.submit(run_scenario_worker, job): job
                    for job in jobs
                }
                for future in concurrent.futures.as_completed(future_to_job):
                    job = future_to_job[future]
                    _, submission_name, scenario, _, _ = job
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = failed_result(
                            submission_name,
                            scenario,
                            "WORKER_ERROR",
                            f"{type(exc).__name__}: {exc}",
                        )
                    all_results.append(result)
                    print_run_result(result)

    aggregate = aggregate_results(all_results)
    print_aggregate_table(aggregate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(all_results, args.output_dir / "evaluation_summary.csv")
    write_csv( args.output_dir / "evaluation_aggregate.csv", aggregate)
    write_prediction_csv(all_results, args.output_dir / "evaluation_predictions.csv")
    write_slot_csv(all_results, args.output_dir / "evaluation_slots.csv")
    write_emitter_csv(all_results, args.output_dir / "evaluation_emitters.csv")
    write_failure_csv(all_results, args.output_dir / "evaluation_failures.csv")
    write_scenario_manifest(scenarios, args.output_dir / "evaluation_scenario_manifest.csv")

    print("\nOutput files:")
    for filename in [
        "evaluation_summary.csv",
        "evaluation_aggregate.csv",
        "evaluation_predictions.csv",
        "evaluation_slots.csv",
        "evaluation_emitters.csv",
        "evaluation_failures.csv",
        "evaluation_scenario_manifest.csv",
    ]:
        print(f"  {args.output_dir / filename}")

    print("\nNote: the SIH statement names nine metrics but does not fully formalize every formula;")
    print("the benchmark conventions are documented in submission_interface_spec.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
