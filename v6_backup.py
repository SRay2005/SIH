"""
Team member: smartscan
Strategy: Universal Multi-Signal Bayesian & Periodic Tracker with Calibrated Predictions
Version: v6

Key design principles:
1. Universal Prediction Engine across ALL Signal Types:
   - Tier 1 (Continuous): Near 100% duty cycle, high confidence.
   - Tier 2 (Periodic): Deterministic phase and interval tracking with negative-evidence verification.
   - Tier 3 (Hopping & Agile): Tracks the empirical active hopping support set. When no deterministic
     signals are due, emits calibrated predictions on high-density hopping channels spaced by at
     least 3 slots to avoid duplicate self-competition under +/-1 tolerance.
   - Ensures non-zero, calibrated predictions across pure hopping, pure agile, and mixed scenarios.

2. Timing & Density-Aware Scheduling:
   - Proactive retune: switches to windows before periodic pulses arrive.
   - High-density dwell: targets windows that maximize active-channel overlap for hopping/agile signals.
   - Continuous discovery: systematic exploration for unseen and quiet bands to catch mode changes.
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import List, Optional
from base_scheduler import BaseScheduler, Observation, Prediction


class Scheduler(BaseScheduler):

    def __init__(self, num_bands: int, window_size: int, config: dict):
        super().__init__(num_bands, window_size, config)
        self.max_action = num_bands - window_size
        self.blind_cost = int(config.get("retune_blind_slots", 1))
        self.max_preds = int(config.get("max_predictions_per_slot", 4))
        self.det_thresh = float(config.get("detection_threshold", 5.0))
        self._init()

    def _init(self):
        N = self.num_bands
        # Observation and hit tracking
        self.obs_count = [0] * N
        self.hit_count = [0] * N
        self.hits = [[] for _ in range(N)]            # List of hit slots
        self.silences = [[] for _ in range(N)]        # List of observed silent slots
        self.last_seen = [-1] * N
        self.last_hit = [-1] * N
        self.snr_ema = [0.0] * N

        # Continuous emitter tracking
        self.is_continuous = [False] * N

        # Periodic emitter models
        self.period = [0] * N
        self.phase = [0] * N
        self.duration = [1] * N
        self.pconf = [0.0] * N

        # Hopping & Agile stochastic tracking
        self.last_stochastic_pred = [-100] * N

        # Scheduling state
        self.cur_action = 0
        self.dwell_time = 0
        self.slot = 0

    def reset(self):
        self._init()

    # -------------------------------------------------------------------------
    # Predictions (Universal across all signal types)
    # -------------------------------------------------------------------------
    def predict(self, slot: int) -> List[Prediction]:
        self.slot = slot
        if slot < 20:
            return []

        candidates = []
        deterministic_count = 0

        # Tier 1 & Tier 2: Deterministic Signals
        for b in range(self.num_bands):
            # 1. Continuous emitter (active on every slot)
            if self.is_continuous[b]:
                candidates.append((0.98, b, slot))
                deterministic_count += 1
                continue

            # 2. Confident periodic emitter
            P = self.period[b]
            if P >= 2 and self.pconf[b] >= 0.70:
                ph = self.phase[b]
                dur = self.duration[b]
                offset = (slot - ph) % P
                if offset < dur:
                    candidates.append((self.pconf[b], b, slot))
                    deterministic_count += 1

        # Tier 3: Stochastic Hopping & Agile Signals
        # If deterministic signals haven't exhausted our prediction budget:
        # Predict on active hopping/agile bands that have not been predicted recently.
        # Spacing of >= 3 slots prevents duplicate self-competition under tolerance = 1.
        has_deterministic_scene = (
            any(self.is_continuous) or any(self.period[b] >= 2 and self.pconf[b] >= 0.70 for b in range(self.num_bands))
        )

        # In pure hopping/agile scenarios, or when budget remains:
        if deterministic_count < self.max_preds:
            stochastic_candidates = []
            for b in range(self.num_bands):
                if self.is_continuous[b] or (self.period[b] >= 2 and self.pconf[b] >= 0.70):
                    continue
                # Must be an established active channel with recent activity
                if self.hit_count[b] >= 3 and self.obs_count[b] >= 6:
                    rate = self.hit_count[b] / self.obs_count[b]
                    # In pure hopping/agile scenarios, predict on top active bands
                    # In mixed scenarios, only predict if channel has high empirical rate
                    min_rate = 0.20 if has_deterministic_scene else 0.08
                    if rate >= min_rate:
                        time_since_pred = slot - self.last_stochastic_pred[b]
                        if time_since_pred >= 3:
                            # Calibrated confidence based on rate
                            conf = min(0.48, max(0.20, rate * 0.7))
                            stochastic_candidates.append((conf, b, slot))

            # Sort stochastic candidates by rate descending
            stochastic_candidates.sort(key=lambda x: -x[0])

            # In mixed scenarios, limit stochastic guesses to 1; in pure scenarios, up to 2
            stochastic_limit = 1 if has_deterministic_scene else 2
            for conf, b, ps in stochastic_candidates[:stochastic_limit]:
                if len(candidates) < self.max_preds:
                    candidates.append((conf, b, ps))
                    self.last_stochastic_pred[b] = slot

        # Sort descending by confidence and cap to max_preds
        candidates.sort(key=lambda x: -x[0])
        results = []
        for conf, band, ps in candidates[:self.max_preds]:
            results.append(Prediction(
                band=band,
                predicted_slot=ps,
                confidence=float(conf),
            ))
        return results

    # -------------------------------------------------------------------------
    # Sensing / Action Selection
    # -------------------------------------------------------------------------
    def select_action(self, slot: int) -> int:
        self.slot = slot
        N, W = self.num_bands, self.window_size

        # Cold start sweep: ensure every band is visited quickly
        dwell_per_window = max(2, self.blind_cost + 1)
        num_windows = (N + W - 1) // W
        cold_start_end = num_windows * dwell_per_window
        if slot < cold_start_end:
            idx = (slot // dwell_per_window) % num_windows
            action = min(idx * W, self.max_action)
            self.cur_action = action
            return action

        # Score all possible receiver windows
        best_action = self.cur_action
        best_score = -1e9

        for action in range(self.max_action + 1):
            window_score = 0.0
            for band in range(action, action + W):
                window_score += self._score_band(band, action, slot)

            # Transition and dwell costs
            if action == self.cur_action:
                # Bonus for staying (avoid retune blindness)
                stay_bonus = 1.2 * (1.0 + self.blind_cost)
                if self.dwell_time > 15:
                    stay_bonus -= 0.1 * (self.dwell_time - 15)
                window_score += stay_bonus
            else:
                # Switching cost (losing blind_cost slots)
                window_score -= 1.0 * self.blind_cost

            if window_score > best_score:
                best_score = window_score
                best_action = action

        if best_action == self.cur_action:
            self.dwell_time += 1
        else:
            self.dwell_time = 0
            self.cur_action = best_action

        return best_action

    def _score_band(self, band: int, action: int, slot: int) -> float:
        val = 0.0
        is_current_window = (action == self.cur_action)

        # 1. Proactive Periodic Pulse Interception
        P = self.period[band]
        if P >= 2 and self.pconf[band] >= 0.5:
            ph = self.phase[band]
            dur = self.duration[band]

            if is_current_window:
                if (slot - ph) % P < dur:
                    val += 5.0 * self.pconf[band]
                elif ((slot + 1) - ph) % P < dur:
                    val += 3.5 * self.pconf[band]
            else:
                arrival_slot = slot + self.blind_cost
                if (arrival_slot - ph) % P < dur:
                    val += 4.5 * self.pconf[band]
                elif (arrival_slot + 1 - ph) % P < dur:
                    val += 3.0 * self.pconf[band]

        # 2. Continuous Fixed Emitters
        if self.is_continuous[band]:
            val += 3.0

        # 3. Channel Activity Rate (vital for clustering hopping/agile channels)
        if self.obs_count[band] >= 5:
            rate = self.hit_count[band] / self.obs_count[band]
            val += rate * 2.5

        # 4. Exploration Bonus (scaled with time unseen)
        if self.last_seen[band] < 0:
            time_unseen = slot + 1
        else:
            time_unseen = slot - self.last_seen[band]

        explore = 0.06 * math.sqrt(time_unseen)
        if self.hit_count[band] > 0 and time_unseen > 12:
            explore += 0.08 * math.sqrt(time_unseen)
        val += min(1.8, explore)

        return val

    # -------------------------------------------------------------------------
    # Observation Updates & Model Fitting
    # -------------------------------------------------------------------------
    def update(self, obs: Observation) -> None:
        slot = obs.slot
        if obs.blind:
            return

        for i, band in enumerate(obs.window):
            self.obs_count[band] += 1
            self.last_seen[band] = slot

            snr = obs.snr[i]
            self.snr_ema[band] = 0.2 * snr + 0.8 * self.snr_ema[band]

            if obs.hit[i]:
                self.hit_count[band] += 1
                self.last_hit[band] = slot
                self.hits[band].append(slot)
                if len(self.hits[band]) > 50:
                    self.hits[band].pop(0)
            else:
                self.silences[band].append(slot)
                if len(self.silences[band]) > 80:
                    self.silences[band].pop(0)

        # Update Continuous Emitter Status
        for band in obs.window:
            if self.obs_count[band] >= 6:
                rate = self.hit_count[band] / self.obs_count[band]
                if rate >= 0.88 and self.snr_ema[band] >= self.det_thresh:
                    self.is_continuous[band] = True
                    self.period[band] = 1
                    self.pconf[band] = 0.95
                else:
                    self.is_continuous[band] = False

        # Periodicity Detection (run periodically)
        if slot > 25 and slot % 10 == 0:
            for band in range(self.num_bands):
                if not self.is_continuous[band] and len(self.hits[band]) >= 3:
                    self._fit_periodic_model(band)

    def _fit_periodic_model(self, band: int) -> None:
        recent_hits = self.hits[band][-25:]
        n_hits = len(recent_hits)
        if n_hits < 3:
            return

        silences = set(self.silences[band][-60:])
        best_p = 0
        best_ph = 0
        best_dur = 1
        best_score = 0.0

        for P in range(2, 45):
            counts = defaultdict(int)
            for h in recent_hits:
                counts[h % P] += 1

            ph = max(counts, key=counts.get)
            match_1 = counts[ph]

            # Check if duration is 2
            ph_next = (ph + 1) % P
            match_2 = counts[ph_next]
            if match_2 >= 2 and (match_1 + match_2) / n_hits >= 0.80:
                dur = 2
                matching_hits = match_1 + match_2
            else:
                dur = 1
                matching_hits = match_1

            hit_ratio = matching_hits / n_hits
            if hit_ratio < 0.75:
                continue

            # Check negative evidence: observed silences on the active phase
            silence_contradictions = 0
            for s in silences:
                offset = (s - ph) % P
                if offset < dur:
                    silence_contradictions += 1

            if silence_contradictions > 1:
                continue

            conf = hit_ratio * 0.7 + (matching_hits / (matching_hits + silence_contradictions + 0.1)) * 0.3

            if conf > best_score:
                best_score = conf
                best_p = P
                best_ph = ph
                best_dur = dur

        if best_p >= 2 and best_score >= 0.70:
            self.period[band] = best_p
            self.phase[band] = best_ph
            self.duration[band] = best_dur
            self.pconf[band] = best_score
        elif self.period[band] >= 2:
            self.pconf[band] *= 0.92
            if self.pconf[band] < 0.40:
                self.period[band] = 0
                self.pconf[band] = 0.0
