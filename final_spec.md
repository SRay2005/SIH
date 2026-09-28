# Smart Scan Strategy for Electronic Warfare — Problem, Research Freedom & Submission Contract

**Purpose of this document**

This file is intended to be given to an AI agent **fresh, as its primary context**, before asking it to design or implement a scheduler for this problem.

The AI should treat this as a complete working brief:

1. understand the actual SIH/DRDO problem;
2. understand the physical and mathematical objective;
3. understand what the benchmark/evaluator can and cannot observe;
4. freely choose the algorithm, model, mathematical formulation, features, learning method, prediction method, and architecture;
5. write code that conforms exactly to the submission interface so that it can be evaluated reproducibly.

**Important:** The evaluator constraints below are **execution/interface facts**, not restrictions on creativity. They define what the benchmark can measure. They do **not** prescribe a particular ML algorithm or forbid more sophisticated ideas. The AI should explore any technically defensible approach—bandits, Bayesian filtering, hidden-state models, POMDPs, reinforcement learning, planning, tracking, change-point detection, temporal point processes, probabilistic forecasting, hybrid models, analytical policies, or combinations thereof—provided the resulting submission can operate through the exposed observation stream.

---

# 0. The Actual SIH26055 Problem

## 0.1 Official problem

**Problem:** Development of a Smart Scan Strategy for Electronic Warfare in the absence of prior reliable intelligence of emitters and their operating characteristics.

**Organization:** DRDO  
**Problem ID:** SIH26055  
**Category:** Software  
**Theme:** Robotics and Drones

The official statement describes a wideband electronic-support receiver that must search a large frequency spectrum even though its instantaneous receiver bandwidth is substantially narrower than the total spectrum. Existing/open-loop approaches commonly use pre-mission or prior information and sweep the spectrum according to a predetermined schedule. Such a sweep can spend time on non-threatening activity while failing to react quickly enough to newly appearing or important emitters.

The central problem is therefore not simply signal detection. It is **sequential search in frequency and time**: the receiver must decide **where to listen and when to listen there**, while the true emitter state is only partially observable.

The official statement explicitly asks for a simulated RF environment containing truth information about emitter transmission status by frequency band and time slot, a receiver/system model, figures of merit for interception performance, a machine-learning-based robust scheduler trained from hits and misses, prediction of intercept behavior, and treatment of periodic scanning emitters. citeturn0search0

## 0.2 Core engineering problem

Think of the environment as a hidden, time-varying RF world:

```text
                  HIDDEN RF ENVIRONMENT
       ┌──────────────────────────────────────┐
       │ Which emitters are active?            │
       │ Which bands are occupied?             │
       │ When will they transmit next?         │
       │ Are they periodic, hopping, agile,    │
       │ bursty, or changing modes?            │
       └──────────────────┬───────────────────┘
                          │
                          │ partial observation
                          v
                 ┌──────────────────┐
                 │ Narrowband /     │
                 │ limited receiver │
                 └────────┬─────────┘
                          │
                          v
                 ┌──────────────────┐
                 │ YOUR SCHEDULER   │
                 │                  │
                 │ infer state      │
                 │ learn online     │
                 │ predict future   │
                 │ choose next      │
                 │ contiguous window│
                 └────────┬─────────┘
                          │
                          v
                    next sensing action
```

The fundamental optimization problem is:

> **From only the receiver measurements obtained so far, learn and infer enough about the hidden RF environment to choose the next contiguous frequency window and sensing time so that future emitters are intercepted quickly and frequently, while accounting for the cost of retuning and uncertainty.**

This is naturally a **partially observable sequential decision problem**. There is no requirement that it be solved using a particular ML family.

## 0.3 What “smart” means

A useful scheduler should be able to improve over a blind/open-loop scan by learning from experience.

It should be able to discover things such as:

- persistent occupied regions;
- periodic emitters;
- burst durations;
- recurring temporal phases;
- frequency hopping tendencies;
- agile/random activity;
- changes in emitter behavior;
- newly appearing emitters;
- disappearance of previously active emitters;
- uncertainty caused by not observing a band;
- the value of exploring an unobserved region;
- the value of remaining on a known useful window rather than paying a retune cost.

The exact representation of these ideas is deliberately left open.

## 0.4 The two-dimensional nature of interception

Interception is not purely a frequency-selection problem.

A receiver can be tuned to the correct frequency but at the wrong time.

Conversely, it can know approximately when an emitter will transmit but be tuned elsewhere.

The scheduler therefore operates in a joint:

```text
frequency × time
```

search space.

The official problem statement explicitly frames interception as a two-dimensional search problem involving adjustment of receiver frequency at the correct time. citeturn0search0

## 0.5 Required figures of merit from the problem statement

The official problem statement names figures of merit including:

- Probability of Detection;
- Probability of False Alarm;
- Sensitivity;
- Average Intercept Rate;
- Average Reward / Cost Function;
- Percentage of Correct Predictions;
- Average Intercept Time Error;
- interception ratio / intercept performance;
- prediction/intercept-time behavior.

The benchmark below makes these concepts executable and reproducible. The mathematical conventions used by the current evaluator are documented later in this file.

---

# 1. Freedom of Approach

**Do not interpret this document as saying “use LinUCB,” “use Bayesian sensing,” or “use periodicity detection.”**

Those are merely possible approaches.

The AI is explicitly encouraged to reason from first principles and consider alternatives such as:

```text
Bayesian filtering
Hidden Markov Models
state-space models
Kalman / particle filtering
change-point detection
multi-armed bandits
restless bandits
contextual bandits
Bayesian optimization
reinforcement learning
model-based RL
POMDP belief-state planning
Monte Carlo tree search
short-horizon model predictive control
temporal point processes
survival / hazard models
sequence models
online clustering
multi-hypothesis tracking
factor graphs
probabilistic graphical models
analytical periodic inference
hybrid probabilistic + heuristic policies
ensemble methods
information-gain exploration
uncertainty-aware scheduling
```

A hybrid approach is completely valid.

For example, a strong design might maintain several competing hypotheses for an emitter:

```text
H1: persistent
H2: periodic with period 7
H3: periodic with period 13
H4: bursty
H5: hopping
H6: mode changed
```

and use the posterior belief over those hypotheses to select the next window.

Another design might learn a direct value function.

Another might explicitly model the environment and perform short-horizon planning.

**The evaluator does not care which mathematical formulation is chosen. It only cares that the submission obeys the observation boundary and interface and produces useful actions/predictions.**

---

# 2. What the AI Is Actually Being Asked to Build

The deliverable is a Python scheduler that sits between:

```text
simulated RF environment
        ↓
receiver observation
        ↓
YOUR ALGORITHM
        ↓
next receiver action
```

The evaluator owns:

- emitter ground truth;
- RF truth generation;
- receiver noise;
- detection threshold;
- retune/blind behavior;
- scenario generation;
- scoring;
- aggregation;
- timing measurement.

The submission owns:

- state representation;
- learning;
- inference;
- prediction;
- exploration;
- exploitation;
- action selection;
- adaptation;
- recovery after misses;
- model-change detection;
- all algorithmic intelligence.

The submission must **not** implement a second simulator or obtain hidden truth.

---

# 3. Information-Theoretic View

At time `t`, the scheduler should conceptually maintain a belief over hidden environment states:

```text
P(hidden RF state at t | observations up to t-1)
```

The action is a contiguous frequency interval.

The observation is generated by the environment after that action.

This means the scheduler faces:

```text
partial observability
+
action-dependent observations
+
time-varying hidden state
+
retuning cost
+
uncertain future activity
```

The most important consequence is:

> **Not observing a band is not the same as observing that band and seeing silence.**

Any model that ignores this distinction is modeling the wrong problem.

---

# 4. There Is No Required Algorithm

The AI should not start from:

> “Which standard algorithm should I use?”

It should start from:

> “What hidden state would make future transmissions predictable, what evidence lets me infer that state from censored observations, and what action maximizes expected future interception after accounting for switching cost?”

Then choose the simplest or most powerful architecture that answers those questions convincingly.

A good implementation may contain multiple layers:

```text
observation preprocessing
        ↓
online state estimation
        ↓
emitter hypothesis / temporal model
        ↓
uncertainty estimation
        ↓
short-horizon prediction
        ↓
window scoring / planning
        ↓
exploration-exploitation policy
        ↓
contiguous action
```

None of those layers is mandatory.

---

# 5. Important Distinction: Benchmark Facts vs Design Choices

This document contains two kinds of information.

## Benchmark facts

These are properties of the current evaluator and must be obeyed:

```text
interface
call order
observation contents
hidden-information boundary
valid action range
prediction validation
blind-slot semantics
scenario generator
metric definitions
timeout behavior
```

## Design choices

These are entirely yours:

```text
model
features
state representation
learning algorithm
prior structure
posterior representation
prediction method
reward shaping inside your own model
exploration strategy
planning horizon
memory length
discounting
change-point detector
ensemble structure
numerical method
```

Do not confuse the two.

---

# 1. What You Submit

The recommended filename is:

```text
scheduler_<yourname>.py
```

The evaluator loads a class named exactly:

```python
class Scheduler(BaseScheduler):
    ...
```

The evaluator requires:

- `Scheduler` exists.
- `Scheduler` is a class.
- `Scheduler` subclasses the evaluator's `BaseScheduler`.
- `Scheduler` can be instantiated with `(num_bands, window_size, config)`.
- `select_action()` and `update()` work for the scenario stream.

The evaluator does **not** mechanically reject every possible extra helper class or every import-time side effect, but submissions should remain self-contained and side-effect free. Do not rely on behavior that is not part of the contract.

---

# 2. Exact Public Interface

The evaluator exposes this interface to submissions:

```python
from dataclasses import dataclass
from typing import List


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
        return []

    def reset(self) -> None:
        pass
```

You **must** implement:

```python
select_action(slot) -> int
update(obs) -> None
```

`predict(slot)` is optional. Returning no predictions is legal, but it produces zero prediction metrics for the scenario.

---

# 3. Import-Time Behavior

Do not run a simulation, train a model, or execute benchmark logic merely because the evaluator imports the file.

A safe submission looks like:

```python
from base_scheduler import BaseScheduler, Observation, Prediction


class Scheduler(BaseScheduler):
    ...
```

Module-level constants, helper functions, type definitions, and class definitions are fine.

---

# 4. Scenario Lifecycle

The evaluator creates one scheduler instance for a scenario, calls `reset()`, then keeps that same instance alive for all slots in that scenario.

Conceptually:

```python
scheduler.reset()

for slot in range(scenario.slots):
    predictions = scheduler.predict(slot)
    action = scheduler.select_action(slot)
    obs = simulator.run_slot(slot, action)
    scheduler.update(obs)
```

After the scenario ends, the evaluator resets the scheduler before the next independent scenario.

Therefore:

- temporal history can live inside the object;
- learned state can live inside the object;
- periodicity tracking can live inside the object;
- state from one scenario must not be assumed valid for the next scenario.

---

# 5. Exact Information Boundary

At `select_action(t)` and `predict(t)`, the scheduler may use:

```text
CURRENT
- current slot index t
- num_bands
- window_size
- public config

PAST
- all observations received before t
- previous actions
- previous windows
- previous SNR measurements
- previous hit/miss measurements
- prior predictions and internal state derived from them
```

The scheduler must not access:

```text
- simulator truth
- emitter objects
- hidden emitter names
- hidden emitter bands
- hidden emitter phases
- hidden schedules
- future transmission times
- future SNR
- future observations
- scoring-engine internals
```

The evaluator deliberately exposes only the selected receiver window. If a band was not selected, there is no observation for that band in that slot.

---

# 6. The Most Important Observation Rule: Missing Is Not Silent

Suppose:

```text
N = 32
k = 4
selected window = [10, 11, 12, 13]
```

The scheduler gets SNR/hit information only for bands 10–13, when the observation is not blind.

There is **no negative evidence** for bands:

```text
0–9 and 14–31
```

Do not convert an unobserved band into:

```python
hit = False
snr = 0
```

That would manufacture information the receiver did not have.

This is one of the central difficulties of the problem: the history is gappy and censored by the scheduler's own actions.

---

# 7. `slot` Semantics

`slot` is the current simulated slot.

When the evaluator calls:

```python
predict(137)
select_action(137)
```

the scheduler is making decisions **before** slot 137 is simulated.

It may use history through slot 136.

It may not use the observation that will be produced by slot 137.

---

# 8. `select_action()` Semantics

`select_action(slot)` returns the **starting band** of one contiguous receiver window.

For:

```text
num_bands = N
window_size = k
```

the valid action range is:

```text
0 <= action <= N - k
```

The receiver window is always constructed as:

```python
list(range(action, action + window_size))
```

Example:

```text
action = 7
window_size = 4
```

means:

```text
[7, 8, 9, 10]
```

A submission cannot select arbitrary non-contiguous bands.

---

# 9. Retuning and Blind-Time Semantics — READ THIS CAREFULLY

The evaluator's receiver model has **retune blindness only when the requested action changes**.

The logic is effectively:

```python
if last_action is not None and action != last_action:
    blind_remaining = max(blind_remaining, retune_blind_slots)

last_action = action

if blind_remaining > 0:
    blind_remaining -= 1
    return blind_observation
```

Therefore:

### Holding the same window does not create a blind slot

If you select the same action repeatedly:

```text
action 10
10
10
10
```

there is no new retune and therefore no new blind period.

### Changing action creates blindness immediately

With:

```text
retune_blind_slots = 1
```

changing from action 10 to action 20 at slot `t` makes the observation for slot `t` blind.

This is important: the blind slot is not an extra clean observation slot after the switch. The action change itself causes the current simulator step to return the blind observation.

### Repeated switching is expensive

If the scheduler changes action every slot, it can spend most or all of the operation blind.

A strong scheduler should therefore treat switching as a real sensing cost, not simply as a change in an index.

---

# 10. `Observation` Semantics

A normal observation is:

```python
Observation(
    slot=slot,
    action=action,
    window=list(range(action, action + window_size)),
    snr=[...],
    hit=[...],
    blind=False,
)
```

When `blind=False`:

```text
len(snr) == window_size
len(hit) == window_size
```

`snr[i]` and `hit[i]` refer to `window[i]`.

When `blind=True`:

```python
snr == []
hit == []
```

A blind observation means:

> The receiver obtained no RF measurement on that slot.

It does **not** mean that all bands in the window were silent.

A correct update implementation should therefore do something like:

```python
if obs.blind:
    return
```

rather than inserting negative evidence.

---

# 11. Public `config`

The evaluator currently provides public receiver/scoring configuration including:

```text
retune_blind_slots
detection_threshold
measurement_noise_std
prediction_time_tolerance
max_predictions_per_slot
urgency_delta_snr
slot_duration
```

Treat this as a **superset**, not as a frozen dictionary definition. Future evaluator versions may add public parameters.

The config does **not** contain hidden emitter intelligence.

Do not assume that config will contain:

```text
emitter bands
emitter phases
emitter periods
future schedules
hop sequences
truth
```

---

# 12. Prediction API

A prediction has the form:

```python
Prediction(
    band=<band index>,
    predicted_slot=<current or future slot>,
    confidence=<0.0 to 1.0>,
)
```

Example:

```python
return [
    Prediction(
        band=17,
        predicted_slot=142,
        confidence=0.93,
    )
]
```

The evaluator validates every returned prediction.

Required conditions:

```text
0 <= band < num_bands
predicted_slot >= current slot
0.0 <= confidence <= 1.0
```

`predict()` must return a list or tuple. Every item must be a `Prediction` object.

---

# 13. Prediction Count Is a Hard Constraint

The current evaluator has a public:

```text
max_predictions_per_slot
```

The default is 4.

Returning **more than this number is not truncated**.

The evaluator raises an error for that slot and marks the scenario as:

```text
CRASH
```

Therefore this is safe:

```python
predictions = predictions[:self.config["max_predictions_per_slot"]]
```

provided the returned objects are otherwise valid.

Never assume the tester will silently throw away extra predictions.

---

# 14. Prediction Ordering

The evaluator calls:

```text
predict(t)
select_action(t)
simulator
update(obs)
```

So a prediction made at slot `t` cannot use slot-`t` observations.

Predictions are matched to truth independently of whether the receiver was blind at the predicted slot. Truth is generated independently of the scheduler's selected action.

---

# 15. Exact Prediction Matching

For scoring, the evaluator constructs truth events as unique pairs:

```text
(slot, band)
```

It therefore **deduplicates multiple emitters that occupy the same band in the same slot** for the prediction metric.

Predictions are processed in this exact order:

```text
1. descending confidence
2. ascending made_at_slot
3. ascending predicted_slot
4. ascending band
```

For each prediction, the scorer considers truth events on the same band and only accepts events satisfying:

```text
predicted_slot - tolerance <= truth_slot <= predicted_slot + tolerance
```

but the lower bound is clipped to the time the prediction was actually made:

```python
low = max(
    made_at_slot,
    predicted_slot - tolerance,
)
```

The chosen truth event is the eligible event with the smallest absolute time difference to `predicted_slot`.

A truth event can be matched only once.

This means two predictions can compete for the same truth event, and the higher-confidence prediction gets first opportunity to claim it.

---

# 16. Exact Prediction Metrics

For a scenario:

```text
valid_predictions = number of predictions that passed validation
correct_predictions = number successfully matched
truth_events = number of unique (slot, band) truth events
```

The primary prediction metric is:

```text
Percentage of Correct Predictions
= correct_predictions / valid_predictions
```

If:

```text
valid_predictions = 0
```

the evaluator records:

```text
Percentage of Correct Predictions = 0.0
```

There is no undefined/NaN/"excluded" convention here.

The evaluator also reports diagnostic prediction recall:

```text
Prediction Recall
= correct_predictions / truth_events
```

and prediction time error:

```text
Average Prediction Time Error
= mean(abs(predicted_slot - matched_truth_slot))
```

as well as mean prediction lead time and mean prediction horizon.

**Important:** the `RunResult` field called `average_intercept_time_error` is currently populated with this prediction time error. The name is historically misleading.

---

# 17. Why Prediction Volume Is Not Monotonic

The main prediction metric is precision:

```text
precision = correct / valid
```

More predictions create more opportunities to be wrong.

Because each truth event can be matched only once, prediction volume is particularly dangerous on sparse scenarios.

A useful upper bound is:

```text
precision <= min(1, truth_events / valid_predictions)
```

when `valid_predictions > truth_events`.

Therefore:

- do not blindly emit the maximum number of predictions every slot;
- do not assume "more guesses" automatically improves the score;
- calibrate prediction confidence and output volume to expected event density.

The `0 / 0` case is also important: zero predictions and zero correct predictions both produce 0.0 precision. One genuine correct prediction can move the metric above zero.

---

# 18. Receiver / Signal Model

The current simulator constructs hidden RF truth first and then constructs noisy SNR measurements independently for each `(slot, band)`.

Conceptually:

```text
true_snr + Gaussian measurement noise
```

The default scenario parameters include:

```text
measurement_noise_std = 1.25
noise_floor = 0.0
```

The detection rule is:

```python
hit = measured_snr >= detection_threshold
```

The scheduler receives only measurements for its selected window and only when the observation is not blind.

---

# 19. Emitter Semantics in the Current Evaluator

The current evaluator recognizes four emitter kinds:

```text
fixed
periodic
hopping
agile
```

The names are worth reading carefully because the code, not the English label, defines their behavior.

## `fixed`

`fixed` emitters use the same interval/duration activation rule as `periodic` emitters:

```python
offset = slot - start - phase
active if offset >= 0 and offset % interval < duration
```

The current implementation does **not** treat `fixed` as synonymous with "always on."

For example:

```python
Emitter("fixed-2", "fixed", [3], interval=2)
```

is active every 2 slots.

A continuous fixed emitter occurs when the interval is 1, such as the benchmark's `01_fixed_continuous` case.

## `periodic`

`periodic` uses the same interval/duration activation mechanism.

Therefore, in this evaluator, the core distinction between `fixed` and `periodic` is primarily the scenario/emitter label; their temporal truth-generation branch is the same.

## `duration`

A duration greater than 1 creates a burst/run of consecutive active slots.

For example:

```text
interval = 8
duration = 2
```

produces two consecutive active slots every cycle.

## `hopping`

Hopping is randomized in both timing and band.

At each slot, the emitter draws a Bernoulli event using `hop_probability` when supplied, otherwise approximately:

```text
1 / interval
```

When a hop occurs, one band is chosen from the hopping sequence.

Therefore hopping is not a deterministic periodic phase process in the current evaluator.

## `agile`

Agile activity is Bernoulli with the scenario's public `agile_duty_cycle`.

When active, one band is randomly selected from the emitter's band set.

This means agile emitters should not be modeled as deterministic periodic transmitters.

## Mode changes

Emitters may change bands, interval, and duration at a specified `mode_change_slot`.

The new parameters apply when:

```text
slot >= mode_change_slot
```

A good adaptive scheduler must therefore handle regime changes instead of assuming one stationary pattern forever.

---

# 20. Current Benchmark Scenario Families

With the evaluator's default parameters:

```text
N = 32 bands
k = 4 band window
slots = 1000
base seed = 20260925
```

the scenario generator currently produces **47 scenarios**:

| IDs | Family | Purpose |
|---|---|---|
| 01–17 | Core deterministic/noisy cases | Fixed, interval, periodic, hopping, agile, mixtures, mode changes, noisy low-SNR |
| 18–41 | 4 × 6 randomized mixture cases | Different emitter densities and random seeds |
| 42–47 | Hard mixed cases | Rare intervals, fast periodic emitters, hopper, agile emitter, and some mode changers |

Representative core scenarios include:

```text
01_fixed_continuous
02_fixed_intervals_2_5
03_periodic_3_10
04_periodic_interval_7
05_periodic_interval_13
06_periodic_long_20
07_periodic_long_31
08_periodic_duration_2
09_hopping_random_timing
10_two_hoppers
11_agile_low_duty
12_agile_high_duty
13_mixed_all_four
14_sparse_long_interval_mix
15_mode_change_5_to_11
16_mode_change_7_to_3_plus_hopper
17_noisy_low_snr
```

The randomized families deliberately vary:

```text
- fixed emitter count
- periodic emitter count
- hopping emitter count
- agile emitter count
- start times
- bands
- intervals
- phases
- durations
- SNR
- hopping probability
- retune blind time
- detection threshold
- measurement noise
- agile duty cycle
```

The hard mixed family adds rare periods chosen from values such as 17, 23, 29, 31 and 37 and faster periodic emitters with intervals such as 2–5.

Do not optimize against one emitter family only.

---

# 21. Interception Is Not the Same as Detection

This is one of the most important scoring distinctions.

The evaluator's interception logic is based on **window coverage**, not receiver `hit`.

A truth event is considered intercepted when:

```text
- the observation for that slot is not blind, and
- the emitter's band is inside the selected window
```

The measured SNR does not need to exceed the detection threshold for the event to count as intercepted.

So:

```text
in-window weak signal
```

can count as an interception even if:

```text
hit == False
```

This means the benchmark has two distinct questions:

```text
Did the scheduler look at the right place?
Did the receiver successfully detect what it looked at?
```

The main Interception Ratio is primarily about the first question.

---

# 22. Exact Interception Ratio

The evaluator builds unique truth events:

```text
(slot, band)
```

Then it scans non-blind observations and counts a truth event as intercepted when its band is inside the selected window.

Therefore:

```text
Interception Ratio
= intercepted_truth_events / total_truth_events
```

where `total_truth_events` counts unique `(slot, band)` pairs.

Truth events in blind slots are not intercepted because the evaluator skips blind observations during interception scoring.

This is different from prediction matching, where truth generation is independent of receiver blindness.

---

# 23. Average Intercept Rate

The current evaluator defines:

```text
Average Intercept Rate
= intercepted_truth_events / scenario.slots
```

It is therefore an event-per-slot quantity in this benchmark.

---

# 24. Average Intercept Time

The current evaluator computes first-intercept delay **from first activity to first window coverage for each emitter**.

For an emitter:

```text
activity_slot = first slot where that emitter is active
intercept_slot = first non-blind observation where its active band is in the selected window
```

then:

```text
delay = intercept_slot - activity_slot
```

The ordinary `Average Intercept Time` averages delays only over emitters that were actually captured.

The evaluator also computes a censored variant internally for diagnostics, which assigns an end-of-scenario censoring delay to uncaptured emitters.

Do not interpret Average Intercept Time as "time until the receiver declares hit=True." It is based on first window coverage.

---

# 25. Probability of Detection (Pd)

For every non-blind observation, the evaluator checks whether an observed band is truly active.

If the band is active:

```text
hit=True  -> TP
hit=False -> FN
```

The metric is:

```text
Pd = TP / (TP + FN)
```

Important consequence:

> Pd is conditional on observing a transmitting band.

A scheduler can have high Pd but poor Interception Ratio because it may rarely tune to the transmitting band in the first place.

---

# 26. Probability of False Alarm (Pfa)

For non-transmitting bands that are actually observed:

```text
hit=True  -> FP
hit=False -> TN
```

The metric is:

```text
Pfa = FP / (FP + TN)
```

Unobserved bands are not treated as false alarms or true negatives because there is no measurement for them.

---

# 27. Sensitivity Definition in This Benchmark

The current evaluator's sensitivity calculation searches observed SNR values for the **highest threshold** that achieves:

```text
Pd >= 0.90
```

It then reports the Pfa at that threshold.

Thus the benchmark's sensitivity is effectively:

```text
SNR threshold at target Pd = 0.90
```

not a conventional "Pd at a fixed Pfa" curve summary.

This is a benchmark convention, not a claim that it is the only possible RF/radar sensitivity definition.

---

# 28. Exact Reward / Cost Function

The current evaluator implements the three-term reward idea as:

```text
reward_total
    = TP
    - 0.05 * switches
    - 0.25 * urgency_miss_count
```

and reports:

```text
Average Reward
    = reward_total / scenario.slots
```

Where:

```text
switches
= number of adjacent action pairs that differ
```

and an `urgency_miss` occurs when, for the same band, the observed SNR increases by at least:

```text
urgency_delta_snr
```

from the previous observed sample, the band is truly transmitting, and the scheduler still has:

```text
hit=False
```

This reward is an evaluator metric, not additional information delivered to the scheduler.

---

# 29. Prediction / Interception Metric Trap

A model that optimizes only prediction precision can still schedule badly.

A model that optimizes only Pd can also schedule badly.

For this benchmark, the roles are roughly:

```text
Interception Ratio
    Did the receiver cover truth events?

Pd / Pfa
    How well did the receiver classify what it actually observed?

Prediction precision
    How often did explicit predictions match truth?

Average Intercept Time
    How quickly did the receiver first cover each emitter?
```

These are complementary, not interchangeable.

---

# 30. Aggregate Metrics — Important Detail

Aggregation is **not the same for every metric**.

For a submission, if every scenario succeeds:

### Pooled detection metrics

The evaluator pools counts across successful scenarios for:

```text
Pd
Pfa
TP
FN
FP
TN
```

### Mean-over-scenario metrics

The evaluator takes the arithmetic mean across successful scenarios for metrics such as:

```text
Average Intercept Rate
Average Reward
Percentage of Correct Predictions
Interception Ratio
Average Intercept Time
Average Intercept Time Error
Prediction Recall diagnostics
Emitter capture rate
switch rate
coverage diagnostics
latency diagnostics
```

Therefore one scenario contributes one scenario-level value to these averages, regardless of how many truth events it contains.

### Failure behavior

The evaluator distinguishes **benchmark completeness** from **metric availability**.

If one or more scenarios fail:

```text
status = FAIL
```

but the evaluator still aggregates metrics over the scenarios that completed successfully.

For example:

```text
47 total scenarios
45 successful
2 failed

status = FAIL
aggregate metrics = mean/pool over the 45 successful scenarios
```

If all scenarios fail, there is no valid scenario-level data from which to compute an aggregate and the relevant aggregate values are reported as unavailable.

Therefore an AI should not treat one failed scenario as making all useful evaluation information disappear.

---

# 31. Additional Diagnostics

The evaluator records significantly more information than the nine SIH-facing metrics.

Scheduling diagnostics include:

```text
switches
switch_rate
mean_action_jump
unique_windows
unique_band_coverage
normalized_action_entropy
new_window_rate
blind_slots
blind_fraction
max_blind_streak
mean_revisit_interval
max_revisit_interval
```

Prediction diagnostics include:

```text
prediction_count
correct_prediction_count
prediction_precision
prediction_recall
prediction_error_median
prediction_lead_time_mean
prediction_horizon_mean
```

Detection diagnostics include:

```text
observed transmissions
detected transmissions
missed transmissions while observed
transmissions missed because they were unobserved
TP / FN / FP / TN
mean observed signal SNR
mean observed silence SNR
sensitivity threshold
```

Emitter diagnostics include:

```text
emitter_count
captured_emitter_count
uncaptured_emitter_count
emitter_capture_rate
per-emitter intercept behavior
```

Latency diagnostics include:

```text
decision p50 / p95 / max
update p50 / p95 / max
```

These are outputs of evaluation, not inputs to the scheduler.

---

# 32. Runtime / Latency Contract

The current evaluator defaults to:

```text
50 ms per slot
```

for the combined:

```text
predict(slot) + select_action(slot)
```

The CLI option is:

```text
--timeout-ms
```

The default is:

```text
50.0
```

The evaluator measures elapsed wall-clock time using `time.perf_counter()` **around the two calls** and checks the elapsed time after they return.

It does not preempt the Python call at the exact deadline. A call that returns after the budget is marked:

```text
TIMEOUT
```

and that scenario fails.

`update(obs)` is timed separately for diagnostics but is not included in the 50 ms decision deadline.

For real submissions, optimize both:

```text
algorithmic quality
and
worst-case latency stability
```

Do not design something that occasionally performs a huge computation every few hundred slots.

---

# 33. CPU / Network / File Constraints

Submissions should assume:

```text
CPU only
no CUDA requirement
no network
no runtime downloads
no external dataset
no hidden-file access
```

Do not inspect:

```text
scenario manifest
truth logs
evaluation CSVs
other submissions
simulator internals
```

at runtime.

The fact that the evaluator itself writes an emitter manifest does not make that manifest a legitimate scheduler input.

---

# 34. Scenario Manifest Is Evaluation Output, Not Scheduler Input

The evaluator writes:

```text
evaluation_summary.csv
evaluation_aggregate.csv
evaluation_predictions.csv
evaluation_slots.csv
evaluation_emitters.csv
evaluation_failures.csv
evaluation_scenario_manifest.csv
```

The scenario manifest intentionally contains emitter metadata for evaluator analysis.

A submission must not use it as a source of prior intelligence.

The scheduler should behave as though it only receives `config` plus its own observation stream.

---

# 35. Evaluator Failure Modes

A run can fail for reasons such as:

```text
LOAD_ERROR
INIT_ERROR
RESET_ERROR
CRASH
TIMEOUT
INVALID_ACTION
SIMULATOR_ERROR
TESTER_ERROR
```

Common submission mistakes include:

```text
- wrong action type
- action outside [0, N-k]
- returning too many predictions
- prediction points into the past
- prediction confidence outside [0,1]
- returning objects that are not Prediction instances
- assuming blind == silence
- assuming missing band == miss
- failing to reset state
- expensive computation in predict/select
```

---

# 36. CLI — Evaluating the Whole Submission Directory

Default behavior:

```bash
python evaluate_submissions.py
```

By default the evaluator scans:

```text
submissions/
```

for files matching:

```text
scheduler_*.py
```

Example:

```text
submissions/
    scheduler_alice.py
    scheduler_bob.py
    scheduler_linucb.py
```

By default the built-in round-robin baseline is also evaluated.

Use:

```bash
python evaluate_submissions.py --no-baseline
```

to omit it.

---

# 37. CLI — Evaluating Specific Scripts

The evaluator now supports exact script selection.

## One script

```bash
python evaluate_submissions.py \
    --submissions ./submissions \
    --script scheduler_alice.py \
    --no-baseline
```

## Multiple scripts using repeated `--script`

```bash
python evaluate_submissions.py \
    --submissions ./submissions \
    --script scheduler_alice.py \
    --script scheduler_bob.py \
    --no-baseline
```

## Multiple scripts using `--scripts`

```bash
python evaluate_submissions.py \
    --submissions ./submissions \
    --scripts scheduler_alice.py scheduler_bob.py \
    --no-baseline
```

## Filename stem

A `.py` suffix is optional:

```bash
python evaluate_submissions.py \
    --submissions ./submissions \
    --scripts scheduler_alice scheduler_bob \
    --no-baseline
```

## Exact path

An explicit script may also be supplied as an existing path:

```bash
python evaluate_submissions.py \
    --script ./submissions/scheduler_alice.py \
    --no-baseline
```

Explicit selectors are exact selections, not substring searches.

If a selector cannot be resolved, the evaluator prints an error and exits instead of silently running a different file.

---

# 38. CLI — `--match`

The older substring filter remains available:

```bash
python evaluate_submissions.py \
    --match linucb
```

This filters the discovered/selected filenames by substring.

It can also be combined with explicit selection when useful.

Use exact `--script` / `--scripts` when you know which files you want.
Use `--match` when you intentionally want substring filtering.

---

# 39. CLI — List Available Scripts

To see which default submission files would be discovered without running them:

```bash
python evaluate_submissions.py --list-scripts
```

The list operation performs no scenario evaluation.

---

# 40. CLI — Useful Evaluation Controls

Common controls:

```text
--submissions PATH
--scripts SCRIPT [SCRIPT ...]
--script SCRIPT
--match TEXT
--no-baseline
--list-scripts
--slots INTEGER
--num-bands INTEGER
--window-size INTEGER
--seed INTEGER
--timeout-ms FLOAT
--output-dir PATH
```

Example:

```bash
python evaluate_submissions.py \
    --submissions ./submissions \
    --scripts scheduler_v7.py scheduler_v8.py \
    --no-baseline \
    --slots 2000 \
    --seed 20260925 \
    --timeout-ms 50 \
    --output-dir evaluation_output_v8
```

Use a different `--output-dir` when you want to preserve results from multiple experiments.

---

# 41. Reference Baseline

The built-in round-robin baseline:

- starts from action 0;
- holds each window for `retune_blind_slots + 1` updates;
- then moves by `window_size`;
- wraps over legal actions;
- does not make predictions;
- does not learn RF behavior.

It is a reference point, not an oracle.

The important baseline lesson is that **dwell time matters** because changing action causes blindness while holding the same action does not.

---

# 42. Designing a Strong Scheduler for This Evaluator

A robust strategy should reason in terms of hidden-state inference rather than simply assigning a permanent score to each band.

Useful internal concepts include:

```text
- belief that a band is active
- recent hit/miss evidence
- persistence
- time since last informative observation
- SNR trend / emergence
- periodic interval hypotheses
- periodic phase hypotheses
- burst duration hypotheses
- mode-change detection
- cross-band transition structure
- uncertainty / information value
- retune cost
```

But these are suggestions, not hidden evaluator features.

The scheduler must construct them from its own observation history.

---

# 43. Partial Observability Has a Critical Consequence

If a band is not observed for 50 slots, that does **not** mean:

```text
50 misses
```

It means:

```text
50 slots of unknown state
```

A good scheduler should therefore distinguish:

```text
observed and silent
```

from:

```text
not observed
```

This matters especially for sparse periodic emitters.

---

# 44. Periodicity Should Be Learned From Observed Evidence

For interval-gated emitters, the observed hit timestamps can reveal:

```text
period
phase
burst duration
```

But the scheduler sees only samples obtained while it was tuned to the relevant band.

Therefore a periodicity detector must account for:

```text
- skipped cycles
- censored observations
- missed pulses
- mode changes
- burst duration
```

A sequence such as:

```text
hit at 5
hit at 25
hit at 45
```

is evidence for an interval near 20, but a missing observation is not automatically evidence that the emitter was silent.

---

# 45. Hopping and Agile Need Different Reasoning

The current simulator deliberately makes:

```text
hopping -> randomized timing + randomized band
agile   -> Bernoulli activity + randomized band
```

So a scheduler that assumes every emitter is periodic can become overconfident and repeatedly search for a phase that does not exist.

For these cases, useful concepts may instead be:

```text
- spatial/band occupancy distribution
- recency
- transition likelihoods
- exploration
- information gain
```

---

# 46. Retune Policy Is Part of the Optimization Problem

Changing windows has a sensing cost because the changed action produces blindness.

Therefore a decision is not simply:

```text
pick the band with maximum predicted probability
```

It is closer to:

```text
pick a contiguous window whose expected future sensing value
justifies the cost of getting there
```

This is why dwell time, hysteresis, and finite-horizon planning can matter even when the RF model is good.

---

# 47. Rare Emitters Create a Discovery vs. Exploitation Tradeoff

The benchmark contains both:

```text
high-rate / easy-to-capture activity
```

and:

```text
sparse long-period activity
```

If the scheduler locks permanently onto the highest-yield window, it can starve rare emitters.

If it explores forever, it wastes the opportunity to exploit a strong learned pattern.

A good strategy should therefore distinguish:

```text
cold start
learning
exploitation
recovery after a miss
mode change
background exploration
```

---

# 48. Do Not Optimize Only for Pd

Because:

```text
Pd = TP / (TP + FN)
```

and TP/FN are calculated only for bands that were actually observed while transmitting, a scheduler can achieve excellent Pd simply by being very accurate whenever it happens to look at a transmitting band.

That does not guarantee a high Interception Ratio.

For this problem, **where and when you look** is a first-class part of performance.

---

# 49. Do Not Optimize Only for Prediction Precision

Prediction precision is:

```text
correct / valid
```

so an extremely conservative scheduler can look artificially clean while producing little useful predictive coverage.

Conversely, a scheduler can predict aggressively and destroy its own precision.

Prediction logic should therefore be calibrated separately from action scheduling.

---

# 50. Correct Development Workflow

Recommended order:

```text
1. Implement and test the exact interface.
2. Handle blind slots correctly.
3. Maintain observation history without filling missing bands as misses.
4. Build a simple robust coverage baseline.
5. Add temporal inference.
6. Add mode-change recovery.
7. Add exploitation/exploration control.
8. Add prediction only after the underlying temporal model works.
9. Measure decision latency.
10. Test the full 47-scenario suite.
11. Inspect failure CSVs and per-scenario metrics.
12. Only then tune hyperparameters.
```

Use selective evaluation while iterating:

```bash
python evaluate_submissions.py \
    --submissions ./submissions \
    --scripts scheduler_my_latest.py \
    --no-baseline \
    --slots 300
```

Then use the complete suite before final submission:

```bash
python evaluate_submissions.py \
    --submissions ./submissions \
    --scripts scheduler_my_latest.py \
    --no-baseline \
    --slots 1000
```

---

# 51. Minimal Valid Submission

```python
"""
Team member: example
Strategy: round-robin baseline
Version: v0
"""

from base_scheduler import BaseScheduler, Observation


class Scheduler(BaseScheduler):

    def __init__(self, num_bands, window_size, config):
        super().__init__(num_bands, window_size, config)
        self.pos = 0

    def select_action(self, slot: int) -> int:
        return self.pos

    def update(self, obs: Observation) -> None:
        self.pos = (
            self.pos + self.window_size
        ) % (
            self.num_bands
            - self.window_size
            + 1
        )

    def reset(self) -> None:
        self.pos = 0
```

This is only an interface example. It does not attempt to solve the learning problem.

---

# 52. Final Per-Slot Contract

For every slot `t`, think of the contract as exactly:

```text
1. Current time t is known.
2. Only information from previous slots is available.
3. Optionally emit zero or more Prediction objects.
4. Choose one contiguous window start.
5. If the action changed, the resulting observation may be blind.
6. Otherwise receive SNR/hit measurements for that window.
7. Update internal state using the observation.
8. Repeat.
```

The central optimization problem is:

> Use a narrow, partially observable receiver to learn enough structure from past measurements to choose the next contiguous window that maximizes useful future interception while paying the real cost of retuning.

---

# 53. Source-of-Truth Rule

When this document and `evaluate_submissions.py` ever disagree, the executable evaluator is the source of truth for what the benchmark actually does.

When developing a new strategy, read both:

```text
evaluate_submissions.py
submission_interface_spec.md
```

The spec explains the contract and the evaluator code defines the exact implementation.

---

# 54. AI Implementation Guidance — How to Work on This Problem

When asked to “write a scheduler,” do not immediately produce the first familiar bandit implementation.

Use this reasoning sequence.

## Step 1 — Understand the observation process

Write down exactly:

```text
what is hidden
what is observed
when it becomes observed
what is missing
what action controls future observations
```

In particular:

```text
unobserved != silent
blind != silent
hit != ground truth
```

## Step 2 — Identify the latent variables

Ask what hidden state actually determines future transmission.

Examples:

```text
active/inactive state
period
phase
burst duration
frequency set
transition probabilities
duty cycle
emitter identity
mode
mode-change time
```

You do not need to use all of these. Use whatever your model needs.

## Step 3 — Separate inference from scheduling

A useful architecture often separates:

```text
“What do I believe is happening?”
```

from:

```text
“Where should I look next?”
```

This makes it easier to diagnose whether poor performance comes from:

- bad inference;
- bad prediction;
- poor exploration;
- excessive switching;
- poor window selection.

## Step 4 — Treat prediction and sensing as related but distinct objectives

The evaluator's prediction metric is precision:

```text
correct predictions / valid predictions
```

while Interception Ratio asks whether the receiver actually covered truth events.

Do not assume that maximizing one automatically maximizes the other.

A scheduler can:

```text
predict extremely accurately but rarely look there
```

or:

```text
look very effectively but make poor explicit predictions
```

The strongest systems should reason about both.

## Step 5 — Account for retuning

An action change can make the current observation blind.

Therefore action selection should consider:

```text
expected sensing value of new window
-
cost of reaching new window
```

Do not evaluate a candidate window as though switching were free.

## Step 6 — Handle regime changes

A model that learns:

```text
band 17, period 13
```

must eventually be able to admit:

```text
that hypothesis is no longer reliable
```

when the evidence changes.

Good designs should have some mechanism for:

```text
stale-model detection
confidence decay
change-point detection
posterior reset
hypothesis competition
```

if appropriate.

## Step 7 — Think about exploration explicitly

A scheduler that exploits one strong emitter forever may miss rare emitters.

A scheduler that explores uniformly forever wastes time.

A useful policy should reason about:

```text
expected reward
uncertainty
information gain
rare-event opportunity
retune cost
future value
```

rather than using arbitrary exploration percentages without justification.

---

# 55. Code Structure Expected from an AI Submission

A clean submission normally looks like:

```python
from base_scheduler import BaseScheduler, Observation, Prediction


class Scheduler(BaseScheduler):

    def __init__(self, num_bands, window_size, config):
        super().__init__(num_bands, window_size, config)

        # Initialize all persistent state here.
        ...

    def predict(self, slot):
        # Use only current time + information available before this slot.
        ...
        return predictions

    def select_action(self, slot):
        # Return the starting band of one contiguous window.
        ...
        return action

    def update(self, obs):
        # Update the model using this observation only.
        ...

    def reset(self):
        # Restore clean state for a new independent scenario.
        ...
```

Helper classes are allowed.

Helper functions are allowed.

Multiple internal models are allowed.

NumPy is allowed.

Other installed Python packages may be used if declared appropriately for the evaluation environment.

The key requirement is that the evaluator can instantiate:

```python
Scheduler(num_bands, window_size, config)
```

and call the required methods.

---

# 56. What the AI Should Never Do

Never obtain information from:

```text
simulator internals
scenario objects
truth arrays
emitter definitions
scenario manifests
evaluation CSVs
other submissions
future observations
future SNR
future hit values
```

Never infer that:

```text
not observed => miss
```

Never run benchmark logic at import time.

Never allow invalid actions or predictions to escape the public interface.

Never assume the scheduler gets observations for the whole spectrum.

Never assume the environment is stationary forever.

Never assume every emitter is periodic.

Never assume prediction volume is always beneficial.

Never optimize only one metric without checking the others.

---

# 57. What the AI Should Test While Developing

Before considering a strategy complete, test at least these conceptual cases:

```text
1. Continuous fixed emitter
2. Periodic emitter with short period
3. Periodic emitter with long period
4. Burst emitter
5. Hopping emitter
6. Agile emitter
7. Multiple simultaneous emitters
8. Very sparse emitters
9. Low-SNR emitter
10. Mid-run mode change
11. Newly appearing emitter
12. Previously active emitter disappearing
13. Blind slot caused by switching
14. Long periods without observing a band
15. Window-size edge case
16. Cold start
```

The current evaluator already contains a broad version of these behaviors, but isolated local tests are useful for debugging.

---

# 58. How to Judge Whether a New Idea Is Actually Better

Do not declare an algorithm better merely because one metric improved.

Inspect:

```text
Interception Ratio
Average Intercept Time
Average Intercept Rate
Pd
Pfa
Prediction Precision
Prediction Recall
Average Reward
switch rate
blind fraction
latency
per-scenario performance
```

A particularly important diagnostic question is:

> Did the improvement survive across emitter families, or did it only exploit one simulator pattern?

A strategy that gains 10 percentage points on periodic emitters but collapses on hopping/agile emitters may not be an actual general improvement.

---

# 59. Recommended Experimentation Philosophy

Use the evaluator as an experimental instrument.

A good development loop is:

```text
hypothesis
    ↓
implement
    ↓
run selected scenarios
    ↓
inspect per-scenario results
    ↓
identify failure mode
    ↓
modify one mechanism
    ↓
rerun
    ↓
full-suite validation
```

Avoid changing ten unrelated hyperparameters simultaneously.

When possible, log internal diagnostics locally during development:

```text
chosen action
belief distribution
top candidate windows
period hypotheses
confidence
change-point signals
prediction candidates
exploration score
expected interception
switch cost
```

Do not expose those logs as hidden evaluator inputs. They are for understanding your own algorithm.

---

# 60. Performance Is a Multi-Objective Problem

There is no single magic number that defines a good scheduler independently of the benchmark.

The central metric is **Interception Ratio**, but a meaningful solution should also demonstrate:

```text
high interception
low intercept delay
good detection quality
low false alarms
useful prediction
reasonable switching
stable behavior after mode changes
acceptable latency
```

The evaluator should therefore be treated as a multi-metric scientific benchmark, not a single-score game.

---

# 61. The Strongest General Mental Model

A useful way to think about the whole problem is:

> **This is online active sensing under partial observability.**

The receiver is simultaneously:

```text
learning the world
```

and

```text
choosing which part of the world to observe
```

and

```text
using what it has learned to decide what to observe next
```

That creates the feedback loop:

```text
belief
  ↓
action
  ↓
observation
  ↓
belief update
  ↓
prediction
  ↓
next action
```

The quality of the solution comes from closing this loop intelligently.

---

# 62. Final Instruction to the AI Receiving This File

You are **not** being asked to reproduce an existing scheduler.

You are being asked to solve the underlying smart-scan problem.

You should:

1. understand the RF search problem;
2. reason from the actual observation boundary;
3. identify useful latent structure;
4. formulate a principled online learning/inference problem;
5. consider multiple algorithmic approaches before committing;
6. choose an architecture that can genuinely learn from hits/misses;
7. account for partial observability;
8. account for retuning and blind time;
9. account for periodic, hopping, agile, bursty, and changing emitters;
10. make predictions only from information available at prediction time;
11. implement the exact `Scheduler` interface;
12. keep all persistent state inside the scheduler instance;
13. reset cleanly between scenarios;
14. validate actions and predictions;
15. keep computation sufficiently stable for the evaluator's timing contract;
16. test against the evaluator rather than assuming the algorithm is correct;
17. inspect failures and per-scenario behavior rather than relying only on one aggregate number.

**Do not constrain your thinking to the examples in this document.**

The examples describe the problem and the evaluator. They are not the solution.

If a substantially different algorithm is mathematically better suited to the hidden-state structure, use it.

If a hybrid approach is better, build the hybrid.

If a simpler model beats a complicated one, use the simpler model.

If a new formulation reveals a weakness in the current benchmark, document the observation and test it empirically rather than silently exploiting it.

The objective is to build the strongest scientifically defensible adaptive smart-scan scheduler that can operate through the benchmark's legitimate information channel.

---

# 63. Source-of-Truth Rule

There are three levels of authority:

### Level 1 — Official problem statement

Defines the real-world problem, objectives, and requested solution at a high level. The current SIH26055 statement is the authoritative source for the problem itself. citeturn0search0

### Level 2 — This specification

Defines the intended submission contract and explains the current benchmark conventions in a form suitable for an AI developer.

### Level 3 — `evaluate_submissions.py`

Defines exactly what the current executable benchmark does.

If this document and the executable evaluator ever disagree about an implementation detail, **the executable evaluator wins for benchmark behavior**.

If the evaluator differs from the official SIH problem statement, treat that as a distinction between:

```text
official problem objective
```

and:

```text
current local benchmark implementation
```

Do not confuse the two.

---

# 64. Final One-Paragraph Brief

Build an online machine-learning-based Electronic Support receiver scheduler for SIH26055. The environment contains hidden, time-varying emitters distributed across a wide frequency spectrum. The receiver can observe only a contiguous limited-bandwidth window at a time, and changing that window can cause blind time. Starting without reliable prior emitter intelligence, the scheduler must learn from its own observations, infer temporal and frequency structure, predict useful future activity where possible, and continually decide where to sense next. It must perform well across persistent, periodic, bursty, hopping, agile, mixed, noisy, and changing emitter behavior. The central objective is to maximize interception ratio and minimize intercept time while maintaining useful detection, false-alarm, sensitivity, reward, and prediction performance. The algorithm is deliberately open-ended: choose whatever principled online inference, learning, planning, probabilistic, or hybrid approach best solves the problem. The only non-negotiable requirement is that the implementation obey the evaluator's legitimate information boundary and Python submission interface so that every strategy can be tested fairly against the same environment.

