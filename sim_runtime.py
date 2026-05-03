import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import optuna
import optuna.visualization as vis
from collections import Counter
from dataclasses import replace

import sim_scenarios
from sim_dataclasses import *

# --------------------------------------------------------------------------------------------------
# Running instructions 
#   RUN - python sim_runtime.py 
# --------------------------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")

# ==================================================================================================
# Main processes
# ==================================================================================================

def producer(state: ProducerState, simulation_state: SimulationState, sim_config: SimConfig, sim_time: float) -> None:
    """Run a producer process that generates the specified item type and places them into a queue."""
    process = sim_config.processes[state.item_type]
    output_type = process.producer.output
    base_production_time = process.producer.production_time

    if sim_time < state.next_ready_time:
        return

    queue_occupancy = simulation_state.queues[output_type]
    queue_capacity = sim_config.processes[output_type].queue_capacity
    if queue_occupancy >= queue_capacity:
        return
    
    if sim_config.use_feedback:
        next_production_time = compute_feedback_time(base_production_time, process.producer.reference_signal,
            process.producer.proportional_gain, process.producer.transport_lag, sim_config, simulation_state,
            None, output_type, sim_time, state, process.producer.integral_gain, state.last_update_time)
    else:
        next_production_time = base_production_time

    finish_time = sim_time + next_production_time
    simulation_state.pending_outputs.append((finish_time, output_type))
    state.next_ready_time = finish_time

def consumer(state: ConsumerState, simulation_state: SimulationState, sim_config: SimConfig, sim_time: float) -> None:
    """Run a consumer process that retrieves and processes items from the provided queue."""
    process = sim_config.processes[state.item_type]
    input_type = process.consumer.input
    output_type = process.consumer.output
    consumption_time = process.consumer.consumption_time

    if is_machine_failed(state.item_type, sim_time, simulation_state):
        return

    # abort if not ready or nothing to consume
    if sim_time < state.next_ready_time or simulation_state.queues[input_type] <= 0:
        return
    
    # if consumption produces an output, consumption only occures when the output queue has space
    if output_type is not None:
        if simulation_state.queues[output_type] >= sim_config.processes[output_type].queue_capacity:
            return
        
    if sim_config.use_feedback:
        adjusted_consumption_time = compute_feedback_time(consumption_time, process.consumer.reference_signal,
            process.consumer.proportional_gain, process.consumer.transport_lag, sim_config, simulation_state,
            input_type, output_type, sim_time, state, process.consumer.integral_gain, state.last_update_time)
    else:
        adjusted_consumption_time = consumption_time

    # consume immediately
    simulation_state.queues[input_type] -= 1
    simulation_state.queue_history[input_type].append((sim_time, simulation_state.queues[input_type]))
    simulation_state.consumer_logs.append(SimulationLogs(state.process_id, state.item_type.value, sim_time))

    # schedule output after processing time
    state.next_ready_time = sim_time + adjusted_consumption_time

    if output_type is not None:
        # output will appear when the process finishes
        simulation_state.pending_outputs.append((state.next_ready_time, output_type))

def get_queue_occupancy(history: list[tuple[float, int]], current_time: float, delay: float) -> int:
    """Returns the occupancy of the queue exactly 'delay' seconds ago."""
    target_time = current_time - delay
    # Assume empty before the delay period has passed
    if target_time <= 0.0:
        return history[0][1]
    
    for timestamp, occupancy in reversed(history):
        if timestamp <= target_time:
            return occupancy
    return 0

def calculate_control_effort(base_time: float, reference_signal: int, proportional_gain: float, 
    transport_lag: float, queue_history: list[tuple[float, int]], sim_time: float, current_queue: int,
    integral_gain: float, state: ProducerState | ConsumerState, last_time: float, override_error=None) -> float:
    """Calculates the adjusted processing time using decoupled PI feedback and Dead-Time."""
    
    # 1. HANDLE DELAY INDEPENDENTLY
    if override_error is not None:
        control_error = override_error
    else:
        # Check if Feedback Delay (Dead-Time) is enabled
        if transport_lag is None or transport_lag <= 0:
            queue_value = current_queue
        else:
            queue_value = get_queue_occupancy(queue_history, sim_time, transport_lag)

        control_error = reference_signal - queue_value

    dt = min(max(sim_time - last_time, 1e-6), 1.0)

    # 2. HANDLE INTEGRAL (PI) INDEPENDENTLY
    if integral_gain > 0:
        # PI Control: Accumulate error over time
        state.error_integral += control_error * dt
        control_variable = (proportional_gain * control_error) + (integral_gain * state.error_integral)
    else:
        # Pure P Control: Reset integral to prevent stale data buildup, only use proportional
        state.error_integral = 0.0
        control_variable = proportional_gain * control_error

    state.last_update_time = sim_time

    # Apply the bounded duration multiplier (actuation)
    time_multiplier = math.exp(-math.atan(control_variable))
    
    return base_time * time_multiplier

def compute_feedback_time(base_time, target, sensitivity, delay, sim_config, simulation_state, 
    input_type, output_type, sim_time, state, damping, last_update_time):

    if target is None:
        return base_time

    if sim_config.feedback_type == FeedbackType.OUTPUT:
        if output_type is None:
            return base_time

        queue_history = simulation_state.queue_history[output_type]
        current_queue = simulation_state.queues[output_type]

        return calculate_control_effort(
            base_time, target, sensitivity, delay, queue_history, sim_time, current_queue, damping, state, last_update_time)

    elif sim_config.feedback_type == FeedbackType.INPUT:
        if input_type is None:
            return base_time
        queue_history = simulation_state.queue_history[input_type]
        current_queue = simulation_state.queues[input_type]
        return calculate_control_effort(base_time, target, sensitivity, delay, queue_history, sim_time, current_queue, damping, state, last_update_time)

    elif sim_config.feedback_type == FeedbackType.DUAL:
        if input_type is None or output_type is None:
            return base_time

        input_q = simulation_state.queues[input_type]
        output_q = simulation_state.queues[output_type]

        # simplest: average error
        combined_error = (
            (target - input_q) +
            (target - output_q)
        ) * 0.5

        return calculate_control_effort(
            base_time, target, sensitivity, delay, None, sim_time, None, damping, state, last_update_time, combined_error
        )

# ==================================================================================================
# Reporting - logs and diagrams
# ==================================================================================================

def log_simulation_parameters(sim_config: SimConfig) -> None:
    """Log the parameters the simulation is using."""
    logging.info(f"The simulation will be running for {sim_config.simulation_timeout_in_seconds} seconds.")
    logging.info(f"Feedback enabled: {sim_config.use_feedback}")

    for item_type, process_config in sim_config.processes.items():
        config_info = f"The {item_type.value} process has - queue occupancy: {process_config.queue_capacity}"
        if process_config.producer.count > 0:
            config_info += (
                f" | producer/s count: {process_config.producer.count}"
                f" | production time(s): {process_config.producer.production_time}"
            )
            if sim_config.use_feedback:
                config_info += (
                    f" | target queue: {process_config.producer.reference_signal}"
                    f" | sensitivity: {process_config.producer.proportional_gain}"
                    f" | delay: {process_config.producer.transport_lag}"
                )
        if process_config.consumer.count > 0:
            config_info += (
                f" | consumer/s count: {process_config.consumer.count}"
                f" | consumption time(s): {process_config.consumer.consumption_time}"
            )
            if sim_config.use_feedback:
                config_info += (
                    f" | target queue: {process_config.consumer.reference_signal}"
                    f" | sensitivity: {process_config.consumer.proportional_gain}"
                    f" | delay: {process_config.consumer.transport_lag}"
                )
        logging.info(config_info)

def log_results(simulation_state: SimulationState) -> None:
    """Log a summary of total produced and consumed items."""
    produced_items = Counter(log.item_type for log in simulation_state.producer_logs)
    consumed_items = Counter(log.item_type for log in simulation_state.consumer_logs)

    logging.info("Simulation summary:")
    all_items = sorted(set(produced_items) | set(consumed_items))
    for item in all_items:
        logging.info(
            f"Item: {item} - produced: {produced_items.get(item, 0)} | consumed: {consumed_items.get(item, 0)}"
        )

def plot_producer_consumer_rates(ax: plt.Axes, start_time: float, producer_logs: list[SimulationLogs],
                                 consumer_logs: list[SimulationLogs], bucket_size: float = 1.0) -> None:
    item_types = {log.item_type for log in producer_logs + consumer_logs}
    ordered_items = [item.value for item in ItemType if item.value in item_types]
    all_times = [log.timestamp - start_time for log in list(producer_logs) + list(consumer_logs)]
    max_t = max(all_times + [0])
    bins = np.arange(0, max_t + bucket_size, bucket_size)
    bin_centres = bins[:-1] + bucket_size / 2

    def extract_times(logs: list[SimulationLogs], item: str) -> list[float]:
        return [log.timestamp - start_time for log in logs if log.item_type == item]

    for item in ordered_items:
        prod_times = extract_times(producer_logs, item)
        cons_times = extract_times(consumer_logs, item)
        prod_counts, _ = np.histogram(prod_times, bins=bins)
        cons_counts, _ = np.histogram(cons_times, bins=bins)

        if len(prod_times) > 0:
            throughput = prod_counts
        else:
            throughput = cons_counts
        
        ax.plot(bin_centres, throughput, "--", markersize=4, alpha=0.9, label=f"Throughput: {item}")
    
    ax.set(
        xlabel="Time (seconds)",
        ylabel="Items per second",
        title=f"Process Throughput Rates",
    )
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend()

def plot_queue_occupancy_over_time(ax: plt.Axes, start_time: float, queue_logs: list[QueueLogs], shocks=None) -> None:
    queues: dict[str, list[QueueLogs]] = {}

    for log in queue_logs:
        queues.setdefault(log.queue_name, []).append(log)

    ordered_names = [item.value for item in ItemType if item.value in queues]
    for queue_name in ordered_names:
        logs = queues[queue_name]

        time_steps = np.array([log.timestamp - start_time for log in logs])
        queue_usages = np.array([log.queue_usage for log in logs])
        # plot line and then shade below
        line, = ax.plot(time_steps, queue_usages, label=queue_name)
        ax.fill_between(time_steps, queue_usages, alpha=0.15, color=line.get_color())

    if shocks:
        for shock in shocks:
            ax.axvline(shock.start_time, linestyle="--", color="red", alpha=0.8)
            ax.axvline(shock.end_time, linestyle="--", color="red", alpha=0.8)
            ax.axvspan(shock.start_time, shock.end_time, color="red", alpha=0.15, label="Shock")
    ax.set(
        xlabel="Time (seconds)",
        ylabel="Queue occupancy",
        title="Queue State Dynamics",
    )
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend()

def plot_results(simulation_state: SimulationState, start_time: float = 0.0, shocks=None) -> None:
    """Create one figure containing subplots."""
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    plot_producer_consumer_rates(ax1, start_time, simulation_state.producer_logs, simulation_state.consumer_logs)
    plot_queue_occupancy_over_time(ax2, start_time, simulation_state.queue_logs, shocks)

    plt.tight_layout()
    plt.show()

# ==================================================================================================
# Simulation runners
# ==================================================================================================

def create_simulation_state(sim_config: SimConfig) -> SimulationState:
    """Create and return a SimulationState dataclass with initialized shared resources."""
    queues = {}
    queue_history = {}
    for item_type in sim_config.processes:
        initial_occupancy = sim_config.initial_queue_occupancy.get(item_type, 0)
        queues[item_type] = initial_occupancy
        queue_history[item_type] = [(0.0, initial_occupancy)]

    return SimulationState(
        producer_logs=[],
        consumer_logs=[],
        queue_logs=[],
        queues=queues,
        queue_history=queue_history,
        pending_outputs=[],
        shocks=[]
    )

def create_producer_consumer_states(sim_config: SimConfig) -> tuple[list[ProducerState], list[ConsumerState]]:
    """Create and return producer and consumer data stores."""
    producers = []
    consumers = []

    for item_type, process in sim_config.processes.items():
        for i in range(process.producer.count):
            producers.append(ProducerState(process_id=i, item_type=item_type))
        for i in range(process.consumer.count):
            consumers.append(ConsumerState(process_id=i, item_type=item_type))

    return producers, consumers

def run_simulation(sim_config: SimConfig, shocks=None) -> SimulationState:
    """Run the simulation as event-driven, with the data defined in the provided sim_config dataclass """
    simulation_state = create_simulation_state(sim_config)
    simulation_state.shocks = shocks if shocks is not None else []
    producers_state, consumers_state = create_producer_consumer_states(sim_config)
    sim_time = 0.0
    duration = sim_config.simulation_timeout_in_seconds
    queue_interval = sim_config.queue_interval
    next_queue_log_time = 0.0
    processes = producers_state + consumers_state

    while sim_time < duration:

        # apply pending outputs whose time has arrived
        ready_outputs = [p for p in simulation_state.pending_outputs if p[0] <= sim_time]

        for timestamp, item in ready_outputs:
            simulation_state.queues[item] += 1
            simulation_state.queue_history[item].append((timestamp, simulation_state.queues[item]))
            simulation_state.producer_logs.append(SimulationLogs(-1, item.value, timestamp))

        simulation_state.pending_outputs = [p for p in simulation_state.pending_outputs if p[0] > sim_time]

        # execute all ready processes
        for state in processes:
            if state.next_ready_time <= sim_time:
                if isinstance(state, ProducerState):
                    producer(state, simulation_state, sim_config, sim_time)
                else:
                    consumer(state, simulation_state, sim_config, sim_time)

        # determine the next event time (either a process becoming ready or an output appearing) to jump to
        process_times = [p.next_ready_time for p in processes if p.next_ready_time > sim_time]
        output_times = [t for t, _ in simulation_state.pending_outputs if t > sim_time]

        all_times = process_times + output_times

        next_event_time = min(all_times) if all_times else None
        if next_event_time is None or next_event_time > duration:
            break

        # log queue occupancies at the defined fixed intervals
        while next_queue_log_time <= next_event_time:
            for item_type, occupancy in simulation_state.queues.items():
                simulation_state.queue_logs.append(QueueLogs(item_type.value, occupancy, next_queue_log_time))
            next_queue_log_time += queue_interval
        
        # advance time to the next event
        sim_time = next_event_time
        
    return simulation_state

# ==================================================================================================
# Optuna
# ==================================================================================================

def objective(trial):

    # The higher the value, the more sharp the turns are on the lines, 
    # if above a threashold, it over reacts resulting in oscillations, if below threashols it stabilises
    test_sensitivity = trial.suggest_float('proportional_gain', 0.01, 0.1)

    production_time = 1.0
    # The higer the value, the more it can diverge from the target, it too high results are spreatic
    test_delay = trial.suggest_float('transport_lag', production_time, production_time * 20)

    sim_config = create_sim_config(test_sensitivity, test_delay)
    sim_state = run_simulation(sim_config)

    warmup_cutoff = sim_config.simulation_timeout_in_seconds * 0.5
    queues = {item.value: [] for item in sim_config.processes}

    for log in sim_state.queue_logs:
        if log.timestamp > warmup_cutoff:
            queues[log.queue_name].append(log.queue_usage)

    score = 0
    for item in sim_config.processes:
        series = queues[item.value]
        capacity = sim_config.processes[item].queue_capacity
        score += oscillation_score(series, capacity)

    return float(score)

def oscillation_score(series, capacity):
    if len(series) < 10:
        return 0
    
    std_dev = np.std(series)
    crossings = 0
    for i in range(2, len(series)):
        a, b, c = series[i-2], series[i-1], series[i]
        if (b > a and b > c) or (b < a and b < c):
            crossings += 1
    min_q = min(series)
    max_q = max(series)
    penalty = 0

    if min_q < 0.1 * capacity:
        penalty += (0.1 * capacity - min_q)
    if max_q > 0.9 * capacity:
        penalty += (max_q - 0.9 * capacity)

    return (std_dev * crossings) - penalty

# ==================================================================================================
# Stability Analysis
# ==================================================================================================

def apply_feedback_params(sim_config: SimConfig, param_dict: dict[str, float]) -> SimConfig:
    """Return a new SimConfig with updated feedback parameters using hierarchical overrides."""

    def resolve(param_dict, specific_key, global_key, default):
        """Helper to resolve parameter with fallback hierarchy."""
        return param_dict.get(specific_key,
               param_dict.get(global_key, default))

    new_processes = {}

    for item_type, process in sim_config.processes.items():

        producer = process.producer
        consumer = process.consumer

        if producer and producer.reference_signal is not None:
            producer = replace(
                producer,
                proportional_gain=resolve(
                    param_dict,
                    f"{item_type.name}_producer_sensitivity",
                    "global_sensitivity",
                    producer.proportional_gain
                ),
                transport_lag=resolve(
                    param_dict,
                    f"{item_type.name}_producer_delay",
                    "global_delay",
                    producer.transport_lag
                )
            )

        if consumer and consumer.reference_signal is not None:
            consumer = replace(
                consumer,
                proportional_gain=resolve(
                    param_dict,
                    f"{item_type.name}_consumer_sensitivity",
                    "global_sensitivity",
                    consumer.proportional_gain
                ),
                transport_lag=resolve(
                    param_dict,
                    f"{item_type.name}_consumer_delay",
                    "global_delay",
                    consumer.transport_lag
                )
            )

        new_processes[item_type] = replace(
            process,
            producer=producer,
            consumer=consumer
        )

    return replace(sim_config, processes=new_processes)

def stability_metrics(series):
    """Return stability components separately."""
    if len(series) < 10:
        return 0, 0, 0
    
    # Core metric - capture variability from equilibrium
    standard_deviation = np.std(series)

    # Capture the magnitude of step-to-step changes
    step_variability = np.std(np.diff(series))

    # Capture long-term drift 
    time_idx = np.arange(len(series))
    drift = np.polyfit(time_idx, series, 1)[0]

    return standard_deviation, step_variability, drift

def run_parametrized_simulation(base_config, x, y, stability_config):

    param_dict = dict(stability_config.get("fixed_params", {}))

    param_dict[stability_config["x_param"]] = x
    param_dict[stability_config["y_param"]] = y

    sim_config = apply_feedback_params(base_config, param_dict)
    sim_state = run_simulation(sim_config)

    return sim_state, sim_config

def rerun_point(base_config, x_values, y_values, idx, stability_config):
    i, j = idx
    x = x_values[i]
    y = y_values[j]

    sim_state, _ = run_parametrized_simulation(base_config, x, y, stability_config)

    return sim_state, x, y

def plot_multiple_heatmaps(base_config, x_values, y_values, std_matrix, diff_matrix, drift_matrix, 
                           max_std, max_diff, max_drift, feedback_type, stability_config):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    top_axes = axes[0]
    bottom_axes = axes[1]

    map_titles = [
        "Standard Deviation",
        "Step Variability",
        "Drift"
    ]

    sim_titles = [
        "Maximum Standard Deviation",
        "Maximum Step Variability",
        "Maximum Drift"
    ]

    matrices = [std_matrix, diff_matrix, drift_matrix]
    max_points = [max_std, max_diff, max_drift]

    for ax, matrix, max_point, map_title in zip(top_axes, matrices, max_points, map_titles):
        dx = x_values[1] - x_values[0]
        dy = y_values[1] - y_values[0]

        extent = [
            x_values[0] - dx/2,
            x_values[-1] + dx/2,
            y_values[0] - dy/2,
            y_values[-1] + dy/2
        ]
        im = ax.imshow(
            matrix,
            origin='lower',
            aspect='auto',
            extent=extent
        )

        i, j = max_point[1]

        x_center  = x_values[j] + 0.1 # offset to align with the center of the cell
        y_center = y_values[i]

        ax.plot(x_center, y_center, 'ro', label='Maximum instability')
        ax.legend()

        ax.set_title(map_title)
        ax.set_xlabel(stability_config["x_param"])
        ax.set_ylabel(stability_config["y_param"])
        fig.colorbar(im, ax=ax)

    for ax, max_point, sim_title in zip(bottom_axes, max_points, sim_titles):
        sim_state, x, y = rerun_point(base_config, x_values, y_values, max_point[1], stability_config)

        plot_queue_occupancy_over_time(ax, 0.0, sim_state.queue_logs)

        ax.set_title(sim_title)

    plt.suptitle(f"System Stability Breakdown – {feedback_type.name.title()}")
    plt.tight_layout()
    plt.show()

def plot_stability_heatmap(sensitivities, delays, matrix, feedback_type):
    plt.figure(figsize=(8,6))
    plt.imshow(
        matrix,
        origin='lower',
        aspect='auto',
        extent=[delays[0], delays[-1], sensitivities[0], sensitivities[-1]]
    )
    plt.colorbar(label="Instability Score")

    plt.xlabel("Feedback Delay")
    plt.ylabel("Reaction Sensitivity")
    plt.title(f"System Stability – {feedback_type.name.title()} Feedback")

    plt.show()

def run_stability_experiment(base_config: SimConfig, stability_config: dict, debug=False):
    x_values = stability_config["x_values"]
    y_values = stability_config["y_values"]

    std_matrix = np.zeros((len(x_values), len(y_values)))
    diff_matrix = np.zeros((len(x_values), len(y_values)))
    drift_matrix = np.zeros((len(x_values), len(y_values)))

    max_std = (-np.inf, None)
    max_diff = (-np.inf, None)
    max_drift = (-np.inf, None)

    def extract_queue_series(sim_config, sim_state, warmup_cutoff):
        queues = {item.value: [] for item in sim_config.processes}
        for log in sim_state.queue_logs:
            if log.timestamp > warmup_cutoff:
                queues[log.queue_name].append(log.queue_usage)
        return queues

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):

            sim_state, sim_config = run_parametrized_simulation(base_config, x, y, stability_config)

            warmup_cutoff = sim_config.simulation_timeout_in_seconds * 0.5
            queues = extract_queue_series(sim_config, sim_state, warmup_cutoff)

            std_score = 0
            diff_score = 0
            drift_score = 0

            for item in sim_config.processes:
                series = queues[item.value]

                std, diff, drift = stability_metrics(series)

                std_score += std
                diff_score += diff
                drift_score += drift

            std_matrix[i, j] = std_score
            diff_matrix[i, j] = diff_score
            drift_matrix[i, j] = drift_score

            if std_score > max_std[0]:
                max_std = (std_score, (i, j))

            if diff_score > max_diff[0]:
                max_diff = (diff_score, (i, j))

            if drift_score > max_drift[0]:
                max_drift = (drift_score, (i, j))

    logging.info("\n--- Stability Experiment Finished ---")
    logging.info(
        f"Grid searched {len(x_values) * len(y_values)} parameter combinations "
        f"({len(x_values)} x-values × {len(y_values)} y-values)")

    if debug:
        plot_multiple_heatmaps(
            base_config,
            x_values, y_values,
            std_matrix, diff_matrix, drift_matrix,
            max_std, max_diff, max_drift,
            base_config.feedback_type, stability_config
        )
    else:
        plot_stability_heatmap(
            x_values, y_values, std_matrix, base_config.feedback_type
        )


# ==================================================================================================
# Simulation Types
# ==================================================================================================

def run_optuna() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    
    logging.info("Starting Optuna optimization... Please wait while it runs 300 simulations.")
    
    study.optimize(objective, n_trials=300)

    contour_plot = vis.plot_contour(study, params=['proportional_gain', 'transport_lag'])
    contour_plot.show()

    best_sensitivity = study.best_params['proportional_gain']
    best_delay = study.best_params['transport_lag']

    logging.info("\n--- Optimization Finished ---")
    logging.info(f"Best Oscillation Score: {study.best_value:.2f}")
    logging.info(f"Winning Parameters: Sensitivity = {best_sensitivity:.4f}, Delay = {best_delay:.2f}s")
    logging.info("\nRunning final simulation with the best parameters to plot results...")

    best_sim_config = create_sim_config(best_sensitivity, best_delay)
    best_sim_state = run_simulation(best_sim_config)

    log_simulation_parameters(best_sim_config)
    log_results(best_sim_state)
    plot_results(best_sim_state)

def run_individual(sim_config: SimConfig, shocks=None) -> None:
    sim_state = run_simulation(sim_config, shocks)
    log_simulation_parameters(sim_config)
    log_results(sim_state)
    plot_results(sim_state, shocks=shocks)

# ==================================================================================================
# Sim Config Population
# ==================================================================================================

def create_sim_config(proportional_gain: float, transport_lag: float) -> SimConfig:
    return SimConfig(
        simulation_timeout_in_seconds=800,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type = FeedbackType.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=0.5,
                    reference_signal=50,
                    proportional_gain=proportional_gain,
                    transport_lag=transport_lag
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5,
                    reference_signal=50,
                    proportional_gain=proportional_gain,
                    transport_lag=transport_lag
                ),
            ),

            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_ROD,
                    output=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    reference_signal=50,
                    proportional_gain=proportional_gain,
                    transport_lag=transport_lag
                ),
            ),

            ItemType.IRON_WIRE: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    reference_signal=50,
                    proportional_gain=proportional_gain,
                    transport_lag=transport_lag
                ),
            ),
        }
    )

def is_machine_failed(item_type: ItemType, sim_time: float, simulation_state: SimulationState) -> bool:
    for shock in simulation_state.shocks:
        if shock.item_type == item_type:
            if shock.start_time <= sim_time <= shock.end_time:
                return True
    return False

# ==================================================================================================
# Main function
# ==================================================================================================

def main() -> None:
    """Main function for running the simulation."""

    # ======== Experiments ========
    # run_optuna()

    # ======== Individual Scenarios ========
    shocks = [
        ShockEvent(
            item_type=ItemType.IRON_ROD,
            start_time=200,
            end_time=250
        )
    ]
    for scenario in [
        # sim_scenarios.get_multiple_oscillations_input_f,
        # sim_scenarios.get_multiple_oscillations_output_f,
        # sim_scenarios.get_multiple_oscillations_dual_f,
        sim_scenarios.get_second_order_sim_no_delay_output_feedback,
        # sim_scenarios.get_balanced_flow,
        # sim_scenarios.get_bottleneck,
        # sim_scenarios.get_starvation,
        # sim_scenarios.get_backpressure_propagation,
    ]:

        sim_config, stability_config = scenario()

        run_individual(sim_config)

        # logging.info(
        #     f"Running {len(stability_config.get("x_values")) * len(stability_config.get("y_values"))} stability experiments..."
        # )

        # run_stability_experiment(
        #     sim_config,
        #     stability_config,
        #     debug=True
        # )

if __name__ == '__main__':
    main()