import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import optuna
import optuna.visualization as vis
from enum import Enum
from collections import Counter
from dataclasses import dataclass, field

import sim_scenarios

# --------------------------------------------------------------------------------------------------
# Running instructions 
#   RUN - python producers_consumers_many_to_many.py
# --------------------------------------------------------------------------------------------------

# ==================================================================================================
# Simulation data structures
# ==================================================================================================

class ItemType(Enum):
    IRON_INGOT = "Iron Ingot"
    IRON_ROD = "Iron Rod"
    IRON_WIRE = "Iron Wire"

@dataclass
class SimulationState:
    producer_logs: list
    consumer_logs: list
    queue_logs: list
    queues: dict[ItemType, int]
    queue_history: dict[ItemType, list[tuple[float, int]]]

@dataclass
class ProducerState:
    process_id: int
    item_type: ItemType
    next_ready_time: float = 0.0


@dataclass
class ConsumerState:
    process_id: int
    item_type: ItemType
    next_ready_time: float = 0.0

@dataclass
class SimulationLogs:
    process_id: int
    item_type: str
    timestamp: float

@dataclass
class QueueLogs:
    queue_name: str
    queue_usage: int
    timestamp: float

@dataclass(frozen=True)
class ProducerConfig:
    count: int = 0
    output: ItemType | None = None
    production_time: float | None = None
    # P-Control below
    target_queue_size: int | None = None
    reaction_sensitivity: float = 0.0
    feedback_delay: float = 0.0

@dataclass(frozen=True)
class ConsumerConfig:
    count: int = 0
    input: ItemType | None = None
    output: ItemType | None = None
    consumption_time: float | None = None
    # P-Control below
    target_queue_size: int | None = None
    reaction_sensitivity: float = 0.0
    feedback_delay: float = 0.0

@dataclass(frozen=True)
class NodeConfig:
    queue_size: int
    producer: ProducerConfig = ProducerConfig()
    consumer: ConsumerConfig = ConsumerConfig()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")

# ==================================================================================================
# Simulation definition
# ==================================================================================================

@dataclass(frozen=True)
class SimConfig:
    simulation_timeout_in_seconds: int = 200
    queue_interval: float = 1.0
    use_feedback: bool = False
    nodes: dict[ItemType, NodeConfig] = field(default_factory=dict)

# ==================================================================================================
# Main processes
# ==================================================================================================

def producer(state: ProducerState, simulation_state: SimulationState, sim_config: SimConfig, sim_time: float) -> None:
    """Run a producer process that generates the specified item type and places them into a queue."""
    node = sim_config.nodes[state.item_type]
    output_type = node.producer.output
    base_production_time = node.producer.production_time

    if sim_time < state.next_ready_time:
        return

    queue_size = simulation_state.queues[output_type]
    max_size = sim_config.nodes[output_type].queue_size
    if queue_size >= max_size:
        return
    
    if sim_config.use_feedback:
        next_production_time = calculate_adjusted_time(
            base_production_time, node.producer.target_queue_size, node.producer.reaction_sensitivity, 
            node.producer.feedback_delay, simulation_state.queue_history[output_type], sim_time)
    else:
        next_production_time = base_production_time

    simulation_state.queues[output_type] += 1
    simulation_state.queue_history[output_type].append((sim_time, simulation_state.queues[output_type]))
    simulation_state.producer_logs.append(SimulationLogs(state.process_id, state.item_type.value, sim_time))
    state.next_ready_time = sim_time + next_production_time

def consumer(state: ConsumerState, simulation_state: SimulationState, sim_config: SimConfig, sim_time: float) -> None:
    """Run a consumer process that retrieves and processes items from the provided queue."""
    node = sim_config.nodes[state.item_type]
    input_type = node.consumer.input
    output_type = node.consumer.output
    consumption_time = node.consumer.consumption_time

    # abort if not ready or nothing to consume
    if sim_time < state.next_ready_time or simulation_state.queues[input_type] <= 0:
        return
    
    # if consumption produces an output, consumption only occures when the output queue has space
    if output_type is not None:
        if simulation_state.queues[output_type] >= sim_config.nodes[output_type].queue_size:
            return
        
    if output_type is None or sim_config.use_feedback is False:
        consumption_time = 1.0 + np.random.normal(0, 0.5)
        adjusted_consumption_time = consumption_time
    else:
        adjusted_consumption_time = calculate_adjusted_time(
            consumption_time, node.consumer.target_queue_size, node.consumer.reaction_sensitivity,
            node.consumer.feedback_delay, simulation_state.queue_history[output_type], sim_time)

    simulation_state.queues[input_type] -= 1
    simulation_state.queue_history[input_type].append((sim_time, simulation_state.queues[input_type]))
    simulation_state.consumer_logs.append(SimulationLogs(state.process_id, state.item_type.value, sim_time))

    if output_type is not None:
        simulation_state.queues[output_type] += 1
        simulation_state.queue_history[output_type].append((sim_time, simulation_state.queues[output_type]))
        simulation_state.producer_logs.append(SimulationLogs(state.process_id, output_type.value, sim_time))

    state.next_ready_time = sim_time + adjusted_consumption_time

def get_delayed_queue_size(history: list[tuple[float, int]], current_time: float, delay: float) -> int:
    """Returns the size of the queue exactly 'delay' seconds ago."""
    target_time = current_time - delay
    # Assume empty before the delay period has passed
    if target_time <= 0.0:
        return history[0][1]
    
    for timestamp, size in reversed(history):
        if timestamp <= target_time:
            return size
    return 0

def calculate_adjusted_time(base_time: float, target_queue_size: int, reaction_sensitivity: float, 
    feedback_delay: float, queue_history: list[tuple[float, int]], sim_time: float) -> float:
    """Calculates the adjusted processing time using a smooth bounded S-curve."""
    delayed_queue = get_delayed_queue_size(queue_history, sim_time, feedback_delay)
    dif_from_target = target_queue_size - delayed_queue
    
    # Calculate the raw control signal
    control_signal = reaction_sensitivity * dif_from_target
    
    # Use exp() and atan() to gracefully bound the time variation
    # This acts as a natural limit, preventing the process from going to sleep forever
    time_multiplier = math.exp(-math.atan(control_signal))
    
    return base_time * time_multiplier
# ==================================================================================================
# Reporting - logs and diagrams
# ==================================================================================================

def log_simulation_parameters(sim_config: SimConfig) -> None:
    """Log the parameters the simulation is using."""
    logging.info(f"The simulation will be running for {sim_config.simulation_timeout_in_seconds} seconds.")
    logging.info(f"Feedback enabled: {sim_config.use_feedback}")

    for item_type, node_config in sim_config.nodes.items():
        config_info = f"The {item_type.value} node has - queue size: {node_config.queue_size}"
        if node_config.producer.count > 0:
            config_info += (
                f" | producer/s count: {node_config.producer.count}"
                f" | production time(s): {node_config.producer.production_time}"
            )
            if sim_config.use_feedback:
                config_info += (
                    f" | target queue: {node_config.producer.target_queue_size}"
                    f" | sensitivity: {node_config.producer.reaction_sensitivity}"
                    f" | delay: {node_config.producer.feedback_delay}"
                )
        if node_config.consumer.count > 0:
            config_info += (
                f" | consumer/s count: {node_config.consumer.count}"
                f" | consumption time(s): {node_config.consumer.consumption_time}"
            )
            if sim_config.use_feedback:
                config_info += (
                    f" | target queue: {node_config.consumer.target_queue_size}"
                    f" | sensitivity: {node_config.consumer.reaction_sensitivity}"
                    f" | delay: {node_config.consumer.feedback_delay}"
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
        title=f"Production & Consumption Throughput (bucket={bucket_size}s)",
    )
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend()

def plot_queue_size_over_time(ax: plt.Axes, start_time: float, queue_logs: list[QueueLogs]) -> None:
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

    ax.set(
        xlabel="Time (seconds)",
        ylabel="Queue size",
        title="Queue Sizes Over Time (1-second buckets)",
    )
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend()

def plot_results(simulation_state: SimulationState, start_time: float = 0.0) -> None:
    """Create one figure containing subplots."""
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    plot_producer_consumer_rates(ax1, start_time, simulation_state.producer_logs, simulation_state.consumer_logs)
    plot_queue_size_over_time(ax2, start_time, simulation_state.queue_logs)

    plt.tight_layout()
    plt.show()

# ==================================================================================================
# Simulation runners
# ==================================================================================================

def create_simulation_state(sim_config: SimConfig) -> SimulationState:
    """Create and return a SimulationState dataclass with initialized shared resources."""
    queues = {}
    queue_history = {}
    for item_type in sim_config.nodes:
        queues[item_type] = 0
        queue_history[item_type] = [(0.0, 0)]

    return SimulationState(
        producer_logs=[],
        consumer_logs=[],
        queue_logs=[],
        queues=queues,
        queue_history=queue_history
    )

def create_producer_consumer_states(sim_config: SimConfig) -> tuple[list[ProducerState], list[ConsumerState]]:
    """Create and return producer and consumer data stores."""
    producers = []
    consumers = []

    for item_type, node in sim_config.nodes.items():
        for i in range(node.producer.count):
            producers.append(ProducerState(process_id=i, item_type=item_type))
        for i in range(node.consumer.count):
            consumers.append(ConsumerState(process_id=i, item_type=item_type))

    return producers, consumers

def run_simulation(sim_config: SimConfig) -> SimulationState:
    """Run the simulation as event-driven, with the data defined in the provided sim_config dataclass """
    simulation_state = create_simulation_state(sim_config)
    producers_state, consumers_state = create_producer_consumer_states(sim_config)
    sim_time = 0.0
    duration = sim_config.simulation_timeout_in_seconds
    queue_interval = sim_config.queue_interval
    next_queue_log_time = 0.0
    processes = producers_state + consumers_state

    while sim_time < duration:
        # execute all ready processes
        for state in processes:
            if state.next_ready_time <= sim_time:
                if isinstance(state, ProducerState):
                    producer(state, simulation_state, sim_config, sim_time)
                else:
                    consumer(state, simulation_state, sim_config, sim_time)

        next_event_time = min((p.next_ready_time for p in processes if p.next_ready_time > sim_time), default=None)
        if next_event_time is None or next_event_time > duration:
            break

        # log queue sizes at the defined fixed intervals
        while next_queue_log_time <= next_event_time:
            for item_type, size in simulation_state.queues.items():
                simulation_state.queue_logs.append(QueueLogs(item_type.value, size, next_queue_log_time))
            next_queue_log_time += queue_interval
        
        sim_time = next_event_time
    return simulation_state

# ==================================================================================================
# Optuna
# ==================================================================================================

def objective(trial):

    test_sensitivity = trial.suggest_float('reaction_sensitivity', 0.01, 0.2)

    production_time = 1.0
    test_delay = trial.suggest_float('feedback_delay', production_time, production_time * 20)

    sim_config = create_sim_config(test_sensitivity, test_delay)

    sim_state = run_simulation(sim_config)

    warmup_cutoff = sim_config.simulation_timeout_in_seconds * 0.5

    queues = {item.value: [] for item in sim_config.nodes}

    for log in sim_state.queue_logs:
        if log.timestamp > warmup_cutoff:
            queues[log.queue_name].append(log.queue_usage)

    score = 0

    for item in sim_config.nodes:
        series = queues[item.value]
        capacity = sim_config.nodes[item].queue_size
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

def run_stability_experiment(sensitivities, delays):

    stability_matrix = np.zeros((len(sensitivities), len(delays)))

    best_score = float("inf")
    best_params = None

    for i, sensitivity in enumerate(sensitivities):
        for j, delay in enumerate(delays):

            sim_config = create_sim_config(sensitivity, delay)
            sim_state = run_simulation(sim_config)

            warmup_cutoff = sim_config.simulation_timeout_in_seconds * 0.5
            queues = extract_queue_series(sim_config, sim_state, warmup_cutoff)

            score = 0

            for item in sim_config.nodes:
                series = queues[item.value]
                score += stability_metric(series)

            stability_matrix[i, j] = score

            if score < best_score:
                best_score = score
                best_params = (sensitivity, delay)

    # ---- concise report ----

    logging.info("\n--- Stability Experiment Finished ---")

    logging.info(
        f"Grid searched {len(sensitivities) * len(delays)} parameter combinations "
        f"({len(sensitivities)} sensitivities × {len(delays)} delays)"
    )

    logging.info(
        f"Most Stable Parameters: Sensitivity = {best_params[0]:.4f}, "
        f"Delay = {best_params[1]:.2f}s"
    )

    logging.info(f"Stability Score: {best_score:.3f}")

    plot_stability_heatmap(sensitivities, delays, stability_matrix)


def extract_queue_series(sim_config, sim_state, warmup_cutoff):
    queues = {item.value: [] for item in sim_config.nodes}

    for log in sim_state.queue_logs:
        if log.timestamp > warmup_cutoff:
            queues[log.queue_name].append(log.queue_usage)

    return queues

def stability_metric(series):
    """Measures whether the queue converges to equilibrium. Low values mean stable, high values mean oscillatory."""
    if len(series) < 10:
        return 0

    std_dev = np.std(series)
    drift = abs(series[-1] - series[0])

    return std_dev + 0.5 * drift

def plot_stability_heatmap(sensitivities, delays, matrix):

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
    plt.title("Stability of Producer–Consumer System")

    plt.show()

# ==================================================================================================
# Sim Config Population
# ==================================================================================================

def create_sim_config(reaction_sensitivity: float, feedback_delay: float) -> SimConfig:
    production_time = 1.0
    return SimConfig(
        simulation_timeout_in_seconds=500,
        queue_interval=1.0,
        use_feedback=True,
        nodes={
            ItemType.IRON_INGOT: NodeConfig(
                queue_size=200,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=production_time,
                    target_queue_size=100,
                    reaction_sensitivity=reaction_sensitivity,
                    feedback_delay=feedback_delay
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0,
                    target_queue_size=100,
                    reaction_sensitivity=reaction_sensitivity,
                    feedback_delay=feedback_delay
                ),
            ),

            ItemType.IRON_ROD: NodeConfig(
                queue_size=200,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_ROD,
                    # output=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    # target_queue_size=100,
                    # reaction_sensitivity=reaction_sensitivity,
                    # feedback_delay=feedback_delay
                ),
            ),

            # ItemType.IRON_WIRE: NodeConfig(
            #     queue_size=200,
            #     consumer=ConsumerConfig(
            #         count=1,
            #         input=ItemType.IRON_WIRE,
            #         consumption_time=1.0
            #     ),
            # ),
        }
    )

# ==================================================================================================
# Simulation Types
# ==================================================================================================

def run_optuna() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    
    logging.info("Starting Optuna optimization... Please wait while it runs 300 simulations.")
    
    study.optimize(objective, n_trials=300)

    contour_plot = vis.plot_contour(study, params=['reaction_sensitivity', 'feedback_delay'])
    contour_plot.show()

    best_sensitivity = study.best_params['reaction_sensitivity']
    best_delay = study.best_params['feedback_delay']

    logging.info("\n--- Optimization Finished ---")
    logging.info(f"Best Oscillation Score: {study.best_value:.2f}")
    logging.info(f"Winning Parameters: Sensitivity = {best_sensitivity:.4f}, Delay = {best_delay:.2f}s")
    logging.info("\nRunning final simulation with the best parameters to plot results...")

    best_sim_config = create_sim_config(best_sensitivity, best_delay)
    best_sim_state = run_simulation(best_sim_config)

    log_simulation_parameters(best_sim_config)
    log_results(best_sim_state)
    plot_results(best_sim_state)

def run_stability_analysis():
    sensitivities = np.linspace(0.01, 0.25, 25)
    delays = np.linspace(1, 25, 25)

    logging.info(f"Running {len(sensitivities) * len(delays)} stability experiments...")
    run_stability_experiment(sensitivities, delays)

def run_individual() -> None:
    sim_config = sim_scenarios.get_multiple_oscillations()
    sim_state = run_simulation(sim_config)

    log_simulation_parameters(sim_config)
    log_results(sim_state)
    plot_results(sim_state)

# ==================================================================================================
# Main function
# ==================================================================================================

def main() -> None:
    """Main function for running the simulation."""
    run_optuna()
    run_stability_analysis()
    # run_individual()

if __name__ == '__main__':
    main()