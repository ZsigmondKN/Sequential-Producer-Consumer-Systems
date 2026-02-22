import logging
import optuna
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from collections import Counter
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------------
# Running instructions 
#   RUN - python producers_consumers_many_to_many.py
# --------------------------------------------------------------------------------------------------

# ==================================================================================================
# Simulation data structures
# ==================================================================================================

class ItemType(Enum):
    IRON_INGOT = "iron ingot"
    IRON_PLATE = "iron plate"
    IRON_COGS = "iron cogs"

@dataclass
class SimulationState:
    producer_logs: list
    consumer_logs: list
    queue_logs: list
    queues: dict[ItemType, int]

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

@dataclass(frozen=True)
class ConsumerConfig:
    count: int = 0
    input: ItemType | None = None
    output: ItemType | None = None
    consumption_time: float | None = None

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
    simulation_timeout_in_seconds: int = 30
    queue_interval: float = 1.0

    nodes: dict[ItemType, NodeConfig] = field(
        default_factory=lambda: {
            ItemType.IRON_INGOT: NodeConfig(
                queue_size=20,
                producer=ProducerConfig(
                    count=5,
                    output=ItemType.IRON_INGOT,
                    production_time=0.5,
                ),
                consumer=ConsumerConfig(
                    count=4,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_PLATE,
                    consumption_time=0.5,
                ),
            ),

            ItemType.IRON_PLATE: NodeConfig(
                queue_size=40,
                consumer=ConsumerConfig(
                    count=3,
                    input=ItemType.IRON_PLATE,
                    output=ItemType.IRON_COGS,
                    consumption_time=0.5,
                ),
            ),

            ItemType.IRON_COGS: NodeConfig(
                queue_size=40,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_COGS,
                    consumption_time=0.5,
                ),
            )
        }
    )

# ==================================================================================================
# Main processes
# ==================================================================================================

def producer(state: ProducerState, simulation_state: SimulationState, sim_config: SimConfig, sim_time: float) -> None:
    """Run a producer process that generates the specified item type and places them into a queue."""
    node = sim_config.nodes[state.item_type]
    output_type = node.producer.output
    production_time = node.producer.production_time

    if sim_time < state.next_ready_time:
        return

    queue_size = simulation_state.queues[output_type]
    max_size = sim_config.nodes[output_type].queue_size
    if queue_size >= max_size:
        return

    simulation_state.queues[output_type] += 1
    simulation_state.producer_logs.append(SimulationLogs(state.process_id, state.item_type.value, sim_time))
    state.next_ready_time = sim_time + production_time

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

    simulation_state.queues[input_type] -= 1
    simulation_state.consumer_logs.append(SimulationLogs(state.process_id, state.item_type.value, sim_time))

    if output_type is not None:
        simulation_state.queues[output_type] += 1
        simulation_state.producer_logs.append(SimulationLogs(state.process_id, output_type.value, sim_time))

    state.next_ready_time = sim_time + consumption_time

# ==================================================================================================
# Reporting - logs and diagrams
# ==================================================================================================

def log_simulation_parameters(sim_config: SimConfig) -> None:
    """Log the parameters the simulation is using."""
    logging.info(f"The simulation will be running for {sim_config.simulation_timeout_in_seconds} seconds. ")
    for item_type, node_config in sim_config.nodes.items():
        config_info = f"The {item_type.value} node has - queue size: {node_config.queue_size}"
        if node_config.producer.count > 0:
            config_info += f" | producer/s count: {node_config.producer.count} | production time(s): {node_config.producer.production_time}"
        if node_config.consumer.count > 0:
            config_info += f" | consumer/s count: {node_config.consumer.count} | consumption time(s): {node_config.consumer.consumption_time}"
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
    # items = sorted({log.item_type for log in list(producer_logs) + list(consumer_logs)})
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
        # ax.plot(bin_centres, prod_counts, "-o", markersize=4, alpha=0.9, label=f"Prod: {item}")
        # ax.plot(bin_centres, cons_counts, "--o", markersize=4, alpha=0.9, label=f"Cons: {item}")

        if len(prod_times) > 0:
            throughput = prod_counts
        else:
            throughput = cons_counts
        
        ax.plot(bin_centres, throughput, "--o", markersize=4, alpha=0.9, label=f"Throughput: {item}")
    
    ax.set(
        xlabel="Time (seconds)",
        ylabel="Items per second",
        # title=f"Production & Consumption Rates (bucket={bucket_size}s)",
        title=f"Production & Consumption Throughput (bucket={bucket_size}s)",
    )
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend()

def plot_queue_size_over_time(ax: plt.Axes, start_time: float, queue_logs: list[QueueLogs], 
                              bar_width: float = 0.4, offset: float = 0) -> None:
    queues: dict[str, list[QueueLogs]] = {}
    for log in queue_logs:
        queues.setdefault(log.queue_name, []).append(log)
    
    for queue_name, logs in queues.items():
        time_steps = [log.timestamp - start_time for log in logs]
        queue_usages = [log.queue_usage for log in logs]
        ax.bar([t + offset for t in time_steps], queue_usages, bar_width, label=queue_name, alpha=0.6)
        offset += bar_width

    ax.set(
        xlabel="Time (seconds)",
        ylabel="Queue size",
        title="Queue Sizes Over Time (1-second buckets)",
    )
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend()

def plot_results(simulation_state: SimulationState, start_time: float = 0.0) -> None:
    """Create one figure containing subplots."""
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharex=False)

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
    for item_type in sim_config.nodes:
        queues[item_type] = 0

    return SimulationState(
        producer_logs=[],
        consumer_logs=[],
        queue_logs=[],
        queues=queues
    )

def create_states(sim_config: SimConfig) -> tuple[list[ProducerState], list[ConsumerState]]:
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
    """Runs the simulation with the data defined in the provided sim_config dataclass."""
    simulation_state = create_simulation_state(sim_config)
    producers, consumers = create_states(sim_config)
    sim_time = 0.0
    dt = 0.01
    duration = sim_config.simulation_timeout_in_seconds
    last_queue_log_time = -sim_config.queue_interval

    while sim_time < duration:

        for state in producers:
            producer(state, simulation_state, sim_config, sim_time)

        for state in consumers:
            consumer(state, simulation_state, sim_config, sim_time)

        if sim_time - last_queue_log_time >= sim_config.queue_interval:
            for item_type, size in simulation_state.queues.items():
                simulation_state.queue_logs.append(
                    QueueLogs(item_type.value, size, sim_time)
                )
            last_queue_log_time = sim_time

        sim_time += dt
    return simulation_state

# ==================================================================================================
# Optuna Objective Setup
# ==================================================================================================

def suggest_parameters(trial: optuna.Trial) -> dict:
    return {
        "ingot_prod_time": trial.suggest_float("ingot_prod_time", 1, 5.0),
        "ingot_cons_time": trial.suggest_float("ingot_cons_time", 1, 5.0),
        "plate_cons_time": trial.suggest_float("plate_cons_time", 1, 5.0),
        "cogs_cons_time": trial.suggest_float("cogs_cons_time", 1, 5.0),

        "ingot_producers": trial.suggest_int("ingot_producers", 1, 5),
        "ingot_consumers": trial.suggest_int("ingot_consumers", 1, 5),
        "plate_consumers": trial.suggest_int("plate_consumers", 1, 5),
        "cogs_consumers": trial.suggest_int("cogs_consumers", 1, 5),

        "ingot_qsize": trial.suggest_int("ingot_qsize", 50, 200),
        "plate_qsize": trial.suggest_int("plate_qsize", 50, 200),
        "cogs_qsize": trial.suggest_int("cogs_qsize", 50, 200),
    }

def populate_sim_config(best_parameters: dict) -> SimConfig:
    return SimConfig(
        simulation_timeout_in_seconds=300,
        queue_interval=1,
        nodes={
            ItemType.IRON_INGOT: NodeConfig(
                queue_size=best_parameters["ingot_qsize"],
                producer=ProducerConfig(
                    count=best_parameters["ingot_producers"],
                    output=ItemType.IRON_INGOT,
                    production_time=best_parameters["ingot_prod_time"],
                ),
                consumer=ConsumerConfig(
                    count=best_parameters["ingot_consumers"],
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_PLATE,
                    consumption_time=best_parameters["ingot_cons_time"],
                ),
            ),
            ItemType.IRON_PLATE: NodeConfig(
                queue_size=best_parameters["plate_qsize"],
                consumer=ConsumerConfig(
                    count=best_parameters["plate_consumers"],
                    input=ItemType.IRON_PLATE,
                    output=ItemType.IRON_COGS,
                    consumption_time=best_parameters["plate_cons_time"],
                ),
            ),
            ItemType.IRON_COGS: NodeConfig(
                queue_size=best_parameters["cogs_qsize"],
                consumer=ConsumerConfig(
                    count=best_parameters["cogs_consumers"],
                    input=ItemType.IRON_COGS,
                    consumption_time=best_parameters["cogs_cons_time"],
                ),
            ),
        },
    )

def objective(trial: optuna.Trial) -> float:
    params = suggest_parameters(trial)
    sim_config = populate_sim_config(params)
    simulation_state = run_simulation(sim_config)

    return compute_score(simulation_state, sim_config, params)

def compute_score(simulation_state: SimulationState, sim_config: SimConfig, params: dict) ->float:
    ingot_prod_rate = params["ingot_producers"] / params["ingot_prod_time"]
    ingot_cons_rate = params["ingot_consumers"] / params["ingot_cons_time"]
    plate_cons_rate = params["plate_consumers"] / params["plate_cons_time"]
    cogs_cons_rate = params["cogs_consumers"] / params["cogs_cons_time"]

    rates = np.array([
        ingot_prod_rate,
        ingot_cons_rate,
        plate_cons_rate,
        cogs_cons_rate,
    ])

    # ---- Oscillation metric: FFT peak energy ----
    total_score = 0.0

    for item_type in ItemType:
        usages = [
            log.queue_usage
            for log in simulation_state.queue_logs
            if log.queue_name == item_type.value
        ]

        if len(usages) > 10:

            # Drop first 30% (remove transient ramp)
            cutoff = int(len(usages) * 0.3)
            usages = usages[cutoff:]

            capacity = sim_config.nodes[item_type].queue_size
            signal = np.array(usages) / capacity

            # Remove mean (remove DC component)
            signal = signal - np.mean(signal)

            # Remove transient
            cutoff = int(len(usages) * 0.3)
            signal = np.array(usages[cutoff:]) / capacity

            if len(signal) < 30:
                continue

            # Light smoothing
            signal = np.convolve(signal, np.ones(3)/3, mode="same")

            # Remove mean
            signal -= np.mean(signal)

            # 1. Amplitude
            amplitude = np.std(signal)

            # 2. Drift penalty
            trend = np.polyfit(np.arange(len(signal)), signal, 1)[0]
            drift_penalty = abs(trend)

            # 3. Autocorrelation peak
            corr = np.correlate(signal, signal, mode="full")
            corr = corr[len(corr)//2:]
            corr[0] = 0

            if len(corr) > 5:
                periodicity = np.max(corr[5:])
            else:
                periodicity = 0

            oscillation_score = (
                amplitude
                * periodicity
                / (1.0 + 10.0 * drift_penalty)
            )

            total_score += oscillation_score

    # --- Enforce near flow equilibrium ---
    rates = np.array([
        ingot_prod_rate,
        ingot_cons_rate,
        plate_cons_rate,
        cogs_cons_rate
    ])

    imbalance = np.std(rates) / (np.mean(rates) + 1e-8)

    # Strong penalty for imbalance
    total_score -= 20.0 * imbalance

    return total_score

# ==================================================================================================
# Main function
# ==================================================================================================

def main() -> None:
    """Main function for running the simulation."""
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=100)

    # print("Best value:", study.best_value)
    # print("Best params:", study.best_params)

    # Rebuild best results from Optuna runs
    # best_parameters = study.best_params
    # best_sim_config = populate_sim_config(best_parameters)
    best_sim_config = SimConfig(
        simulation_timeout_in_seconds=60,
        queue_interval=1.0,
        nodes={
            ItemType.IRON_INGOT: NodeConfig(
                queue_size=20,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=0.5,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_PLATE,
                    consumption_time=1,
                ),
            ),

            ItemType.IRON_PLATE: NodeConfig(
                queue_size=20,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_PLATE,
                    output=ItemType.IRON_COGS,
                    consumption_time=0.5,
                ),
            ),

            ItemType.IRON_COGS: NodeConfig(
                queue_size=20,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_COGS,
                    consumption_time=0.5,
                ),
            )
        }
    )
    best_simulation_state = run_simulation(best_sim_config)

    log_simulation_parameters(best_sim_config)
    log_results(best_simulation_state)
    plot_results(best_simulation_state)

if __name__ == '__main__':
    main()