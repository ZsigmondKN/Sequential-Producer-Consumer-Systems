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
class SimRunTime:
    producer_logs: list
    consumer_logs: list
    queue_logs: list
    queues: dict[ItemType, int]

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
# Simulation configurations
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

def producer(process_id: int, item_type: ItemType, sim_runtime: SimRunTime, 
             sim_config: SimConfig, sim_time: float, worker_state: dict) -> None:
    """Run a producer process that generates the specified item type and places them into a queue."""
    node = sim_config.nodes[item_type]
    output_type = node.producer.output
    production_time = node.producer.production_time

    while sim_time >= worker_state["next_ready"]:
        queue_size = sim_runtime.queues[output_type]
        max_size = sim_config.nodes[output_type].queue_size

        if queue_size < max_size:
            sim_runtime.queues[output_type] += 1
            sim_runtime.producer_logs.append(
                SimulationLogs(process_id, item_type.value, sim_time)
            )
            worker_state["next_ready"] += production_time
        else:
            break

def consumer(process_id: int, item_type: ItemType, sim_runtime: SimRunTime,
             sim_config: SimConfig, sim_time: float, worker_state: dict) -> None:
    """Run a consumer process that retrieves and processes items from the provided queue."""
    node = sim_config.nodes[item_type]
    input_type = node.consumer.input
    output_type = node.consumer.output
    consumption_time = node.consumer.consumption_time

    while sim_time >= worker_state["next_ready"]:

        # If no input available → stall
        if sim_runtime.queues[input_type] == 0:
            break

        # If there is an output queue and it's full → stall
        if output_type:
            next_node = sim_config.nodes[output_type]
            if sim_runtime.queues[output_type] >= next_node.queue_size:
                break

        # Perform consumption
        sim_runtime.queues[input_type] -= 1

        sim_runtime.consumer_logs.append(
            SimulationLogs(process_id, item_type.value, sim_time)
        )

        # Produce output if applicable
        if output_type:
            sim_runtime.queues[output_type] += 1
            sim_runtime.producer_logs.append(
                SimulationLogs(process_id, output_type.value, sim_time)
            )

        worker_state["next_ready"] += consumption_time

# ==================================================================================================
# Logging in terminal
# ==================================================================================================

def log_simulation_parameters(sim_runtime: SimRunTime, sim_config: SimConfig) -> None:
    """Log the parameters the simulation is using."""
    logging.info(f"The simulation will be running for {sim_config.simulation_timeout_in_seconds} seconds. ")
    for item_type, node_config in sim_config.nodes.items():
        config_info = f"The {item_type.value} node has - queue size: {node_config.queue_size}"
        if node_config.producer.count > 0:
            config_info += f" | producer/s count: {node_config.producer.count} | production time(s): {node_config.producer.production_time}"
        if node_config.consumer.count > 0:
            config_info += f" | consumer/s count: {node_config.consumer.count} | consumption time(s): {node_config.consumer.consumption_time}"
        logging.info(config_info)

def log_results(sim_runtime: SimRunTime) -> None:
    """Log a summary of total produced and consumed items."""
    produced_items = Counter(log.item_type for log in sim_runtime.producer_logs)
    consumed_items = Counter(log.item_type for log in sim_runtime.consumer_logs)

    logging.info("Simulation summary:")
    all_items = sorted(set(produced_items) | set(consumed_items))
    for item in all_items:
        logging.info(
            f"Item: {item} - produced: {produced_items.get(item, 0)} | consumed: {consumed_items.get(item, 0)}"
        )

# ==================================================================================================
# Diagram generation
# ==================================================================================================

def plot_producer_consumer_rates(ax: plt.Axes, start_time: float, producer_logs: list[SimulationLogs],
                                 consumer_logs: list[SimulationLogs], bucket_size: float = 1.0) -> None:
    items = sorted({log.item_type for log in list(producer_logs) + list(consumer_logs)})
    all_times = [log.timestamp - start_time for log in list(producer_logs) + list(consumer_logs)]
    max_t = max(all_times + [0])
    bins = np.arange(0, max_t + bucket_size, bucket_size)
    bin_centres = bins[:-1] + bucket_size / 2

    def extract_times(logs: list[SimulationLogs], item: str) -> list[float]:
        return [log.timestamp - start_time for log in logs if log.item_type == item]

    for item in items:
        prod_times = extract_times(producer_logs, item)
        prod_counts, _ = np.histogram(prod_times, bins=bins)
        ax.plot(bin_centres, prod_counts, "-o", markersize=4, alpha=0.9, label=f"Prod: {item}")
        cons_times = extract_times(consumer_logs, item)
        cons_counts, _ = np.histogram(cons_times, bins=bins)
        ax.plot(bin_centres, cons_counts, "--o", markersize=4, alpha=0.9, label=f"Cons: {item}")
    
    ax.set(
        xlabel="Time (seconds)",
        ylabel="Items per second",
        title=f"Production & Consumption Rates (bucket={bucket_size}s)",
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

def plot_results(start_time: float, sim_runtime: SimRunTime) -> None:
    """Create one figure containing subplots."""
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharex=False)

    plot_producer_consumer_rates(ax1, start_time, sim_runtime.producer_logs, sim_runtime.consumer_logs)
    plot_queue_size_over_time(ax2, start_time, sim_runtime.queue_logs)

    plt.tight_layout()
    plt.show()

# ==================================================================================================
# Main function helpers
# ==================================================================================================

def create_sim_runtime(sim_config: SimConfig) -> SimRunTime:
    """Create and return a SimRunTime dataclass with initialized shared resources."""
    return SimRunTime(
        producer_logs=[],
        consumer_logs=[],
        queue_logs=[],
        queues={
            item_type: 0
            for item_type in sim_config.nodes.keys()
        }
    )

# ==================================================================================================
# Simulation Logic
# ==================================================================================================

def run_simulation(sim_config: SimConfig) -> SimRunTime:

    sim_runtime = create_sim_runtime(sim_config)

    # Create worker state
    workers = []
    for item_type, node in sim_config.nodes.items():
        for i in range(node.producer.count):
            workers.append({
                "type": "producer",
                "id": i,
                "item_type": item_type,
                "next_ready": 0.0
            })

        for i in range(node.consumer.count):
            workers.append({
                "type": "consumer",
                "id": i,
                "item_type": item_type,
                "next_ready": 0.0
            })

    # Simulation loop
    sim_time = 0.0
    dt = 0.01
    simulation_duration = sim_config.simulation_timeout_in_seconds
    last_queue_log_time = -sim_config.queue_interval

    while sim_time < simulation_duration:

        for worker in workers:
            if worker["type"] == "producer":
                producer(worker["id"], worker["item_type"],
                         sim_runtime, sim_config,
                         sim_time, worker)
            else:
                consumer(worker["id"], worker["item_type"],
                         sim_runtime, sim_config,
                         sim_time, worker)

        # Log queue sizes
        if sim_time - last_queue_log_time >= sim_config.queue_interval:
            for item_type, size in sim_runtime.queues.items():
                sim_runtime.queue_logs.append(
                    QueueLogs(item_type.value, size, sim_time)
                )
            last_queue_log_time = sim_time

        sim_time += dt

    return sim_runtime


# ==================================================================================================
# Optuna Objective Setup
# ==================================================================================================

def objective(trial: optuna.Trial) -> float:

    # Suggest parameters
    ingot_prod_time = trial.suggest_float("ingot_prod_time", 0.1, 2.0)
    ingot_cons_time = trial.suggest_float("ingot_cons_time", 0.1, 2.0)
    plate_cons_time = trial.suggest_float("plate_cons_time", 0.1, 2.0)
    cogs_cons_time = trial.suggest_float("cogs_cons_time", 0.1, 2.0)

    ingot_producers = trial.suggest_int("ingot_producers", 1, 8)
    ingot_consumers = trial.suggest_int("ingot_consumers", 1, 8)
    plate_consumers = trial.suggest_int("plate_consumers", 1, 6)
    cogs_consumers = trial.suggest_int("cogs_consumers", 1, 4)

    ingot_qsize = trial.suggest_int("ingot_qsize", 5, 100)
    plate_qsize = trial.suggest_int("plate_qsize", 5, 100)
    cogs_qsize = trial.suggest_int("cogs_qsize", 5, 100)

    # Build config
    sim_config = SimConfig(
        simulation_timeout_in_seconds=60,
        queue_interval=0.5,
        nodes={
            ItemType.IRON_INGOT: NodeConfig(
                queue_size=ingot_qsize,
                producer=ProducerConfig(
                    count=ingot_producers,
                    output=ItemType.IRON_INGOT,
                    production_time=ingot_prod_time,
                ),
                consumer=ConsumerConfig(
                    count=ingot_consumers,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_PLATE,
                    consumption_time=ingot_cons_time,
                ),
            ),
            ItemType.IRON_PLATE: NodeConfig(
                queue_size=plate_qsize,
                consumer=ConsumerConfig(
                    count=plate_consumers,
                    input=ItemType.IRON_PLATE,
                    output=ItemType.IRON_COGS,
                    consumption_time=plate_cons_time,
                ),
            ),
            ItemType.IRON_COGS: NodeConfig(
                queue_size=cogs_qsize,
                consumer=ConsumerConfig(
                    count=cogs_consumers,
                    input=ItemType.IRON_COGS,
                    consumption_time=cogs_cons_time,
                ),
            ),
        },
    )

    sim_runtime = run_simulation(sim_config)

    # ---- Oscillation metric: FFT peak energy ----
    total_score = 0.0

    for item_type in ItemType:
        usages = [
            log.queue_usage
            for log in sim_runtime.queue_logs
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

            # Compute FFT
            fft_vals = np.fft.rfft(signal)
            power = np.abs(fft_vals)

            # Ignore zero-frequency (DC)
            if len(power) > 1:
                peak_energy = np.max(power[1:]) / len(signal)
                total_score += peak_energy

    return total_score



# ==================================================================================================
# Main function
# ==================================================================================================

def main() -> None:
    """Main function for running the simulation."""
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    # Rebuild config from best params
    p = study.best_params

    best_config = SimConfig(
        simulation_timeout_in_seconds=30,
        queue_interval=0.5,
        nodes={
            ItemType.IRON_INGOT: NodeConfig(
                queue_size=p["ingot_qsize"],
                producer=ProducerConfig(
                    count=p["ingot_producers"],
                    output=ItemType.IRON_INGOT,
                    production_time=p["ingot_prod_time"],
                ),
                consumer=ConsumerConfig(
                    count=p["ingot_consumers"],
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_PLATE,
                    consumption_time=p["ingot_cons_time"],
                ),
            ),
            ItemType.IRON_PLATE: NodeConfig(
                queue_size=p["plate_qsize"],
                consumer=ConsumerConfig(
                    count=p["plate_consumers"],
                    input=ItemType.IRON_PLATE,
                    output=ItemType.IRON_COGS,
                    consumption_time=p["plate_cons_time"],
                ),
            ),
            ItemType.IRON_COGS: NodeConfig(
                queue_size=p["cogs_qsize"],
                consumer=ConsumerConfig(
                    count=p["cogs_consumers"],
                    input=ItemType.IRON_COGS,
                    consumption_time=p["cogs_cons_time"],
                ),
            ),
        },
    )

    sim_runtime = run_simulation(best_config)

    log_results(sim_runtime)
    plot_results(0.0, sim_runtime)

if __name__ == '__main__':
    main()