import multiprocessing
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from queue import Empty, Full
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
    stop_event: multiprocessing.Event # type: ignore
    producer_logs: list
    consumer_logs: list
    queue_logs: list[str, int, float]
    queues: dict[ItemType, multiprocessing.Queue]

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

def producer(process_id: int, item_type: ItemType, sim_runtime: SimRunTime, sim_config: SimConfig) -> None:
    """Run a producer process that generates the specified item type and places them into a queue."""
    node = sim_config.nodes[item_type]
    output_queue = sim_runtime.queues[node.producer.output]
    base_production_time = node.producer.production_time

    while not sim_runtime.stop_event.is_set():
        time.sleep(base_production_time)
        item = (item_type.value, base_production_time)

        queue_update(output_queue, "put", sim_runtime.stop_event, item=item)
        sim_runtime.producer_logs.append(SimulationLogs(process_id, item_type.value, time.time()))

def consumer(process_id: int, item_type: ItemType, sim_runtime: SimRunTime, sim_config: SimConfig) -> None:
    """Run a consumer process that retrieves and processes items from the provided queue."""
    node = sim_config.nodes[item_type]
    output_type = node.consumer.output
    input_queue = sim_runtime.queues[node.consumer.input] if node.consumer.input else None
    output_queue = sim_runtime.queues[output_type] if output_type else None
    consumption_time = node.consumer.consumption_time

    while not sim_runtime.stop_event.is_set():
        item = queue_update(input_queue, "get", sim_runtime.stop_event)
        if item is None:
            break
        time.sleep(consumption_time)
        sim_runtime.consumer_logs.append(SimulationLogs(process_id, item_type.value, time.time()))
        if output_queue:
            new_item = (output_type.value, item[1])
            queue_update(output_queue, "put", sim_runtime.stop_event, item=new_item)
            sim_runtime.producer_logs.append(SimulationLogs(process_id, output_type.value, time.time()))

def track_queue_sizes(process_id: int, sim_runtime: SimRunTime, sim_config: SimConfig) -> None:
    """Track and log the sizes of the queues at regular intervals."""
    while not sim_runtime.stop_event.is_set():
        snapshot = []
        for item_type, queue in sim_runtime.queues.items():
            sim_runtime.queue_logs.append(QueueLogs(item_type.value, queue.qsize(), time.time()))
            snapshot.append(f"{item_type.value}: {queue.qsize()}")
        logging.info("Queues contain - " + " | ".join(snapshot))
        time.sleep(sim_config.queue_interval)

# ==================================================================================================
# Helper processes
# ==================================================================================================

def queue_update(queue: multiprocessing.Queue, action: str, stop_event, timeout: float=0.1, 
                 max_timeout: float=2.0, item: tuple=None) -> None:
    """Performs the defined action on the provided queue with exponential timeouts for better efficiency."""
    while not stop_event.is_set():
        try:
            if action == "put":
                queue.put(item, timeout=timeout)
                break
            elif action == "get":
                return queue.get(timeout=timeout)
        except (Full, Empty):
            timeout = min(timeout * 2, max_timeout)
    # if stopping, return None for get actions
    return None

# ==================================================================================================
# Process management functions
# ==================================================================================================

def start_processes(n_processes: int, target: callable, args: tuple) -> list:
    """Create and start multiprocessing processes."""
    process_type = getattr(args[0], 'value', 'generic')
    logging.info(f"Starting {n_processes} {process_type} {target.__name__} processes.")

    process_list = []
    for i in range(n_processes):
        process = multiprocessing.Process(target=target, args=(i,) + args)
        process.start()
        process_list.append(process)
    return process_list

def join_processes(process_list: list) -> None:
    """Wait for all processes in the list to complete."""
    for process in process_list:
        process.join(timeout=3)

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
        stop_event = multiprocessing.Event(),
        producer_logs = multiprocessing.Manager().list(),
        consumer_logs = multiprocessing.Manager().list(),
        queue_logs = multiprocessing.Manager().list(),
        queues = {
            item_type: multiprocessing.Queue(item_params.queue_size) for item_type, item_params in sim_config.nodes.items()
        }
    )

# ==================================================================================================
# Main function
# ==================================================================================================

def main() -> None:
    """Main function for running the simulation."""
    sim_config = SimConfig()
    processes_list = []
    simulation_start_time = time.time()
    sim_runtime = create_sim_runtime(sim_config)

    log_simulation_parameters(sim_runtime, sim_config)

    # Start tracking
    processes_list.extend(start_processes(1, track_queue_sizes, (sim_runtime, sim_config)))
    # Start producers and consumers
    for item_type, node in sim_config.nodes.items():
        if node.producer.count > 0:
            processes_list.extend(start_processes(node.producer.count, producer, (item_type, sim_runtime, sim_config)))
        if node.consumer.count > 0:
            processes_list.extend(start_processes(node.consumer.count, consumer, (item_type, sim_runtime, sim_config)))

    # Run the simulation for a fixed amount of time
    time.sleep(sim_config.simulation_timeout_in_seconds)

    # Signal all processes to stop and wait for completion
    sim_runtime.stop_event.set()
    join_processes(processes_list)

    log_results(sim_runtime)
    plot_results(simulation_start_time, sim_runtime)

if __name__ == '__main__':
    main()