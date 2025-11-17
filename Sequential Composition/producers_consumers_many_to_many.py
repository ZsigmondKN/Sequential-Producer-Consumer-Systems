import multiprocessing
import random
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from queue import Empty, Full
from collections import Counter
from dataclasses import dataclass


# ==================================================================================================
# Simulation data structures
# ==================================================================================================

class ItemType(Enum):
    IRON_INGOT = "iron ingot"
    IRON_PLATE = "iron plate"

@dataclass
class SharedResources:
    stop_event: multiprocessing.Event # type: ignore
    producer_logs: list
    consumer_logs: list
    queue_logs: list
    queue_iron_ingot: multiprocessing.Queue
    queue_iron_plate: multiprocessing.Queue

@dataclass
class SimulationLogs:
    process_id: int
    item_type: str
    timestamp: float

@dataclass
class QueueLogs:
    iron_ingot_queue_size: int
    iron_plate_queue_size: int
    timestamp: float

@dataclass(frozen=True)
class Stage:
    input: str | None
    output: str | None
    time: float
    output_type: ItemType | None = None

# ==================================================================================================
# Simulation configurations
# ==================================================================================================

@dataclass(frozen=True)
class SimConfig:
    simulation_timeout_in_seconds: int = 30
    queue_interval: float = 1

    iron_ingot_producers: int = 5
    iron_ingot_consumers: int = 3
    iron_plate_consumers: int = 2

    iron_ingot_production_time: float = 0.5
    iron_ingot_consumption_time: float = 0.5
    iron_plate_consumption_time: float = 0.5

    iron_ingot_queue_size: int = 20
    iron_plate_queue_size: int = 40

CFG = SimConfig()

SEQUENCE_MODEL = {
    ItemType.IRON_INGOT: {
        "producer": Stage(
            input=None,
            output="queue_iron_ingot",
            time=CFG.iron_ingot_production_time,
        ),
        "consumer": Stage(
            input="queue_iron_ingot",
            output="queue_iron_plate",
            time=CFG.iron_ingot_consumption_time,
            output_type=ItemType.IRON_PLATE,
        ),
    },

    ItemType.IRON_PLATE: {
        "producer": Stage(
            input=None,
            output="queue_iron_plate",
            time=None,
        ),
        "consumer": Stage(
            input="queue_iron_plate",
            output=None,
            time=CFG.iron_plate_consumption_time,
            output_type=None,
        )
    }
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")

# ==================================================================================================
# Process target functions
# ==================================================================================================

def producer(process_id: int, item_type: ItemType, shared_r: SharedResources) -> None:
    """Run a producer process that generates the specified item type and places them into a queue."""
    model = SEQUENCE_MODEL[item_type]["producer"]
    base_production_time = model.time
    output_queue = getattr(shared_r, model.output)
    max_queue_size = output_queue._maxsize
    while not shared_r.stop_event.is_set():

        queue_size = output_queue.qsize()

        production_time = logistic_slowdown(queue_size, max_queue_size, base_production_time)
        time.sleep(production_time)
        item = (item_type.value, production_time)

        queue_update(output_queue, "put", shared_r.stop_event, item=item)
        shared_r.producer_logs.append(SimulationLogs(process_id, item_type.value, time.time()))

def consumer(process_id: int, item_type: ItemType, shared_r: SharedResources) -> None:
    """Run a consumer process that retrieves and processes items from the provided queue."""
    model = SEQUENCE_MODEL[item_type]["consumer"]

    input_queue = getattr(shared_r, model.input)
    output_queue = getattr(shared_r, model.output) if model.output else None
    consumption_time = model.time
    output_type = model.output_type

    while not shared_r.stop_event.is_set():
        item = queue_update(input_queue, "get", shared_r.stop_event)
        if item is None:
            break
        time.sleep(consumption_time)
        shared_r.consumer_logs.append(SimulationLogs(process_id, item_type.value, time.time()))
        if output_queue:
            new_item = (output_type.value, item[1])
            queue_update(output_queue, "put", shared_r.stop_event, item=new_item)
            shared_r.producer_logs.append(SimulationLogs(process_id, output_type.value, time.time()))

def track_queue_sizes(process_id: int, shared_r: SharedResources) -> None:
    """Track and log the sizes of the queues at regular intervals."""
    while not shared_r.stop_event.is_set():
        shared_r.queue_logs.append(QueueLogs(shared_r.queue_iron_ingot.qsize(), 
                                             shared_r.queue_iron_plate.qsize(), 
                                                     time.time()))
        logging.info(f"Queue contains: {shared_r.queue_iron_ingot.qsize()} iron ingots, "
                    f"{shared_r.queue_iron_plate.qsize()} iron plates")
        time.sleep(CFG.queue_interval)

# ==================================================================================================
# Process target helper functions
# ==================================================================================================

def logistic_slowdown(qsize, max_qsize, base_time):
    """
    Smoothly increase production time as the queue fills.
    Tunable S-curve controlled by midpoint, steepness and slowdown range.
    """

    fill = qsize / max_qsize

    # --- Tunable parameters ---
    midpoint = 0.5      # where slowdown begins (0–1)
    steepness = 20      # how fast the curve rises
    min_multiplier = 1  # base_time × this when queue empty
    max_multiplier = 2  # base_time × this when queue full

    # --- Logistic S-curve ---
    logistic_value = 1 / (1 + np.exp(-steepness * (fill - midpoint)))

    # Map logistic output (0→1) to slowdown range (min_multiplier→max_multiplier)
    multiplier = min_multiplier + logistic_value * (max_multiplier - min_multiplier)

    return base_time * multiplier

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

def log_simulation_parameters() -> None:
    """Log the parameters the simulation is using."""
    logging.info(f"The simulation will be running for {CFG.simulation_timeout_in_seconds} seconds. ")
    logging.info(f"The iron ingot material will have {CFG.iron_ingot_producers} producers, "
                 f"{CFG.iron_ingot_consumers} consumers and a max queue size of {CFG.iron_ingot_queue_size}.")
    logging.info(f"The iron plate material will have {CFG.iron_plate_consumers} consumers "
                 f"and a max queue size of {CFG.iron_plate_queue_size}.")
def log_results(shared_r: SharedResources) -> None:
    """Log a summary of total produced and consumed items."""
    produced_items_count_dict = Counter(log.item_type for log in shared_r.producer_logs)
    consumed_items_count_dict = Counter(log.item_type for log in shared_r.consumer_logs)

    ingot_val = ItemType.IRON_INGOT.value
    plate_val = ItemType.IRON_PLATE.value

    logging.info(f"Item type: {ingot_val} - produced {produced_items_count_dict[ingot_val]}, "
                 f"consumed {consumed_items_count_dict[ingot_val]}.")
    logging.info(f"Item type: {plate_val} - produced {produced_items_count_dict[plate_val]}, "
                 f"consumed {consumed_items_count_dict[plate_val]}.")

# ==================================================================================================
# Diagram generation
# ==================================================================================================

def plot_producer_consumer_rates(
    ax: plt.Axes,
    start_time: float,
    producer_logs: list[SimulationLogs],
    consumer_logs: list[SimulationLogs],
    bucket_size: float = 1.0,
    smooth_window: int = 3,  # seconds of rolling smoothing
) -> None:

    # --- Extract timestamps per type ---
    def extract_times(logs, item):
        return [log.timestamp - start_time for log in logs if log.item_type == item]

    prod_ingot = extract_times(producer_logs, ItemType.IRON_INGOT.value)
    prod_plate = extract_times(producer_logs, ItemType.IRON_PLATE.value)
    cons_ingot = extract_times(consumer_logs, ItemType.IRON_INGOT.value)
    cons_plate = extract_times(consumer_logs, ItemType.IRON_PLATE.value)

    # --- Determine histogram bin edges ---
    max_t = max(
        prod_ingot + prod_plate + cons_ingot + cons_plate + [0]
    )
    bins = np.arange(0, max_t + bucket_size, bucket_size)

    # --- Compute binned rates ---
    def rate(times):
        return np.histogram(times, bins=bins)[0]

    prod_ingot_rate = rate(prod_ingot)
    prod_plate_rate = rate(prod_plate)
    cons_ingot_rate = rate(cons_ingot)
    cons_plate_rate = rate(cons_plate)

    # --- Apply smoothing (rolling mean) ---
    def smooth(arr):
        if smooth_window <= 1:
            return arr
        kernel = np.ones(smooth_window) / smooth_window
        return np.convolve(arr, kernel, mode="same")

    prod_ingot_rate_s = smooth(prod_ingot_rate)
    prod_plate_rate_s = smooth(prod_plate_rate)
    cons_ingot_rate_s = smooth(cons_ingot_rate)
    cons_plate_rate_s = smooth(cons_plate_rate)

    # Use the *bucket centers* for nicer alignment
    t = bins[:-1] + bucket_size / 2

    # --- Line plots with markers for visibility ---
    ax.plot(t, prod_ingot_rate_s, "-o", label="Prod: Iron Ingot",
            color="blue", markersize=4, alpha=0.9)
    ax.plot(t, prod_plate_rate_s, "-o", label="Prod: Iron Plate",
            color="cyan", markersize=4, alpha=0.9)
    ax.plot(t, cons_ingot_rate_s, "-o", label="Cons: Iron Ingot",
            color="red", markersize=4, alpha=0.9)
    ax.plot(t, cons_plate_rate_s, "-o", label="Cons: Iron Plate",
            color="orange", markersize=4, alpha=0.9)

    # Slight transparency helps overlapping points remain visible
    for line in ax.lines:
        line.set_alpha(0.85)
        line.set_linewidth(1.6)

    ax.set(
        xlabel="Time (seconds)",
        ylabel="Items per second",
        title=f"Production & Consumption Rates (bucket={bucket_size}s, smooth={smooth_window}s)"
    )
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend()

def plot_queue_size_over_time(ax: plt.Axes, start_time: float, queue_logs: list[QueueLogs]) -> None:
    queue_times = [log.timestamp - start_time for log in queue_logs]
    iron_ingot_sizes = [log.iron_ingot_queue_size for log in queue_logs]
    iron_plate_sizes = [log.iron_plate_queue_size for log in queue_logs]

    bar_width = 0.4
    ax.bar([t - bar_width / 2 for t in queue_times], iron_ingot_sizes, width=bar_width,
           label="Iron Ingot Queue", color="green", alpha=0.6)
    ax.bar([t + bar_width / 2 for t in queue_times], iron_plate_sizes, width=bar_width,
           label="Iron Plate Queue", color="orange", alpha=0.6)
    ax.set(
        xlabel="Time (seconds)",
        ylabel="Queue size",
        title="Queue Sizes Over Time (1-second buckets)"
    )
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend()

def plot_results(start_time: float, shared_r: SharedResources) -> None:
    """Create one figure containing subplots."""
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharex=False)

    plot_producer_consumer_rates(ax1, start_time, shared_r.producer_logs, 
                                      shared_r.consumer_logs)
    plot_queue_size_over_time(ax2, start_time, shared_r.queue_logs)

    plt.tight_layout()
    plt.show()

# ==================================================================================================
# Main function helpers
# ==================================================================================================

def create_shared_resources() -> SharedResources:
    """Create and return a SharedResources dataclass with initialized shared resources."""
    return SharedResources(
        stop_event = multiprocessing.Event(),
        producer_logs = multiprocessing.Manager().list(),
        consumer_logs = multiprocessing.Manager().list(),
        queue_logs = multiprocessing.Manager().list(),
        queue_iron_ingot = multiprocessing.Queue(CFG.iron_ingot_queue_size),
        queue_iron_plate = multiprocessing.Queue(CFG.iron_plate_queue_size),
    )

# ==================================================================================================
# Main function
# ==================================================================================================

def main() -> None:
    """Main function for running the simulation."""
    processes_list = []
    simulation_start_time = time.time()
    shared_resources = create_shared_resources()

    # Start all producer, consumer, and tracking processes
    log_simulation_parameters()
    processes_list.extend(start_processes(1, track_queue_sizes, (shared_resources,)))
    processes_list.extend(start_processes(CFG.iron_ingot_producers, producer, (ItemType.IRON_INGOT, shared_resources)))
    processes_list.extend(start_processes(CFG.iron_ingot_consumers, consumer, (ItemType.IRON_INGOT, shared_resources)))
    processes_list.extend(start_processes(CFG.iron_plate_consumers, consumer, (ItemType.IRON_PLATE, shared_resources)))

    # Run the simulation for a fixed amount of time
    time.sleep(CFG.simulation_timeout_in_seconds)

    # Signal all processes to stop and wait for completion
    shared_resources.stop_event.set()
    join_processes(processes_list)

    log_results(shared_resources)
    plot_results(simulation_start_time, shared_resources)

if __name__ == '__main__':
    main()