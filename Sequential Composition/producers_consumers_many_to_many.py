from enum import Enum
import multiprocessing
from queue import Empty, Full
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import logging

SIMULATION_TIMEOUT_IN_SECONDS = 20
DEFAULT_NUM_PRODUCERS = 5
DEFAULT_NUM_CONSUMERS = 2
MAX_QUEUE_SIZE = 20
QUEUE_INTERVAL = 1
LOG_FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"

class ItemType(Enum):
    IRON_INGOT = "iron ingot"
    IRON_PLATE = "iron plate"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT)

def producer(process_id: int, item_type: ItemType, stop_event, producer_logs: list, 
             queue: multiprocessing.Queue) -> None:
    """Run a producer process that generates the specified item type and places them into a queue."""
    while not stop_event.is_set():
        production_time = round(random.random(), 2)
        time.sleep(production_time) # does it really need to sleep
        item = (item_type.value, production_time)
        queue_update(queue, "put", stop_event, item=item)

        producer_logs.append((process_id, production_time, time.time()))
        logging.debug(f"Produced: {item_type.value}, over {production_time} seconds.")
    logging.debug(f"Stopping {item_type.value} producer {process_id} gracefully.")

def consumer(process_id: int, item_type: ItemType, stop_event, consumer_logs: list, 
             primary_queue: multiprocessing.Queue, secondary_queue: multiprocessing.Queue=None) -> None:
    """Run a consumer process that retrieves and processes items from the provided queue."""
    while not stop_event.is_set():
        item = queue_update(primary_queue, "get", stop_event)
        if item is None:
            break
        service_time = round(random.random(), 2)
        time.sleep(service_time)
        if secondary_queue:
            secondary_queue.put((ItemType.IRON_PLATE.value, item[1])) #improve later

        consumer_logs.append((process_id, service_time, time.time()))
        logging.debug(f"Consumed: {item[0]}, over {service_time} seconds.")
    logging.debug(f"Stopping {item_type.value} consumer {process_id} gracefully.")

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

def track_queue_sizes(process_id: int, queue_iron_ingot: multiprocessing.Queue, 
                      queue_iron_plate: multiprocessing.Queue, queue_logs: list, stop_event) -> None:
    """Track and log the sizes of the queues at regular intervals."""
    while not stop_event.is_set():
        queue_logs.append((queue_iron_ingot.qsize(), queue_iron_plate.qsize(), time.time()))
        logging.info(f"Queue contains: {queue_iron_ingot.qsize()} iron ingots, "
                     f"{queue_iron_plate.qsize()} iron plates")
        time.sleep(QUEUE_INTERVAL)
    logging.debug(f"Stopping track queue process gracefully.")

def start_process(target: callable, args: tuple) -> multiprocessing.Process:
    """Create and start a single multiprocessing process."""
    logging.info(f"Starting a {target.__name__} process.")
    
    process = multiprocessing.Process(target=target, args=args)
    process.start()
    return process

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

def log_simulation_parameters() -> None:
    """Log the parameters the simulation is using."""
    logging.info(f"Simulation will be running for {SIMULATION_TIMEOUT_IN_SECONDS} seconds with, "
                 f"{DEFAULT_NUM_PRODUCERS} producers, {DEFAULT_NUM_CONSUMERS} consumers and a "
                 f"max queue size of {MAX_QUEUE_SIZE}.")

def log_results(producer_logs: list, consumer_logs: list) -> None:
    """Log a summary of total produced and consumed items."""
    logging.info(f"Logged {len(producer_logs)} produced and {len(consumer_logs)} consumed items.")

def plot_cumulative_producer_consumer(ax, start_time: float, producer_logs: list, consumer_logs: list) -> None:
    prod_times = [p[2] - start_time for p in producer_logs]
    cons_times = [c[2] - start_time for c in consumer_logs]

    bins = np.arange(0, np.ceil(max(prod_times + cons_times)) + 1)
    cumulative_produced = np.cumsum(np.histogram(prod_times, bins=bins)[0])
    cumulative_consumed = np.cumsum(np.histogram(cons_times, bins=bins)[0])

    ax.plot(bins[:-1], cumulative_produced, label="Cumulative Produced", color="blue", marker="o")
    ax.plot(bins[:-1], cumulative_consumed, label="Cumulative Consumed", color="red", marker="o")
    ax.set(
        xlabel="Time (seconds)",
        ylabel="Total items",
        title="Cumulative Production and Consumption"
    )
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend()

def plot_queue_size_over_time(ax, start_time: float, queue_logs: list) -> None:
    queue_times = [q[2] - start_time for q in queue_logs]
    iron_ingot_sizes = [q[0] for q in queue_logs]
    iron_plate_sizes = [q[1] for q in queue_logs]

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

def plot_results(start_time: float, producer_logs: list, consumer_logs: list, queue_logs: list) -> None:
    """Create one figure containing subplots."""
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharex=False)

    plot_cumulative_producer_consumer(ax1, start_time, producer_logs, consumer_logs)
    plot_queue_size_over_time(ax2, start_time, queue_logs)

    plt.tight_layout()
    plt.show()

def main() -> None:
    """Main function for running the simulation."""
    processes_list = []
    producer_logs = multiprocessing.Manager().list()
    consumer_logs = multiprocessing.Manager().list()
    queue_iron_ingot = multiprocessing.Queue(MAX_QUEUE_SIZE)
    queue_iron_plate = multiprocessing.Queue(MAX_QUEUE_SIZE)
    queue_logs = multiprocessing.Manager().list()
    simulation_start_time = time.time()

    # Flag to signal processes to stop
    stop_event = multiprocessing.Event()
    log_simulation_parameters()

    # Start all producer, consumer, and tracking processes
    processes_list.extend(
        start_processes(1, track_queue_sizes, (queue_iron_ingot, queue_iron_plate, queue_logs, 
                                               stop_event)))
    processes_list.extend(
        start_processes(DEFAULT_NUM_PRODUCERS, producer, (ItemType.IRON_INGOT, stop_event, 
                                                          producer_logs, queue_iron_ingot)))
    processes_list.extend(
        start_processes(DEFAULT_NUM_CONSUMERS, consumer, (ItemType.IRON_INGOT, stop_event, 
                                                          consumer_logs, queue_iron_ingot, 
                                                          queue_iron_plate)))
    processes_list.extend(
        start_processes(DEFAULT_NUM_CONSUMERS, consumer, (ItemType.IRON_PLATE, stop_event, 
                                                          consumer_logs, queue_iron_plate)))

    # Run the simulation for a fixed amount of time
    time.sleep(SIMULATION_TIMEOUT_IN_SECONDS)

    # Signal all processes to stop and wait for completion
    stop_event.set()
    join_processes(processes_list)

    log_results(producer_logs, consumer_logs)
    plot_results(simulation_start_time, producer_logs, consumer_logs, queue_logs)

if __name__ == '__main__':
    main()