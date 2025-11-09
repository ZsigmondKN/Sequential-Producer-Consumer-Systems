from enum import Enum
import multiprocessing
from queue import Empty, Full
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import logging

SIMULATION_TIME_OUT_IN_SECONDS = 20
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
    check_s = 0.1
    max_check_s = 2.0
    while not stop_event.is_set():
        production_time = round(random.random(), 2)
        time.sleep(production_time) # does it really need to sleep
        item = (item_type.value, production_time)
        try:
            queue.put(item, timeout=check_s)
        except Full:
            check_s = min(check_s * 2, max_check_s)
            continue

        producer_logs.append((process_id, production_time, time.time()))
        logging.debug(f"Produced: {item_type.value}, over {production_time} seconds.")
    logging.debug(f"Stopping {item_type.value} producer {process_id} gracefully.")

def consumer(process_id: int, item_type: ItemType, stop_event, consumer_logs: list, 
             primary_queue: multiprocessing.Queue, secondary_queue: multiprocessing.Queue = None) -> None:
    """Run a consumer process that retrieves and processes items from the provided queue."""
    check_s = 0.1
    max_check_s = 2.0
    while not stop_event.is_set():
        try:
            item = primary_queue.get(timeout=check_s)
        except Empty:
            check_s = min(check_s * 2, max_check_s)
            continue

        service_time = round(random.random(), 2)
        time.sleep(service_time)
        if secondary_queue:
            secondary_queue.put((ItemType.IRON_PLATE.value, item[1])) #improve later

        consumer_logs.append((process_id, service_time, time.time()))
        logging.debug(f"Consumed: {item[0]}, over {service_time} seconds.")
    logging.debug(f"Stopping {item_type.value} consumer {process_id} gracefully.")

def track_queue_sizes(queue_iron_ingot: multiprocessing.Queue, queue_iron_plate: multiprocessing.Queue, 
                      queue_logs: list, stop_event) -> None:
    """Track and log the sizes of the queues at regular intervals."""
    while not stop_event.is_set():
        queue_logs.append((queue_iron_ingot.qsize(), queue_iron_plate.qsize(), time.time()))
        logging.info(f"Queue contains: {queue_iron_ingot.qsize()} iron ingots, "
                     f"{queue_iron_plate.qsize()} iron plates")
        time.sleep(QUEUE_INTERVAL)
    logging.debug(f"Stopping track queue process gracefully.")

def start_process(target: callable, args: tuple) -> multiprocessing.Process:
    """Create and start a multiprocessing process."""
    process = multiprocessing.Process(target=target, args=args)
    process.start()
    return process

def start_processes(n_processes: int, target: callable, process_list: list, args: tuple) -> None:
    """Create and start multiprocessing processes."""
    process_type = getattr(args[0], 'value', 'generic')
    logging.info(f"Starting {n_processes} {process_type} {target.__name__} processes.")

    for i in range(n_processes):
        process = multiprocessing.Process(target=target, args=(i,) + args)
        process.start()
        process_list.append(process)

def join_processes(process_list: list) -> None:
    """Wait for all processes in the list to complete."""
    for process in process_list:
        process.join(timeout=3)

def log_simulation_parameters() -> None:
    """Log the parameters the simulation is using."""
    logging.info(f"Simulation will be running for {SIMULATION_TIME_OUT_IN_SECONDS} seconds with, "
                 f"{DEFAULT_NUM_PRODUCERS} producers, {DEFAULT_NUM_CONSUMERS} consumers and a "
                 f"max queue size of {MAX_QUEUE_SIZE}.")

def log_results(producer_logs: list, consumer_logs: list) -> None:
    """Log a summary of total produced and consumed items."""
    logging.info(f"Logged {len(producer_logs)} produced and {len(consumer_logs)} consumed items.")

def plot_results(start_time: float, producer_logs: list, consumer_logs: list, 
                 queue_logs: list) -> None:
    """Plot cumulative produced vs. consumed items over time."""
    prod_times = [p[2] - start_time for p in producer_logs]
    cons_times = [c[2] - start_time for c in consumer_logs]

    if not prod_times and not cons_times:
        logging.warning("No production or consumption events to plot.")
        return

    bins = np.arange(0, np.ceil(max(prod_times + cons_times)) + 1)
    cumulative_produced = np.cumsum(np.histogram(prod_times, bins=bins)[0])
    cumulative_consumed = np.cumsum(np.histogram(cons_times, bins=bins)[0])

    queue_times = [q[2] - start_time for q in queue_logs] if queue_logs else []
    iron_ingot_sizes = [q[0] for q in queue_logs] if queue_logs else []
    iron_plate_sizes = [q[1] for q in queue_logs] if queue_logs else []

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharex=True)

    ax1.plot(bins[:-1], cumulative_produced, label='Cumulative Produced', color='blue', marker='o')
    ax1.plot(bins[:-1], cumulative_consumed, label='Cumulative Consumed', color='red', marker='o')
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Total items")
    ax1.set_title("Cumulative Production and Consumption")
    ax1.grid(alpha=0.4, linestyle=':')
    ax1.legend()

    bar_width = 0.4
    ax2.bar([t - bar_width/2 for t in queue_times], iron_ingot_sizes, width=bar_width, 
            label='Iron Ingot Queue', color='green')
    ax2.bar([t + bar_width/2 for t in queue_times], iron_plate_sizes, width=bar_width,
            label='Iron Plate Queue', color='orange')
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Queue size")
    ax2.set_title("Queue Sizes Over Time (1-second buckets)")
    ax2.grid(alpha=0.4, linestyle=':')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    """Main entry point for the multiprocessing producer-consumer simulation."""
    producer_processes_list = []
    consumer_processes_list = []
    producer_logs = multiprocessing.Manager().list()
    consumer_logs = multiprocessing.Manager().list()
    queue_iron_ingot = multiprocessing.Queue(MAX_QUEUE_SIZE)
    queue_iron_plate = multiprocessing.Queue(MAX_QUEUE_SIZE)
    queue_logs = multiprocessing.Manager().list()
    simulation_start_time = time.time()

    # Flag to signal processes to stop
    stop_event = multiprocessing.Event()
    log_simulation_parameters()

    track_queue_process = start_process(track_queue_sizes, 
                                        (queue_iron_ingot, queue_iron_plate, queue_logs, stop_event))
    start_processes(DEFAULT_NUM_PRODUCERS, producer, producer_processes_list, 
                    (ItemType.IRON_INGOT, stop_event, producer_logs, queue_iron_ingot))
    start_processes(DEFAULT_NUM_CONSUMERS, consumer, consumer_processes_list, 
                    (ItemType.IRON_INGOT, stop_event, consumer_logs, queue_iron_ingot, queue_iron_plate))
    start_processes(DEFAULT_NUM_CONSUMERS, consumer, consumer_processes_list, 
                    (ItemType.IRON_PLATE, stop_event, consumer_logs, queue_iron_plate))

    # Run the simulation for a fixed amount of time
    time.sleep(SIMULATION_TIME_OUT_IN_SECONDS)

    # Signal all processes to stop and wait for completion
    stop_event.set()
    join_processes(producer_processes_list)
    join_processes(consumer_processes_list)
    track_queue_process.join()

    log_results(producer_logs, consumer_logs)
    plot_results(simulation_start_time, producer_logs, consumer_logs, queue_logs)
