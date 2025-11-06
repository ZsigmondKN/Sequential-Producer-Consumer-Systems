from enum import Enum
import multiprocessing
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import logging

DEFAULT_NUM_PRODUCERS = 5
DEFAULT_NUM_CONSUMERS = 2
ITEMS_PER_PRODUCER = 10
SENTINEL = None
LOG_FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"

class ItemType(Enum):
    IRON_INGOT = "iron ingot"
    IRON_PLATE = "iron plate"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT)

def producer(process_id: int, item_type: ItemType, producer_logs: list, 
             barrier: multiprocessing.Barrier, # type: ignore
             queue: multiprocessing.Queue) -> None:
    """Run a producer process that generates a fixed number of items and places them into a shared 
    queue.

    Each process simulates the production of items by taking a random amount of time for production. 
    When all producers have finished, a sentinel value is added to the queue. This signals to the 
    consumers that production stopped.
    """
    for item_index in range(ITEMS_PER_PRODUCER):
        production_time = round(random.random(), 2)
        time.sleep(production_time)
        item = (item_type.value, item_index, production_time)
        queue.put(item)

        producer_logs.append((process_id, item_index, production_time, time.time()))
        logging.info(f"Producer {process_id} produced an item: {item}")

    barrier.wait()
    if process_id == 0:
        for _ in range(DEFAULT_NUM_CONSUMERS):
            queue.put(SENTINEL)

def consumer(process_id: int, item_type: ItemType, consumer_logs: list,
             primary_queue: multiprocessing.Queue, 
             secondary_queue: multiprocessing.Queue = None) -> None:
    """Run a consumer process that retrieves and processes items from the queue.

    The consumer continues until it encounters the sentinel value, signaling the end of production. 
    Each consumed item is logged with a timestamp and simulated service time.
    """
    while True:
        item = primary_queue.get()
        service_time = random.random()
        time.sleep(service_time)
        if item is SENTINEL:
            if secondary_queue:
                secondary_queue.put(SENTINEL)
            break
        if secondary_queue:
            secondary_queue.put((ItemType.IRON_PLATE.value, item[1], item[2])) #improve later

        consumer_logs.append((process_id, item[0], service_time, time.time()))
        logging.info(f"Consumer {process_id} processed an item: {item}")

def start_process(n_processes: int, target: callable, process_list: list, args: tuple) -> None:
    """Create and start multiprocessing processes."""
    process_type = getattr(args[0], 'value', 'generic')
    logging.info(f"Starting {n_processes} {process_type} {target.__name__} processes")

    for i in range(n_processes):
        process = multiprocessing.Process(target=target, args=(i,) + args)
        process.start()
        process_list.append(process)

def join_processes(process_list: list) -> None:
    """Wait for all processes in the list to complete."""
    for process in process_list:
        process.join()

def log_results(producer_logs: list, consumer_logs: list) -> None:
    """Log a summary of total produced and consumed items."""
    logging.info(f"Logged {len(producer_logs)} produced and {len(consumer_logs)} consumed items.")

def plot_results(start_time: float, producer_logs: list, consumer_logs: list) -> None:
    """Plot cumulative produced vs. consumed items over time."""
    production_times = [p[3] for p in producer_logs] if producer_logs else []
    consumption_times = [c[3] for c in consumer_logs] if consumer_logs else []

    if not production_times and not consumption_times:
        logging.warning("No production or consumption events to plot.")
        return

    production_durations = [t - start_time for t in production_times]
    consumption_durations = [t - start_time for t in consumption_times]
    max_elapsed_time = int(np.ceil(max(production_durations + consumption_durations)))
    bins = np.arange(0, max_elapsed_time + 1, 1) if max_elapsed_time > 0 else np.array([0.0, 1.0])
    cumulative_produced = np.cumsum(np.histogram(production_durations, bins=bins)[0])
    cumulative_consumed = np.cumsum(np.histogram(consumption_durations, bins=bins)[0])

    plt.figure(figsize=(8, 4))
    plt.plot(bins[:-1], cumulative_produced, label='Cumulative Produced', color='blue', marker='o')
    plt.plot(bins[:-1], cumulative_consumed, label='Cumulative Consumed', color='red', marker='o')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Total items")
    plt.title("Cumulative Production vs Consumption")
    plt.grid(alpha=0.4, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

# entry point
if __name__ == '__main__':
    """Main entry point for the multiprocessing producer-consumer simulation."""
    producer_processes_list = []
    consumer_processes_list = []
    producer_logs = multiprocessing.Manager().list()
    consumer_logs = multiprocessing.Manager().list()
    queue_iron_ingot = multiprocessing.Queue()
    queue_iron_plate = multiprocessing.Queue()
    barrier = multiprocessing.Barrier(DEFAULT_NUM_PRODUCERS)
    simulation_start_time = time.time()

    start_process(DEFAULT_NUM_PRODUCERS, producer, producer_processes_list, 
                  (ItemType.IRON_INGOT, producer_logs, barrier, queue_iron_ingot))
    start_process(DEFAULT_NUM_CONSUMERS, consumer, consumer_processes_list, 
                  (ItemType.IRON_INGOT, consumer_logs, queue_iron_ingot, queue_iron_plate))
    start_process(DEFAULT_NUM_CONSUMERS, consumer, consumer_processes_list, 
                  (ItemType.IRON_PLATE, consumer_logs, queue_iron_plate))

    # wait for the processes to finish
    join_processes(producer_processes_list)
    join_processes(consumer_processes_list)

    log_results(producer_logs, consumer_logs)
    plot_results(simulation_start_time, producer_logs, consumer_logs)
