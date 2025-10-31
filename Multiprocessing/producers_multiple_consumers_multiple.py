import multiprocessing
import random
import time
import matplotlib.pyplot as plt

# producer task
def producer(queue: multiprocessing.Queue, barrier, identifier: int, producer_logs: list):
    print(f"Producer {identifier}: Running", flush=True)
    for i in range(10):
        # generate a value
        value = round(random.random(), 2)
        # block
        time.sleep(value)
        # create a tuple of index and value
        item = (i, value)
        # add to the queue
        queue.put(item)
        # log the production
        producer_logs.append((identifier, i, value, time.time()))
        print(f"++ Producer {identifier}: produced {item}", flush=True)
    # add done
    barrier.wait()
    print(f"Producers: Done", flush=True)
    if identifier == 1:
        # this is the last producer
        queue.put(None)

# consumer task
def consumer(queue: multiprocessing.Queue, identifier: int, consumer_logs: list):
    print(f"Consumer {identifier}: Running", flush=True)
    while True:
        # get a unit of work
        item = queue.get()
        # simulate service time
        service_time = random.random()
        time.sleep(service_time)
        # check for stop
        if item is None:
            # put the stop signal back for other consumers
            queue.put(item)
            # stop running
            break
        # log the consumption
        consumer_logs.append((identifier, item[0], service_time, time.time()))
        print(f"-- Consumer {identifier}: consumed {item}", flush=True)

# entry point
if __name__ == '__main__':
    # define number of producers and consumers
    n_producers = 5
    n_consumers = 2
    # create producer and consumer lists
    producer_processes = []
    consumer_processes = []
    # create producer and consumer logs
    manager = multiprocessing.Manager()
    producer_logs = manager.list()
    consumer_logs = manager.list()
    # create the shared queue
    queue = multiprocessing.Queue()
    # create a shared barrier for producers
    barrier = multiprocessing.Barrier(n_producers)
    # start the producer processes
    for i in range(n_producers):
        producer_process = multiprocessing.Process(target=producer, args=(queue,barrier,i,producer_logs))
        producer_process.start()
        producer_processes.append(producer_process)
    # start the consumer processes
    for i in range(n_consumers):
        consumer_process = multiprocessing.Process(target=consumer, args=(queue,i,consumer_logs))
        consumer_process.start()
        consumer_processes.append(consumer_process)
    # wait for the processes to finish
    for producer_process in producer_processes:
        producer_process.join()
    for consumer_process in consumer_processes:
        consumer_process.join()
    print("Simulation: Done", flush=True)

    print(f"Logged {len(producer_logs)} produced and {len(consumer_logs)} consumed items.")

    # get time of productions and consumptions
    prod_times = [x[3] for x in producer_logs]
    cons_times = [x[3] for x in consumer_logs]

    import numpy as np

    # Normalize times relative to start
    t0 = min(prod_times + cons_times)
    prod_seconds = [t - t0 for t in prod_times]
    cons_seconds = [t - t0 for t in cons_times]

    # Choose bin width (e.g. 1 second)
    max_time = max(prod_seconds + cons_seconds)
    bins = np.arange(0, max_time + 1, 1)  # 1-second bins

    # Count events per bin
    prod_hist, _ = np.histogram(prod_seconds, bins)
    cons_hist, _ = np.histogram(cons_seconds, bins)

    # Optional: also show cumulative totals
    plt.plot(bins[:-1], np.cumsum(prod_hist), label='Cumulative Produced', color='blue')
    plt.plot(bins[:-1], np.cumsum(cons_hist), label='Cumulative Consumed', color='red')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Total items")
    plt.title("Cumulative Production vs Consumption")
    plt.legend()
    plt.show()
