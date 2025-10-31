import multiprocessing
import random
import time
from queue import Empty

# producer task
def producer(queue: multiprocessing.Queue):
    print("Producer: Running", flush=True)
    for i in range(10):
        # generate a value
        value = round(random.random(), 2)
        # block
        time.sleep(value)
        # add to the queue
        queue.put(value)
        print(f"++ Producer: produced {value}", flush=True)
    # add done
    queue.put(None)
    print("Producer: Done", flush=True)

# consumer task
def consumer(queue: multiprocessing.Queue):
    print("Consumer: Running", flush=True)
    while True:
        # get a unit of work
        try:
            item = queue.get(timeout=0.5)
        except Empty:
            print("// Consumer: gave up waiting...", flush=True)
            continue
        # check for stop
        if item is None:
            break
        print(f"-- Consumer: consumed {item}", flush=True)
    # all done
    print("Consumer: Done", flush=True)

# entry point
if __name__ == '__main__':
    # create the shared queue
    queue = multiprocessing.Queue()
    # start the consumer process
    consumer_process = multiprocessing.Process(target=consumer, args=(queue,))
    consumer_process.start()
    # start the producer process
    producer_process = multiprocessing.Process(target=producer, args=(queue,))
    producer_process.start()
    # wait for the processes to finish
    producer_process.join()
    consumer_process.join()
    print("Main: Done", flush=True)
