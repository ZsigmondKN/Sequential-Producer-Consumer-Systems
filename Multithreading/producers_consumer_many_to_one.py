import queue
import random
import threading
from time import sleep

# create a shared queue
buffer = queue.Queue(maxsize=10)

# create a shared barrier for producers
n_producers = 3
barrier = threading.Barrier(n_producers)

# producer task
def producer(barrier: threading.Barrier, buffer: queue.Queue, identifier: int):
    print(f"Producer {identifier}: Running")
    for i in range(5):
        # generate a value
        value = round(random.random(), 2)
        # block, to simulate effort
        sleep(value)
        # create a tuple of index and value
        item = (i, value)
        # add item to buffer
        buffer.put(item)
        # report progress
        print(f"++ Producer {identifier}: produced {item}")
    # wait for other producers
    barrier.wait()
    print(f"Producer {identifier}: Done")
    # signal completion
    if identifier == 0:
        buffer.put(None)

# consumer task
def consumer(buffer: queue.Queue):
    print("Consumer: Running")
    while True:
        # retrieve an item from buffer
        item = buffer.get()
        # check for completion signal
        if item is None:
            break
        #block, to simulate effort
        sleep(item[1])
        # report progress
        print(f"-- Consumer: consumed {item}")
    # display completion message
    print("Consumer: Done")


# create a producer thread
producer_threads = [threading.Thread(target=producer, args=(barrier, buffer, i)) for i in range(n_producers)]
# start the producer threads
for producer_thread in producer_threads:
    producer_thread.start()

# create a consumer thread
consumer_thread = threading.Thread(target=consumer, args=(buffer,))
# start the consumer thread
consumer_thread.start()

# wait for all threads to finish
for producer_thread in producer_threads:
    producer_thread.join()
consumer_thread.join()