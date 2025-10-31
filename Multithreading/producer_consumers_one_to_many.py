import queue
import random
import threading
from time import sleep

n_consumers = 3

# create a shared queue
buffer = queue.Queue(maxsize=10)

# producer task
def producer(buffer: queue.Queue):
    print("Producer: Running")
    for i in range(10):
        # generate a value
        value = round(random.random(), 2)
        # block, to simulate effort
        sleep(value)
        # create a tuple of index and value
        item = (i, value)
        # add item to buffer
        buffer.put(item)
        # report progress
        print(f"++ Producer: produced {item}")
    # signal completion
    buffer.put(None)
    print("Producer: Done")

# consumer task
def consumer(buffer: queue.Queue, identifier: int):
    print(f"Consumer {identifier}: Running")
    while True:
        # retrieve an item from buffer
        item = buffer.get()
        # check for completion signal
        if item is None:
            # add the signal back for other consumers
            buffer.put(item)
            # stop running
            break
        #block, to simulate effort
        sleep(item[1])
        # report progress
        print(f"-- Consumer {identifier}: consumed {item}")
    # display completion message
    print(f"Consumer {identifier}: Done")

# create a producer thread
producer_thread = threading.Thread(target=producer, args=(buffer,))
# start the producer thread
producer_thread.start()

# create consumer threads
consumer_threads = [threading.Thread(target=consumer, args=(buffer, i)) for i in range(n_consumers)]
# start the consumer threads
for consumer_thread in consumer_threads:
    consumer_thread.start()

# wait for all threads to finish
producer_thread.join()
for consumer_thread in consumer_threads:
    consumer_thread.join()