# Repository Objectives

This repository contains Python implementations of producer-consumer systems used in my dissertation. The objective is to simulate sequentially composed producer-consumer systems and to observe the dynamics that are produced.

## Structure

### 01_multiprocessing
Implementations using Python’s `multiprocessing` library.
- **producer_consumer_basic.py** — Minimal single producer-consumer example.
- **producer_consumer_busy_wait.py** — Busy waiting implementation.
- **producer_consumer_waiting.py** — Using synchronization primitives.
- **producers_multiple_consumers_multiple.py** — Multiple producers and consumers.

### 02_multithreading
Implementations using Python’s `threading` library.
- **producer_consumer_one_to_one.py** — Single producer and single consumer.
- **producer_consumer_one_to_many.py** — One producer, multiple consumers.
- **producers_many_consumer_one.py** — Multiple producers, single consumer.
- **producers_many_consumers_many.py** — Multiple producers and multiple consumers.