import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from collections import Counter
from dataclasses import dataclass, field
import sim_scenarios

# --------------------------------------------------------------------------------------------------
# Running instructions 
#   RUN - python producers_consumers_many_to_many.py
# --------------------------------------------------------------------------------------------------

# ==================================================================================================
# Simulation data structures
# ==================================================================================================

class ItemType(Enum):
    IRON_INGOT = "Iron Ingot"
    IRON_ROD = "Iron Rod"
    IRON_WIRE = "Iron Wire"

@dataclass
class SimulationState:
    producer_logs: list
    consumer_logs: list
    queue_logs: list
    queues: dict[ItemType, int]
    queue_history: dict[ItemType, list[tuple[float, int]]]

@dataclass
class ProducerState:
    process_id: int
    item_type: ItemType
    next_ready_time: float = 0.0


@dataclass
class ConsumerState:
    process_id: int
    item_type: ItemType
    next_ready_time: float = 0.0

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
    # P-Control below
    target_queue_size: int | None = None
    reaction_sensitivity: float = 0.0
    feedback_delay: float = 0.0

@dataclass(frozen=True)
class ConsumerConfig:
    count: int = 0
    input: ItemType | None = None
    output: ItemType | None = None
    consumption_time: float | None = None
    # P-Control below
    target_queue_size: int | None = None
    reaction_sensitivity: float = 0.0
    feedback_delay: float = 0.0

@dataclass(frozen=True)
class NodeConfig:
    queue_size: int
    producer: ProducerConfig = ProducerConfig()
    consumer: ConsumerConfig = ConsumerConfig()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%H:%M:%S")

# ==================================================================================================
# Simulation definition
# ==================================================================================================

@dataclass(frozen=True)
class SimConfig:
    simulation_timeout_in_seconds: int = 200
    queue_interval: float = 1.0
    use_feedback: bool = False
    nodes: dict[ItemType, NodeConfig] = field(default_factory=dict)

# ==================================================================================================
# Main processes
# ==================================================================================================

def producer(state: ProducerState, simulation_state: SimulationState, sim_config: SimConfig, sim_time: float) -> None:
    """Run a producer process that generates the specified item type and places them into a queue."""
    node = sim_config.nodes[state.item_type]
    output_type = node.producer.output
    base_production_time = node.producer.production_time

    if sim_time < state.next_ready_time:
        return

    queue_size = simulation_state.queues[output_type]
    max_size = sim_config.nodes[output_type].queue_size
    if queue_size >= max_size:
        return
    
    if sim_config.use_feedback:
        next_production_time = calculate_adjusted_time(
            base_production_time, node.producer.target_queue_size, node.producer.reaction_sensitivity, 
            node.producer.feedback_delay, simulation_state.queue_history[output_type], sim_time)
    else:
        next_production_time = base_production_time

    simulation_state.queues[output_type] += 1
    simulation_state.queue_history[output_type].append((sim_time, simulation_state.queues[output_type]))
    simulation_state.producer_logs.append(SimulationLogs(state.process_id, state.item_type.value, sim_time))
    state.next_ready_time = sim_time + next_production_time

def consumer(state: ConsumerState, simulation_state: SimulationState, sim_config: SimConfig, sim_time: float) -> None:
    """Run a consumer process that retrieves and processes items from the provided queue."""
    node = sim_config.nodes[state.item_type]
    input_type = node.consumer.input
    output_type = node.consumer.output
    consumption_time = node.consumer.consumption_time

    # abort if not ready or nothing to consume
    if sim_time < state.next_ready_time or simulation_state.queues[input_type] <= 0:
        return
    
    # if consumption produces an output, consumption only occures when the output queue has space
    if output_type is not None:
        if simulation_state.queues[output_type] >= sim_config.nodes[output_type].queue_size:
            return
        
    if output_type is None or sim_config.use_feedback is False:
        adjusted_consumption_time = consumption_time
    else:
        adjusted_consumption_time = calculate_adjusted_time(
            consumption_time, node.consumer.target_queue_size, node.consumer.reaction_sensitivity,
            node.consumer.feedback_delay, simulation_state.queue_history[output_type], sim_time)

    simulation_state.queues[input_type] -= 1
    simulation_state.queue_history[input_type].append((sim_time, simulation_state.queues[input_type]))
    simulation_state.consumer_logs.append(SimulationLogs(state.process_id, state.item_type.value, sim_time))

    if output_type is not None:
        simulation_state.queues[output_type] += 1
        simulation_state.queue_history[output_type].append((sim_time, simulation_state.queues[output_type]))
        simulation_state.producer_logs.append(SimulationLogs(state.process_id, output_type.value, sim_time))

    state.next_ready_time = sim_time + adjusted_consumption_time

def get_delayed_queue_size(history: list[tuple[float, int]], current_time: float, delay: float) -> int:
    """Returns the size of the queue exactly 'delay' seconds ago."""
    target_time = current_time - delay
    # Assume empty before the delay period has passed
    if target_time <= 0.0:
        return 0
    
    for timestamp, size in reversed(history):
        if timestamp <= target_time:
            return size
    return 0

def calculate_adjusted_time(base_time: float, target_queue_size: int, reaction_sensitivity: float, 
    feedback_delay: float, queue_history: list[tuple[float, int]], sim_time: float) -> float:
    """Calculates the adjusted processing time using a smooth bounded S-curve."""
    delayed_queue = get_delayed_queue_size(queue_history, sim_time, feedback_delay)
    dif_from_target = target_queue_size - delayed_queue
    
    # Calculate the raw control signal
    control_signal = reaction_sensitivity * dif_from_target
    
    # Use exp() and atan() to gracefully bound the time variation
    # This acts as a natural limit, preventing the process from going to sleep forever
    time_multiplier = math.exp(-math.atan(control_signal))
    
    return base_time * time_multiplier
# ==================================================================================================
# Reporting - logs and diagrams
# ==================================================================================================

def log_simulation_parameters(sim_config: SimConfig) -> None:
    """Log the parameters the simulation is using."""
    logging.info(f"The simulation will be running for {sim_config.simulation_timeout_in_seconds} seconds. ")
    for item_type, node_config in sim_config.nodes.items():
        config_info = f"The {item_type.value} node has - queue size: {node_config.queue_size}"
        if node_config.producer.count > 0:
            config_info += f" | producer/s count: {node_config.producer.count} | production time(s): {node_config.producer.production_time}"
        if node_config.consumer.count > 0:
            config_info += f" | consumer/s count: {node_config.consumer.count} | consumption time(s): {node_config.consumer.consumption_time}"
        logging.info(config_info)

def log_results(simulation_state: SimulationState) -> None:
    """Log a summary of total produced and consumed items."""
    produced_items = Counter(log.item_type for log in simulation_state.producer_logs)
    consumed_items = Counter(log.item_type for log in simulation_state.consumer_logs)

    logging.info("Simulation summary:")
    all_items = sorted(set(produced_items) | set(consumed_items))
    for item in all_items:
        logging.info(
            f"Item: {item} - produced: {produced_items.get(item, 0)} | consumed: {consumed_items.get(item, 0)}"
        )

def plot_producer_consumer_rates(ax: plt.Axes, start_time: float, producer_logs: list[SimulationLogs],
                                 consumer_logs: list[SimulationLogs], bucket_size: float = 1.0) -> None:
    item_types = {log.item_type for log in producer_logs + consumer_logs}
    ordered_items = [item.value for item in ItemType if item.value in item_types]
    all_times = [log.timestamp - start_time for log in list(producer_logs) + list(consumer_logs)]
    max_t = max(all_times + [0])
    bins = np.arange(0, max_t + bucket_size, bucket_size)
    bin_centres = bins[:-1] + bucket_size / 2

    def extract_times(logs: list[SimulationLogs], item: str) -> list[float]:
        return [log.timestamp - start_time for log in logs if log.item_type == item]

    for item in ordered_items:
        prod_times = extract_times(producer_logs, item)
        cons_times = extract_times(consumer_logs, item)
        prod_counts, _ = np.histogram(prod_times, bins=bins)
        cons_counts, _ = np.histogram(cons_times, bins=bins)

        if len(prod_times) > 0:
            throughput = prod_counts
        else:
            throughput = cons_counts
        
        ax.plot(bin_centres, throughput, "--", markersize=4, alpha=0.9, label=f"Throughput: {item}")
    
    ax.set(
        xlabel="Time (seconds)",
        ylabel="Items per second",
        title=f"Production & Consumption Throughput (bucket={bucket_size}s)",
    )
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend()

def plot_queue_size_over_time(ax: plt.Axes, start_time: float, queue_logs: list[QueueLogs]) -> None:
    queues: dict[str, list[QueueLogs]] = {}

    for log in queue_logs:
        queues.setdefault(log.queue_name, []).append(log)

    ordered_names = [item.value for item in ItemType if item.value in queues]
    for queue_name in ordered_names:
        logs = queues[queue_name]

        time_steps = np.array([log.timestamp - start_time for log in logs])
        queue_usages = np.array([log.queue_usage for log in logs])
        # plot line and then shade below
        line, = ax.plot(time_steps, queue_usages, label=queue_name)
        ax.fill_between(time_steps, queue_usages, alpha=0.15, color=line.get_color())

    ax.set(
        xlabel="Time (seconds)",
        ylabel="Queue size",
        title="Queue Sizes Over Time (1-second buckets)",
    )
    ax.grid(alpha=0.4, linestyle=":")
    ax.legend()

def plot_results(simulation_state: SimulationState, start_time: float = 0.0) -> None:
    """Create one figure containing subplots."""
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    plot_producer_consumer_rates(ax1, start_time, simulation_state.producer_logs, simulation_state.consumer_logs)
    plot_queue_size_over_time(ax2, start_time, simulation_state.queue_logs)

    plt.tight_layout()
    plt.show()

# ==================================================================================================
# Simulation runners
# ==================================================================================================

def create_simulation_state(sim_config: SimConfig) -> SimulationState:
    """Create and return a SimulationState dataclass with initialized shared resources."""
    queues = {}
    queue_history = {}
    for item_type in sim_config.nodes:
        queues[item_type] = 0
        queue_history[item_type] = [(0.0, 0)]

    return SimulationState(
        producer_logs=[],
        consumer_logs=[],
        queue_logs=[],
        queues=queues,
        queue_history=queue_history
    )

def create_producer_consumer_states(sim_config: SimConfig) -> tuple[list[ProducerState], list[ConsumerState]]:
    """Create and return producer and consumer data stores."""
    producers = []
    consumers = []

    for item_type, node in sim_config.nodes.items():
        for i in range(node.producer.count):
            producers.append(ProducerState(process_id=i, item_type=item_type))
        for i in range(node.consumer.count):
            consumers.append(ConsumerState(process_id=i, item_type=item_type))

    return producers, consumers

def run_simulation(sim_config: SimConfig) -> SimulationState:
    """Run the simulation as event-driven, with the data defined in the provided sim_config dataclass """
    simulation_state = create_simulation_state(sim_config)
    producers_state, consumers_state = create_producer_consumer_states(sim_config)
    sim_time = 0.0
    duration = sim_config.simulation_timeout_in_seconds
    queue_interval = sim_config.queue_interval
    next_queue_log_time = 0.0
    processes = producers_state + consumers_state

    while sim_time < duration:
        # execute all ready processes
        for state in processes:
            if state.next_ready_time <= sim_time:
                if isinstance(state, ProducerState):
                    producer(state, simulation_state, sim_config, sim_time)
                else:
                    consumer(state, simulation_state, sim_config, sim_time)

        next_event_time = min((p.next_ready_time for p in processes if p.next_ready_time > sim_time), default=None)
        if next_event_time is None or next_event_time > duration:
            break

        # log queue sizes at the defined fixed intervals
        while next_queue_log_time <= next_event_time:
            for item_type, size in simulation_state.queues.items():
                simulation_state.queue_logs.append(QueueLogs(item_type.value, size, next_queue_log_time))
            next_queue_log_time += queue_interval
        
        sim_time = next_event_time
    return simulation_state

# ==================================================================================================
# Main function
# ==================================================================================================

def main() -> None:
    """Main function for running the simulation."""

    sim_config = sim_scenarios.get_smooth_waves()
    sim_state = run_simulation(sim_config)

    log_simulation_parameters(sim_config)
    log_results(sim_state)
    plot_results(sim_state)

if __name__ == '__main__':
    main()