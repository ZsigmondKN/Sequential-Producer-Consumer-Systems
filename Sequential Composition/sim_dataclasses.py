from enum import Enum
from dataclasses import dataclass, field

# ==================================================================================================
# Simulation data structures
# ==================================================================================================

class ItemType(Enum):
    IRON_INGOT = "Iron Ingot"
    IRON_ROD = "Iron Rod"
    IRON_WIRE = "Iron Wire"

class FeedbackType(Enum):
    OUTPUT = "output_queue"
    INPUT = "input_queue"
    DUAL = "dual_queue"

@dataclass
class SimulationState:
    producer_logs: list
    consumer_logs: list
    queue_logs: list
    queues: dict[ItemType, int]
    queue_history: dict[ItemType, list[tuple[float, int]]]
    pending_outputs: list[tuple[float, ItemType]]

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

@dataclass(frozen=True)
class SimConfig:
    simulation_timeout_in_seconds: int
    queue_interval: float
    use_feedback: bool
    feedback_type: FeedbackType
    nodes: dict[ItemType, NodeConfig]