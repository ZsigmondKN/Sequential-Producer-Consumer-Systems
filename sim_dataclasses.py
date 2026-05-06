from enum import Enum
from dataclasses import dataclass, field

# ==================================================================================================
# Simulation data structures
# ==================================================================================================

class ItemType(Enum):
    IRON_INGOT = "Iron Ingot"
    IRON_ROD = "Iron Rod"
    IRON_WIRE = "Iron Wire"
    IRON_MESH = "Iron Mesh"
    IRON_FILTER = "Iron Filter"

class FeedbackType(Enum):
    OUTPUT = "output_queue"
    INPUT = "input_queue"
    DUAL = "dual_queue"

@dataclass
class FailureEvent:
    item_type: ItemType
    start_time: float
    end_time: float

@dataclass
class SurgeEvent:
    item_type: ItemType
    trigger_time: float
    fill_to_capacity: bool = True
    amount: int | None = None

@dataclass
class SimulationState:
    producer_logs: list
    consumer_logs: list
    queue_logs: list
    queues: dict[ItemType, int]
    queue_history: dict[ItemType, list[tuple[float, int]]]
    pending_outputs: list[tuple[float, ItemType]]
    failures: list[FailureEvent]
    surges: list[SurgeEvent]

@dataclass
class ProducerState:
    process_id: int
    item_type: ItemType
    next_ready_time: float = 0.0
    error_integral: float = 0.0
    last_update_time: float = 0.0
    control_signal: float = 0.0

@dataclass
class ConsumerState:
    process_id: int
    item_type: ItemType
    next_ready_time: float = 0.0
    error_integral: float = 0.0
    last_update_time: float = 0.0
    control_signal: float = 0.0

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
    # PI-Control below
    reference_signal: int | None = None
    proportional_gain: float = 0.0
    transport_lag: float = 0.0
    integral_gain: float = 0.0

@dataclass(frozen=True)
class ConsumerConfig:
    count: int = 0
    input: ItemType | None = None
    output: ItemType | None = None
    consumption_time: float | None = None
    # PI-Control below
    reference_signal: int | None = None
    proportional_gain: float = 0.0
    transport_lag: float = 0.0
    integral_gain: float = 0.0

@dataclass(frozen=True)
class ProcessConfig:
    queue_capacity: int
    producer: ProducerConfig = ProducerConfig()
    consumer: ConsumerConfig = ConsumerConfig()

@dataclass(frozen=True)
class SimConfig:
    simulation_timeout_in_seconds: int
    queue_interval: float
    use_feedback: bool
    feedback_type: FeedbackType
    processes: dict[ItemType, ProcessConfig]
    initial_queue_occupancy: dict[ItemType, int] = field(default_factory=dict)