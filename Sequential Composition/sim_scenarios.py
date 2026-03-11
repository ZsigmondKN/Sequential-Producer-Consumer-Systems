# sim_scenarios.py

from producers_consumers_many_to_many import (
    ItemType, SimConfig, NodeConfig, ProducerConfig, ConsumerConfig
)

def get_smooth_waves() -> SimConfig:
    """A balanced setup that creates beautiful, sustained, rolling waves."""
    return SimConfig(
        simulation_timeout_in_seconds=250,
        queue_interval=1.0,
        use_feedback=True,
        nodes={
            ItemType.IRON_INGOT: NodeConfig(
                queue_size=250,  
                producer=ProducerConfig(
                    count=1, 
                    output=ItemType.IRON_INGOT, 
                    production_time=1.0,        
                    target_queue_size=125, 
                    reaction_sensitivity=0.05, 
                    feedback_delay=12.0         
                ),
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_INGOT, 
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0, 
                    target_queue_size=125, 
                    reaction_sensitivity=0.05, 
                    feedback_delay=12.0
                ),
            ),
            ItemType.IRON_ROD: NodeConfig(
                queue_size=250,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_ROD, 
                    output=ItemType.IRON_WIRE,
                    consumption_time=1.0, 
                    target_queue_size=125, 
                    reaction_sensitivity=0.05, 
                    feedback_delay=12.0
                ),
            ),
            ItemType.IRON_WIRE: NodeConfig(
                queue_size=250,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_WIRE, 
                    consumption_time=1.0
                ),
            )
        }
    )

def get_something() -> SimConfig:
    """A balanced setup that creates beautiful, sustained, rolling waves."""
    return SimConfig(
        simulation_timeout_in_seconds=60,
        queue_interval=1.0,
        use_feedback=False,
        nodes={
            ItemType.IRON_INGOT: NodeConfig(
                queue_size=10,  
                producer=ProducerConfig(
                    count=1, 
                    output=ItemType.IRON_INGOT, 
                    production_time=0.5
                ),
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_INGOT, 
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5
                ),
            ),
            ItemType.IRON_ROD: NodeConfig(
                queue_size=10,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_ROD, 
                    output=ItemType.IRON_WIRE,
                    consumption_time=0.5
                ),
            ),
            ItemType.IRON_WIRE: NodeConfig(
                queue_size=10,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_WIRE, 
                    consumption_time=1.0
                ),
            )
        }
    )