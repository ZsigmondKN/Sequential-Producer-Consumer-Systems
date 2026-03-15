# sim_scenarios.py

import numpy as np

from sim_dataclasses import (
    ItemType, SimConfig, NodeConfig, ProducerConfig, ConsumerConfig, FeedbackType
)

def get_multiple_oscillations_dual_f() -> tuple[SimConfig, dict]:
    """Configuration extracted from the optimized run producing stable oscillations."""

    stability_config = {
        "sensitivities": np.linspace(0.01, 5, 25),
        "delays": np.linspace(1, 100, 25),
    }

    sim_config = SimConfig(
        simulation_timeout_in_seconds=800,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type=FeedbackType.DUAL,
        nodes={
            ItemType.IRON_INGOT: NodeConfig(
                queue_size=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=0.5,
                    target_queue_size=50,
                    reaction_sensitivity=0.09996484739955049,
                    feedback_delay=19.06984611636426,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5,
                    target_queue_size=50,
                    reaction_sensitivity=0.09996484739955049,
                    feedback_delay=19.06984611636426,
                ),
            ),
            ItemType.IRON_ROD: NodeConfig(
                queue_size=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_ROD,
                    output=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    target_queue_size=50,
                    reaction_sensitivity=0.09996484739955049,
                    feedback_delay=19.06984611636426,
                ),
            ),
            ItemType.IRON_WIRE: NodeConfig(
                queue_size=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    target_queue_size=50,
                    reaction_sensitivity=0.09996484739955049,
                    feedback_delay=19.06984611636426,
                ),
            ),
        },
    )

    return sim_config, stability_config

def get_multiple_oscillations_input_f() -> tuple[SimConfig, dict]:
    """Configuration extracted from the optimized run producing stable oscillations."""

    stability_config = {
        "sensitivities": np.linspace(0.01, 5, 25),
        "delays": np.linspace(1, 100, 25),
    }

    sim_config = SimConfig(
        simulation_timeout_in_seconds=500,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type=FeedbackType.INPUT,
        nodes={
            ItemType.IRON_INGOT: NodeConfig(
                queue_size=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=0.5,
                    target_queue_size=50,
                    reaction_sensitivity=0.016486536720973884,
                    feedback_delay=25.657783155731092,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5,
                    target_queue_size=50,
                    reaction_sensitivity=0.016486536720973884,
                    feedback_delay=25.657783155731092,
                ),
            ),
            ItemType.IRON_ROD: NodeConfig(
                queue_size=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_ROD,
                    output=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    target_queue_size=50,
                    reaction_sensitivity=0.016486536720973884,
                    feedback_delay=25.657783155731092,
                ),
            ),
            ItemType.IRON_WIRE: NodeConfig(
                queue_size=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    target_queue_size=50,
                    reaction_sensitivity=0.016486536720973884,
                    feedback_delay=25.657783155731092,
                ),
            ),
        },
    )

    return sim_config, stability_config

def get_multiple_oscillations_output_f() -> tuple[dict, SimConfig]:
    """Configuration extracted from the optimized run producing stable oscillations."""
    stability_config = {
        "sensitivities" : np.linspace(0.01, 5, 25),
        "delays" : np.linspace(1, 100, 25)
    }
    sim_config =  SimConfig(
        simulation_timeout_in_seconds=500,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type = FeedbackType.OUTPUT,
        nodes={
            ItemType.IRON_INGOT: NodeConfig(
                queue_size=100,  
                producer=ProducerConfig(
                    count=1, 
                    output=ItemType.IRON_INGOT, 
                    production_time=1.0,        
                    target_queue_size=50, 
                    reaction_sensitivity=0.1437399434876802, 
                    feedback_delay=19.651752527891375         
                ),
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_INGOT, 
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0, 
                    target_queue_size=50, 
                    reaction_sensitivity=0.1437399434876802, 
                    feedback_delay=19.651752527891375
                ),
            ),

            ItemType.IRON_ROD: NodeConfig(
                queue_size=100,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_ROD, 
                    output=ItemType.IRON_WIRE,
                    consumption_time=1.0, 
                    target_queue_size=50, 
                    reaction_sensitivity=0.1437399434876802, 
                    feedback_delay=19.651752527891375
                ),
            ),

            ItemType.IRON_WIRE: NodeConfig(
                queue_size=100,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_WIRE, 
                    consumption_time=1.0,
                    target_queue_size=50, 
                    reaction_sensitivity=0.1437399434876802, 
                    feedback_delay=19.651752527891375
                ),
            )
        }
    )
    return sim_config, stability_config

def get_a_single_oscillation() -> SimConfig:
    """Configuration extracted from the optimized run producing stable oscillations."""
    return SimConfig(
        simulation_timeout_in_seconds=500,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type = FeedbackType.OUTPUT,
        nodes={
            ItemType.IRON_INGOT: NodeConfig(
                queue_size=200,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    target_queue_size=100,
                    reaction_sensitivity=0.19647332747872684,
                    feedback_delay=19.109745987839684
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    consumption_time=1.0,
                ),
            ),
        }
    )

def get_smooth_waves() -> SimConfig:
    """A balanced setup that creates beautiful, sustained, rolling waves."""
    return SimConfig(
        simulation_timeout_in_seconds=250,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type = FeedbackType.OUTPUT,
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

def get_blocking() -> SimConfig:
    """A balanced setup that creates beautiful, sustained, rolling waves."""
    return SimConfig(
        simulation_timeout_in_seconds=60,
        queue_interval=1.0,
        use_feedback=False,
        feedback_type = FeedbackType.OUTPUT,
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