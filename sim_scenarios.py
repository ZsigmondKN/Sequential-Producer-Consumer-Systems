# sim_scenarios.py

import numpy as np

from sim_dataclasses import (
    ItemType, SimConfig, ProcessConfig, ProducerConfig, ConsumerConfig, FeedbackType
)

def get_balanced_flow() -> tuple[SimConfig, None]:
    stability_config = None
    sim_config = SimConfig(
        simulation_timeout_in_seconds=60,
        queue_interval=1.0,
        use_feedback=False,
        feedback_type = FeedbackType.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=10,  
                producer=ProducerConfig(
                    count=1, 
                    output=ItemType.IRON_INGOT, 
                    production_time=0.5
                ),
                consumer=ConsumerConfig(
                    count=2, 
                    input=ItemType.IRON_INGOT, 
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0
                ),
            ),
            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=10,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_ROD, 
                    consumption_time=0.5
                ),
            ),
        }
    )
    return sim_config, stability_config


def get_bottleneck() -> tuple[SimConfig, None]:
    stability_config = None
    sim_config = SimConfig(
        simulation_timeout_in_seconds=60,
        queue_interval=1.0,
        use_feedback=False,
        feedback_type = FeedbackType.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=10,  
                producer=ProducerConfig(
                    count=1, 
                    output=ItemType.IRON_INGOT, 
                    production_time=0.5
                ),
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_INGOT, 
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0
                ),
            ),
            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=10,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_ROD, 
                    consumption_time=0.5
                ),
            ),
        }
    )
    return sim_config, stability_config

def get_starvation() -> tuple[SimConfig, None]:
    stability_config = None
    sim_config = SimConfig(
        simulation_timeout_in_seconds=60,
        queue_interval=1.0,
        use_feedback=False,
        feedback_type = FeedbackType.OUTPUT,
        initial_queue_occupancy={
            ItemType.IRON_INGOT: 10,
        },
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=10,  
                producer=ProducerConfig(
                    count=1, 
                    output=ItemType.IRON_INGOT, 
                    production_time=1.0
                ),
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_INGOT, 
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5
                ),
            ),
            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=10,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_ROD, 
                    consumption_time=0.5
                ),
            ),
        }
    )
    return sim_config, stability_config

def get_backpressure_propagation() -> tuple[SimConfig, None]:
    stability_config = None
    sim_config = SimConfig(
        simulation_timeout_in_seconds=60,
        queue_interval=1.0,
        use_feedback=False,
        feedback_type = FeedbackType.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=10,  
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
            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=10,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_ROD, 
                    output=ItemType.IRON_WIRE,
                    consumption_time=0.5
                ),
            ),
            ItemType.IRON_WIRE: ProcessConfig(
                queue_capacity=10,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_WIRE, 
                    consumption_time=1.0
                ),
            )
        }
    )
    return sim_config, stability_config

def get_atomic_second_order_system() -> tuple[SimConfig, dict]:
    stability_config = None
    sim_config = SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type = FeedbackType.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.1,
                    integral_gain=0.02,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    consumption_time=1.0,
                ),
            ),
        }
    )

    return sim_config, stability_config

def get_sequential_higher_order_system_three_processes() -> tuple[SimConfig, dict]:
    stability_config = None
    sim_config = SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type = FeedbackType.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.1,
                    integral_gain=0.02,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0,
                    reference_signal=50, 
                    proportional_gain=0.1,
                    integral_gain=0.02,
                ),
            ),
            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_ROD, 
                    consumption_time=1.0, 
                ),
            ),
        }
    )

    return sim_config, stability_config

def get_sequential_higher_order_system_four_processes() -> tuple[SimConfig, dict]:
    stability_config = None
    sim_config = SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type = FeedbackType.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.1,
                    integral_gain=0.02,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0,
                    reference_signal=50, 
                    proportional_gain=0.1,
                    integral_gain=0.02,
                ),
            ),
            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_ROD,
                    output=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.1,
                    integral_gain=0.02,
                ),
            ),
            ItemType.IRON_WIRE: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.1,
                    integral_gain=0.02,
                ),
            ),
        }
    )

    return sim_config, stability_config

def get_sequential_higher_order_system_five_processes() -> tuple[SimConfig, dict]:
    stability_config = None
    sim_config = SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type = FeedbackType.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.1,
                    integral_gain=0.02,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0,
                    reference_signal=50, 
                    proportional_gain=0.1,
                    integral_gain=0.02,
                ),
            ),
            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_ROD,
                    output=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.1,
                    integral_gain=0.02,
                ),
            ),
            ItemType.IRON_WIRE: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_WIRE,
                    output=ItemType.IRON_MESH,
                    consumption_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.1,
                    integral_gain=0.02,
                ),
            ),
            ItemType.IRON_MESH: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_MESH,
                    consumption_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.1,
                    integral_gain=0.02,
                ),
            ),
        }
    )

    return sim_config, stability_config

def get_multiple_oscillations_dual_f() -> tuple[SimConfig, dict]:
    """Configuration extracted from the optimized run producing stable oscillations."""

    stability_config = {
        "mode": "global",
        "x_param": "global_sensitivity",
        "y_param": "global_delay",
        "x_values": np.linspace(0.01, 5, 25),
        "y_values": np.linspace(1, 100, 25),
    }

    sim_config = SimConfig(
        simulation_timeout_in_seconds=800,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type=FeedbackType.DUAL,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=0.5,
                    reference_signal=50,
                    proportional_gain=0.09996484739955049,
                    transport_lag=19.06984611636426,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5,
                    reference_signal=50,
                    proportional_gain=0.09996484739955049,
                    transport_lag=19.06984611636426,
                ),
            ),
            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_ROD,
                    output=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.09996484739955049,
                    transport_lag=19.06984611636426,
                ),
            ),
            ItemType.IRON_WIRE: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.09996484739955049,
                    transport_lag=19.06984611636426,
                ),
            ),
        },
    )

    return sim_config, stability_config

def get_multiple_oscillations_input_f() -> tuple[SimConfig, dict]:
    """Configuration extracted from the optimized run producing stable oscillations."""

    stability_config = {
        "mode": "global",
        "x_param": "global_sensitivity",
        "y_param": "global_delay",
        "x_values": np.linspace(0.01, 5, 25),
        "y_values": np.linspace(1, 100, 25),
    }

    sim_config = SimConfig(
        simulation_timeout_in_seconds=500,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type=FeedbackType.INPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=0.5,
                    reference_signal=50,
                    proportional_gain=0.016486536720973884,
                    transport_lag=25.657783155731092,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5,
                    reference_signal=50,
                    proportional_gain=0.016486536720973884,
                    transport_lag=25.657783155731092,
                ),
            ),
            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_ROD,
                    output=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.016486536720973884,
                    transport_lag=25.657783155731092,
                ),
            ),
            ItemType.IRON_WIRE: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    reference_signal=50,
                    proportional_gain=0.016486536720973884,
                    transport_lag=25.657783155731092,
                ),
            ),
        },
    )

    return sim_config, stability_config

def get_multiple_oscillations_output_f() -> tuple[dict, SimConfig]:
    """Configuration extracted from the optimized run producing stable oscillations."""

    stability_config = {
        "mode": "global",
        "x_param": "global_sensitivity",
        "y_param": "global_delay",
        "x_values": np.linspace(0.01, 5, 25),
        "y_values": np.linspace(1, 100, 25),
    }
    
    sim_config =  SimConfig(
        simulation_timeout_in_seconds=500,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type = FeedbackType.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,  
                producer=ProducerConfig(
                    count=1, 
                    output=ItemType.IRON_INGOT, 
                    production_time=1.0,        
                    reference_signal=50, 
                    proportional_gain=0.1437399434876802, 
                    transport_lag=19.651752527891375         
                ),
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_INGOT, 
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0, 
                    reference_signal=50, 
                    proportional_gain=0.1437399434876802, 
                    transport_lag=19.651752527891375
                ),
            ),

            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_ROD, 
                    output=ItemType.IRON_WIRE,
                    consumption_time=1.0, 
                    reference_signal=50, 
                    proportional_gain=0.1437399434876802, 
                    transport_lag=19.651752527891375
                ),
            ),

            ItemType.IRON_WIRE: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_WIRE, 
                    consumption_time=1.0,
                    reference_signal=50, 
                    proportional_gain=0.1437399434876802, 
                    transport_lag=19.651752527891375
                ),
            )
        }
    )
    return sim_config, stability_config

def get_a_single_oscillation() -> tuple[SimConfig, None]:
    """Configuration extracted from the optimized run producing stable oscillations."""
    stability_config = None
    sim_config = SimConfig(
        simulation_timeout_in_seconds=500,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type = FeedbackType.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=200,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=100,
                    proportional_gain=0.19647332747872684,
                    transport_lag=19.109745987839684
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    consumption_time=1.0,
                ),
            ),
        }
    )
    return sim_config, stability_config

def get_smooth_waves() -> tuple[SimConfig, None]:
    """A balanced setup that creates beautiful, sustained, rolling waves."""
    stability_config = None
    sim_config = SimConfig(
        simulation_timeout_in_seconds=250,
        queue_interval=1.0,
        use_feedback=True,
        feedback_type = FeedbackType.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=250,  
                producer=ProducerConfig(
                    count=1, 
                    output=ItemType.IRON_INGOT, 
                    production_time=1.0,        
                    reference_signal=125, 
                    proportional_gain=0.05, 
                    transport_lag=12.0         
                ),
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_INGOT, 
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0, 
                    reference_signal=125, 
                    proportional_gain=0.05, 
                    transport_lag=12.0
                ),
            ),
            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=250,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_ROD, 
                    output=ItemType.IRON_WIRE,
                    consumption_time=1.0, 
                    reference_signal=125, 
                    proportional_gain=0.05, 
                    transport_lag=12.0
                ),
            ),
            ItemType.IRON_WIRE: ProcessConfig(
                queue_capacity=250,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_WIRE, 
                    consumption_time=1.0
                ),
            )
        }
    )
    return sim_config, stability_config