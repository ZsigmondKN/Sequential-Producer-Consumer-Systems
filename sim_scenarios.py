# sim_scenarios.py

import numpy as np

from sim_dataclasses import (
    ItemType, SimConfig, ProcessConfig, ProducerConfig, ConsumerConfig, FeedbackDirection
)

# ==================================================================================================
# Open Loop
# ==================================================================================================

def get_balanced_flow() -> tuple[SimConfig, None]:
    stability_config = None
    sim_config = SimConfig(
        simulation_timeout_in_seconds=60,
        queue_interval=1.0,
        use_feedback=False,
        feedback_direction = FeedbackDirection.OUTPUT,
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
        feedback_direction = FeedbackDirection.OUTPUT,
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
        feedback_direction = FeedbackDirection.OUTPUT,
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
        feedback_direction = FeedbackDirection.OUTPUT,
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

# ==================================================================================================
# Closed loop - baseline
# ==================================================================================================

def get_atomic_second_order_system_imbalanced() -> tuple[SimConfig, dict]:
    stability_config = {
        "mode": "global",
        "x_param": "global_proportional_gain",
        "y_param": "global_integral_gain",
        "x_values": np.linspace(0.0, 1.0, 25),
        "y_values": np.linspace(0.0, 0.1, 25),
    }
    local_p_gain = 0.0519
    local_i_gain = 0.02
    sim_config = SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction = FeedbackDirection.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=50,
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
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

# ==================================================================================================
# Closed loop - Feedback Mechanisms
# ==================================================================================================

def get_p_control_sequential_three_processes_imbalanced() -> tuple[SimConfig, dict]:
    stability_config = {
        "mode": "global",
        "x_param": "global_proportional_gain",
        "y_param": "global_integral_gain",
        "x_values": np.linspace(0.0, 1.0, 25),
        "y_values": np.linspace(0.0, 0.1, 25),
    }
    local_p_gain = 1
    local_i_gain = 0
    sim_config = SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction = FeedbackDirection.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=50,
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5,
                    reference_signal=50, 
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
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

def get_pi_control_sequential_three_processes_imbalanced() -> tuple[SimConfig, dict]:
    stability_config = {
        "mode": "global",
        "x_param": "global_proportional_gain",
        "y_param": "global_integral_gain",
        "x_values": np.linspace(0.0, 1.0, 25),
        "y_values": np.linspace(0.0, 0.1, 25),
    }
    local_p_gain = 1
    local_i_gain = 0.005
    sim_config = SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction = FeedbackDirection.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=50,
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5,
                    reference_signal=50, 
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
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

def get_pi_control_sequential_three_processes_balanced() -> tuple[SimConfig, dict]:
    stability_config = {
        "mode": "global",
        "x_param": "global_proportional_gain",
        "y_param": "global_integral_gain",
        "x_values": np.linspace(0.0, 1.0, 25),
        "y_values": np.linspace(0.0, 0.1, 25),
    }
    local_p_gain = 0.2628
    local_i_gain = 0.0252
    sim_config = SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction = FeedbackDirection.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=50,
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0,
                    reference_signal=50, 
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
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

def get_p_control_with_delay_sequential_three_processes_imbalanced() -> tuple[SimConfig, dict]:
    stability_config = {
        "mode": "global",
        "x_param": "global_proportional_gain",
        "y_param": "global_transport_lag",
        "x_values": np.linspace(0.0, 5.0, 25),
        "y_values": np.linspace(1, 100, 25),
    }
    local_p_gain = 0.85
    local_t_lag = 50
    sim_config = SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction = FeedbackDirection.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=50,
                    proportional_gain=local_p_gain,
                    transport_lag=local_t_lag
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5,
                    reference_signal=50, 
                    proportional_gain=local_p_gain,
                    transport_lag=local_t_lag
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

def get_pi_control_with_delay_sequential_three_processes_imbalanced() -> tuple[SimConfig, dict]:
    stability_config = {
        "mode": "global",
        "x_param": "global_proportional_gain",
        "y_param": "global_transport_lag",
        "x_values": np.linspace(0.0, 20.0, 25),
        "y_values": np.linspace(1, 100, 25),
    }
    local_p_gain = 1.2576455
    local_i_gain = 0.0006293
    local_t_lag = 26.1150727
    sim_config = SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction = FeedbackDirection.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=50,
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
                    transport_lag=local_t_lag
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5,
                    reference_signal=50, 
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
                    transport_lag=local_t_lag
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

# ==================================================================================================
# Closed loop - Sequence Length Experiment
# ==================================================================================================

def get_input_pi_control_sequential_three_processes_imbalanced() -> tuple[SimConfig, dict]:
    stability_config = {
        "mode": "global",
        "x_param": "global_proportional_gain",
        "y_param": "global_integral_gain",
        "x_values": np.linspace(0.0, 1.0, 25),
        "y_values": np.linspace(0.0, 0.1, 25),
    }
    local_p_gain = 0.2628
    local_i_gain = 0.0252
    sim_config =  SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction = FeedbackDirection.INPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,  
                producer=ProducerConfig(
                    count=1, 
                    output=ItemType.IRON_INGOT, 
                    production_time=0.5,        
                    reference_signal=50, 
                    proportional_gain=local_p_gain, 
                    integral_gain=local_i_gain,
                ),
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_INGOT, 
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5, 
                    reference_signal=50, 
                    proportional_gain=local_p_gain, 
                    integral_gain=local_i_gain,
                ),
            ),
            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_ROD,
                    consumption_time=1.0,
                    proportional_gain=local_p_gain, 
                    integral_gain=local_i_gain,
                ),
            ),
        }
    )
    return sim_config, stability_config

def get_dual_pi_control_sequential_three_processes_imbalanced() -> tuple[SimConfig, dict]:
    stability_config = {
        "mode": "global",
        "x_param": "global_proportional_gain",
        "y_param": "global_integral_gain",
        "x_values": np.linspace(0.0, 1.0, 25),
        "y_values": np.linspace(0.0, 0.1, 25),
    }
    local_p_gain = 0.2628
    local_i_gain = 0.0252
    sim_config =  SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction = FeedbackDirection.DUAL,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,  
                producer=ProducerConfig(
                    count=1, 
                    output=ItemType.IRON_INGOT, 
                    production_time=0.5,        
                    reference_signal=50, 
                    proportional_gain=local_p_gain, 
                    integral_gain=local_i_gain,
                ),
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_INGOT, 
                    output=ItemType.IRON_ROD,
                    consumption_time=0.5, 
                    reference_signal=50, 
                    proportional_gain=local_p_gain, 
                    integral_gain=local_i_gain,
                ),
            ),
            ItemType.IRON_ROD: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_ROD,
                    consumption_time=1.0,
                    proportional_gain=local_p_gain, 
                    integral_gain=local_i_gain,
                ),
            ),
        }
    )
    return sim_config, stability_config

# ==================================================================================================
# Closed loop - Sequence Length Experiment
# ==================================================================================================

def get_pi_control_sequential_four_processes_balanced() -> tuple[SimConfig, dict]:
    stability_config = {
        "mode": "global",
        "x_param": "global_proportional_gain",
        "y_param": "global_integral_gain",
        "x_values": np.linspace(0.0, 1.0, 25),
        "y_values": np.linspace(0.0, 0.1, 25),
    }
    local_p_gain = 0.3723502
    local_i_gain = 0.0214396
    sim_config = SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction = FeedbackDirection.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=50,
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0,
                    reference_signal=50, 
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
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
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
                ),
            ),
            ItemType.IRON_WIRE: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_WIRE,
                    consumption_time=1.0,
                    reference_signal=50,
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
                ),
            ),
        }
    )

    return sim_config, stability_config

def get_pi_control_sequential_five_processes_balanced() -> tuple[SimConfig, dict]:
    stability_config = {
        "mode": "global",
        "x_param": "global_proportional_gain",
        "y_param": "global_integral_gain",
        "x_values": np.linspace(0.0, 1.0, 25),
        "y_values": np.linspace(0.0, 0.1, 25),
    }
    local_p_gain = 0.0956814
    local_i_gain = 0.0137609
    sim_config = SimConfig(
        simulation_timeout_in_seconds=1000,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction = FeedbackDirection.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,
                producer=ProducerConfig(
                    count=1,
                    output=ItemType.IRON_INGOT,
                    production_time=1.0,
                    reference_signal=50,
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
                ),
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_INGOT,
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0,
                    reference_signal=50, 
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
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
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
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
                    proportional_gain=local_p_gain,
                    integral_gain=local_i_gain,
                ),
            ),
            ItemType.IRON_MESH: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1,
                    input=ItemType.IRON_MESH,
                    consumption_time=1.0,
                    reference_signal=50,
                ),
            ),
        }
    )

    return sim_config, stability_config

# ==================================================================================================
# Closed loop - Old Delay Experiments
# ==================================================================================================

def get_multiple_oscillations_output_f() -> tuple[dict, SimConfig]:
    """Configuration extracted from the optimized run producing stable oscillations."""
    stability_config = {
        "mode": "global",
        "x_param": "global_proportional_gain",
        "y_param": "global_transport_lag",
        "x_values": np.linspace(0.01, 5, 25),
        "y_values": np.linspace(1, 100, 25),
    }
    sim_config =  SimConfig(
        simulation_timeout_in_seconds=800,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction = FeedbackDirection.OUTPUT,
        processes={
            ItemType.IRON_INGOT: ProcessConfig(
                queue_capacity=100,  
                producer=ProducerConfig(
                    count=1, 
                    output=ItemType.IRON_INGOT, 
                    production_time=1.0,        
                    reference_signal=50, 
                    proportional_gain=2, 
                    transport_lag=19.651752527891375,
                    integral_gain = 0.0        
                ),
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_INGOT, 
                    output=ItemType.IRON_ROD,
                    consumption_time=1.0, 
                    reference_signal=50, 
                    proportional_gain=2, 
                    transport_lag=19.651752527891375,
                    integral_gain = 0.0
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
                    proportional_gain=2, 
                    transport_lag=19.651752527891375,
                    integral_gain = 0.0,
                ),
            ),

            ItemType.IRON_WIRE: ProcessConfig(
                queue_capacity=100,
                consumer=ConsumerConfig(
                    count=1, 
                    input=ItemType.IRON_WIRE, 
                    consumption_time=1.0,
                    reference_signal=50, 
                    proportional_gain=1, 
                    transport_lag=19.651752527891375,
                    integral_gain = 0.0
                ),
            )
        }
    )
    return sim_config, stability_config

def get_multiple_oscillations_dual_f() -> tuple[SimConfig, dict]:
    """Configuration extracted from the optimized run producing stable oscillations."""
    stability_config = {
        "mode": "global",
        "x_param": "global_proportional_gain",
        "y_param": "global_transport_lag",
        "x_values": np.linspace(0.01, 5, 25),
        "y_values": np.linspace(1, 100, 25),
    }
    sim_config = SimConfig(
        simulation_timeout_in_seconds=800,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction=FeedbackDirection.DUAL,
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
        "x_param": "global_proportional_gain",
        "y_param": "global_transport_lag",
        "x_values": np.linspace(0.01, 5, 25),
        "y_values": np.linspace(1, 100, 25),
    }
    sim_config = SimConfig(
        simulation_timeout_in_seconds=500,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction=FeedbackDirection.INPUT,
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

def get_a_single_oscillation() -> tuple[SimConfig, None]:
    """Configuration extracted from the optimized run producing stable oscillations."""
    stability_config = None
    sim_config = SimConfig(
        simulation_timeout_in_seconds=500,
        queue_interval=1.0,
        use_feedback=True,
        feedback_direction = FeedbackDirection.OUTPUT,
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
        feedback_direction = FeedbackDirection.OUTPUT,
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