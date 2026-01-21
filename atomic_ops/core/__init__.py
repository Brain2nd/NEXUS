"""
Core SNN components - Logic gates, neurons, and spike mode control
"""
from .spike_mode import SpikeMode
from .training_mode import TrainingMode
from .accumulator import (
    Accumulator,
    SequentialAccumulator,
    ParallelAccumulator,
    PartialProductAccumulator,
    create_accumulator,
    create_partial_product_accumulator,
)
from .neurons import SimpleIFNode, SimpleLIFNode, DynamicThresholdIFNode, SignBitNode
from .logic_gates import _create_neuron
from .logic_gates import *
from .vec_logic_gates import (
    VecAND, VecOR, VecNOT, VecXOR, VecMUX,
    VecORTree, VecANDTree,
    VecHalfAdder, VecFullAdder,
    VecAdder, VecSubtractor, VecComparator
)

# STE (Straight-Through Estimator) for SNN training
from .ste import (
    # Autograd Functions
    STELinearFunction,
    STEEmbeddingFunction,
    STERMSNormFunction,
    STELayerNormFunction,
    STEExpFunction,
    STESigmoidFunction,
    STETanhFunction,
    STESiLUFunction,
    STEGELUFunction,
    STESoftmaxFunction,
    STEReLUFunction,
    # Convenience wrapper functions
    ste_linear,
    ste_embedding,
    ste_rmsnorm,
    ste_layernorm,
    ste_exp,
    ste_sigmoid,
    ste_tanh,
    ste_silu,
    ste_gelu,
    ste_softmax,
    ste_relu,
)
