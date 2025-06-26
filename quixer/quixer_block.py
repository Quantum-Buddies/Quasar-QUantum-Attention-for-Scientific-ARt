"""quantum_transformers.quixer.quixer_block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A PyTorch-compatible implementation of the Quixer attention block.

This module implements the quantum attention mechanism described in the
Quixer paper (arXiv:2406.04305) by Quantinuum. It is designed to be a
differentiable `torch.nn.Module` that can be seamlessly integrated into
larger classical architectures like our U-Net-style Diffusion Transformer.

The core logic is implemented using PennyLane and is intended to be run on
the `lightning.gpu` backend for CUDA-accelerated simulation.

Stage 1 Implementation: LCU with a variational non-linearity (stand-in for QSVT).
"""

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import math

# --------------------------------------------------------------------------
# Helper Quantum Layers
# --------------------------------------------------------------------------

def unitary_ansatz(params, wires):
    """A simple unitary ansatz for token embedding."""
    qml.RY(params[0], wires=wires[0])
    qml.RZ(params[1], wires=wires[0])
    if len(wires) > 1:
        qml.RY(params[2], wires=wires[1])
        qml.RZ(params[3], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])

def qsvt_sequence(phi_angles, U, U_dagger, wires):
    """
    The core QSVT sequence that applies a polynomial transformation.
    
    Args:
        phi_angles (tensor): The sequence of phase angles (the trainable weights).
        U (function): The block-encoding unitary (our SELECT oracle).
        U_dagger (function): The inverse of the block-encoding unitary.
        wires (list): The wires the rotations act on.
    """
    for i in range(len(phi_angles)):
        # Simplified: assumes rotation on first data qubit. A full implementation
        # might use a more complex reflection operator.
        qml.RZ(phi_angles[i], wires=wires[0])
        
        # Alternate between U and U_dagger
        if i % 2 == 0:
            U()
        else:
            U_dagger()

def select_op(unitary_params_list, data_wires, ancilla_wires):
    """The SELECT oracle for the LCU procedure."""
    def _select():
        for i, params in enumerate(unitary_params_list):
            qml.ctrl(unitary_ansatz, control=ancilla_wires, control_values=f"{i:0{len(ancilla_wires)}b}")(params, wires=data_wires)
    return _select

# --------------------------------------------------------------------------
# Quixer Block
# --------------------------------------------------------------------------

class QuixerBlock(nn.Module):
    """
    A full implementation of the Quixer quantum attention block using LCU and QSVT.
    """
    def __init__(self, hidden_size: int, data_qubits: int, max_seq_len: int, qsvt_poly_degree: int = 5):
        super().__init__()

        self.hidden_size = hidden_size
        self.data_qubits = data_qubits
        self.max_seq_len = max_seq_len
        self.ancilla_qubits = math.ceil(math.log2(max_seq_len))
        self.total_qubits = self.data_qubits + self.ancilla_qubits

        # 1. Classical linear maps
        self.ansatz_param_count = 4 # From our unitary_ansatz
        self.token_to_unitary_params = nn.Linear(hidden_size, self.ansatz_param_count)
        self.lcu_amp_generator = nn.Sequential(
            nn.Linear(hidden_size * max_seq_len, 2**self.ancilla_qubits), # Output size must match 2^ancillas
            nn.Softmax(dim=-1)
        )
        self.readout_mapper = nn.Linear(self.data_qubits, hidden_size)

        # The QSVT phase angles are the trainable weights of the quantum layer
        self.qsvt_poly_degree = qsvt_poly_degree
        self.qsvt_phase_shape = (qsvt_poly_degree + 1,)
        
        self.qnn = self._create_torch_layer()

    def _create_torch_layer(self):
        """Creates the differentiable TorchLayer for the Quixer mechanism."""
        dev = qml.device("lightning.gpu", wires=self.total_qubits)
        
        @qml.qnode(dev, interface="torch")
        def quixer_circuit(inputs, weights):
            """
            inputs: A flat 1D tensor containing the concatenated unitary_params and lcu_amps.
            weights: The trainable QSVT phase angles.
            """
            # Unpack the flat inputs tensor
            num_unitary_params = self.max_seq_len * self.ansatz_param_count
            unitary_params_flat = inputs[:num_unitary_params]
            lcu_amps = inputs[num_unitary_params:]

            unitary_params = unitary_params_flat.reshape(self.max_seq_len, self.ansatz_param_count)

            data_wires = list(range(self.data_qubits))
            ancilla_wires = list(range(self.data_qubits, self.total_qubits))
            
            # LCU State Preparation
            qml.StatePrep(lcu_amps, wires=ancilla_wires)

            # Define the signal operator and its inverse
            select_oracle = select_op(unitary_params, data_wires, ancilla_wires)
            select_oracle_inv = qml.adjoint(select_oracle)
            
            # Apply the QSVT transformation
            qsvt_sequence(weights, U=select_oracle, U_dagger=select_oracle_inv, wires=data_wires)

            return [qml.expval(qml.PauliZ(i)) for i in data_wires]

        # Define the shape of the trainable weights for the TorchLayer
        weight_shapes = {"weights": self.qsvt_phase_shape}
        return qml.qnn.TorchLayer(quixer_circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Quixer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, hidden_size)
        
        Returns:
            Tensor of the same shape after applying quantum attention.
        """
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        # Pad or truncate sequence-dependent parameters to max_seq_len
        padded_x = torch.zeros(batch_size, self.max_seq_len, self.hidden_size, device=x.device)
        padded_x[:, :seq_len, :] = x
        
        # This part processes the entire batch. It can be slow.
        # For a real implementation, batch processing on the quantum device
        # would need to be handled by the device plugin or manual batching.
        outputs = []
        for i in range(batch_size):
            # Generate parameters from the classical input for this batch item
            single_x_flat = padded_x[i].flatten()
            
            # LCU amps need to be padded to 2**ancilla_qubits if seq_len is not a power of 2
            raw_lcu_amps = self.lcu_amp_generator(single_x_flat)
            lcu_amps = torch.zeros(2**self.ancilla_qubits, device=x.device)
            lcu_amps[:raw_lcu_amps.shape[0]] = raw_lcu_amps
            lcu_amps = lcu_amps / torch.sqrt(torch.sum(lcu_amps**2)) # Normalize for state prep

            unitary_params = self.token_to_unitary_params(padded_x[i])

            # Flatten and concatenate all inputs for the QNode into a single tensor
            qnode_inputs = torch.cat([
                unitary_params.flatten(),
                lcu_amps
            ])

            # Run the quantum circuit via the TorchLayer
            quantum_output = self.qnn(qnode_inputs)
            
            # Map quantum output back to classical hidden size
            classical_output = self.readout_mapper(quantum_output)
            
            # The quantum process outputs a single context vector. In the original
            # paper, this is combined with the token sequence. Here we create an
            # output sequence by broadcasting the context.
            # This makes it compatible with the expected output of an attention layer.
            # We expand the single context vector to the sequence length.
            # The residual connection will be handled outside this block.
            updated_x_single = classical_output.unsqueeze(0).expand(seq_len, -1)
            outputs.append(updated_x_single)

        return torch.stack(outputs)
