import os
import numpy as np
from typing import Dict, Any, List, Tuple
from qiskit import QuantumCircuit
import qiskit_superstaq as qss
from ..quantum.circuit_manager import build_bb84_circuit, measure_bb84, circuit_png_data_uri, bb84_stats_data_uri
from .base import ProtocolSession

from qkd_webapp.config import Config

provider = qss.SuperstaqProvider(api_key=Config.SUPERSTAQ_API_KEY)
backend = provider.get_backend("cq_sqale_simulator")

class BB84Session(ProtocolSession):
    """Full BB84 protocol with Superstaq measurement like CLI."""

    def __init__(self, room_id: str):
        super().__init__(room_id)
        self.state = 'init'  # init -> alice_prepared -> eve_intercept -> bob_measured -> done
        self.alice_bits = None
        self.alice_bases = None
        self.bob_bits = None
        self.bob_bases = None
        self.circuit = None
        self.final_circuit = None
        self.eve_forged_circuit = None
        self.eve_present = False
        self.num_qubits = 0

    async def handle(self, sender: str, msg: Dict[str, Any]):
        action = msg.get('action')
        to_send: List[Tuple[str, Dict[str, Any]]] = []

        # Alice starts protocol
        if self.state == 'init' and sender == 'alice' and action == 'start':
            self.num_qubits = msg.get('num_qubits', 9)
            self.eve_present = msg.get('_eve_present', False)
            
            # Alice generates random bits/bases and prepares the corresponding quantum circuit
            print(f"Alice is preparing {self.num_qubits} qubits...")
            self.alice_bits = np.random.randint(2, size=self.num_qubits)
            self.alice_bases = np.random.randint(2, size=self.num_qubits)  # 0 for Z-basis, 1 for X-basis
            print(f"  Alice's private bits:  {self.alice_bits}")
            print(f"  Alice's private bases: {self.alice_bases}")

            # Create quantum circuit
            qc = QuantumCircuit(self.num_qubits, self.num_qubits)
            for i in range(self.num_qubits):
                if self.alice_bits[i] == 1:
                    qc.x(i)  # Prepare |1>
                if self.alice_bases[i] == 1:
                    qc.h(i)  # Change to X-basis
            
            self.circuit = qc
            
            # Send to Alice (notification)
            payload = {
                'event': 'alice_prepared',
                'bits': self.alice_bits.tolist(),
                'bases': self.alice_bases.tolist(), 
                'circuit_image': circuit_png_data_uri(qc),
                'num_qubits': self.num_qubits
            }
            to_send.append(('alice', payload))
            
            # Handle Eve or direct to Bob
            if self.eve_present:
                print("DEBUG: Eve is present, sending to Eve first")
                to_send.append(('eve', {
                    'event': 'qubits_intercept', 
                    'num_qubits': self.num_qubits, 
                    'protocol': 'BB84'
                }))
            else:
                print("DEBUG: Sending directly to Bob (no Eve)")
                to_send.append(('bob', {
                    'event': 'qubits_received', 
                    'num_qubits': self.num_qubits, 
                    'protocol': 'BB84', 
                    'source': 'Alice'
                }))
            
            # Alice waits for classical communication
            to_send.append(('alice', {
                'event': 'classical_waiting', 
                'message': 'Classical channel open. Waiting for Bob...'
            }))
            self.state = 'alice_prepared'

        # Eve intercepts and forwards modified circuit
        elif self.state == 'alice_prepared' and sender == 'eve' and action == 'eve_forward':
            qc_from_alice = self.circuit.copy()
            
            print("Eve is intercepting and measuring the qubits...")
            eve_bases = np.random.randint(2, size=self.num_qubits)
            print(f"  Eve's private bases: {eve_bases}")

            # Eve measures in her chosen bases
            qc_eve_measure = qc_from_alice.copy()
            for i in range(self.num_qubits):
                if eve_bases[i] == 1:  # X-basis measurement
                    qc_eve_measure.h(i)
            qc_eve_measure.measure_all()

            print("Eve is running simulation to get measurement results...")
            try:
                job = backend.run(qc_eve_measure, method="dry-run", shots=1024)
                result = job.result()
                counts = result.get_counts()
                if counts:
                    measured_string = max(counts.keys(), key=counts.get)  # Get most frequent result
                    # Reverse the bit string to match qubit order
                    eve_bits = [int(b) for b in measured_string[::-1]]
                else:
                    # Fallback to random bits if measurement fails
                    eve_bits = np.random.randint(2, size=self.num_qubits).tolist()
            except Exception as e:
                print(f"Eve's measurement failed: {e}, using random bits")
                eve_bits = np.random.randint(2, size=self.num_qubits).tolist()

            print(f"  Eve's measured bits: {eve_bits}")

            # Eve prepares new qubits based on her results to send to Bob
            print("Eve is preparing new (fake) qubits to send to Bob...")
            qc_to_bob = QuantumCircuit(self.num_qubits, self.num_qubits)
            for i in range(self.num_qubits):
                if eve_bits[i] == 1:
                    qc_to_bob.x(i)
                if eve_bases[i] == 1:  # Prepare in the same basis she measured in
                    qc_to_bob.h(i)

            # Replace circuit with Eve's forged circuit
            self.circuit = qc_to_bob
            self.eve_forged_circuit = qc_to_bob
            print("Fake qubits sent to Bob. Eve's work is done.")
            
            # Send notifications
            to_send.append(('eve', {
                'event': 'eve_forwarded', 
                'message': 'Modified circuit sent to Bob.'
            }))
            to_send.append(('bob', {
                'event': 'qubits_received', 
                'num_qubits': self.num_qubits, 
                'protocol': 'BB84', 
                'source': 'Eve'
            }))

        # Bob measures qubits
        elif self.state == 'alice_prepared' and sender == 'bob' and action == 'measure':
            qc_received = self.circuit.copy()
            
            print(f"Bob is measuring {self.num_qubits} qubits...")
            self.bob_bases = np.random.randint(2, size=self.num_qubits)
            print(f"  Bob's private bases: {self.bob_bases}")

            # Add measurement gates based on Bob's choices
            qc_received.barrier()
            for i in range(self.num_qubits):
                if self.bob_bases[i] == 1:
                    qc_received.h(i)
            
            # Ensure classical registers exist before measuring
            if qc_received.num_clbits < self.num_qubits:
                qc_received.add_register(qiskit.ClassicalRegister(self.num_qubits - qc_received.num_clbits))
            
            qc_received.measure_all()

            print("Running circuit on Superstaq simulator...")
            try:
                job = backend.run(qc_received, method="dry-run", shots=1024)
                result = job.result()
                counts = result.get_counts()
                if counts:
                    measured_string = max(counts.keys(), key=counts.get)  # Get most frequent result
                    # The bit string is ordered from right-to-left (q_n...q_1q_0), so reverse it
                    self.bob_bits = [int(b) for b in measured_string[::-1]]
                else:
                    # Fallback if no counts
                    self.bob_bits = np.random.randint(2, size=self.num_qubits).tolist()
            except Exception as e:
                print(f"Bob's measurement failed: {e}, using random bits")
                self.bob_bits = np.random.randint(2, size=self.num_qubits).tolist()
            
            print(f"  Bob's measured bits: {self.bob_bits}")
            self.final_circuit = qc_received
            
            # Classical communication & sifting
            print(f"\n--- Classical Communication ---")
            print(f"Connecting to Alice's classical channel...")
            
            # Notify Alice that Bob connected
            to_send.append(('alice', {
                'event': 'bob_connected', 
                'message': f'Bob connected for key sifting.'
            }))
            
            print(f"Sent Bob's bases to Alice.")
            to_send.append(('alice', {
                'event': 'bob_bases_sent', 
                'bases': self.bob_bases.tolist()
            }))
            
            # Alice compares bases and creates sifted key
            matching_indices = [i for i, (a, b) in enumerate(zip(self.alice_bases, self.bob_bases)) if a == b]
            print(f"Received matching indices: {matching_indices}")
            sifted_alice = [int(self.alice_bits[i]) for i in matching_indices]
            print(f"Received Alice's key for comparison: {sifted_alice}")
            
            # Notify Alice about the matching process
            to_send.append(('alice', {
                'event': 'bases_compared', 
                'matching_indices': matching_indices, 
                'sifted_key': sifted_alice
            }))
            
            # Bob creates his sifted key
            sifted_bob = [int(self.bob_bits[i]) for i in matching_indices]
            print(f"Bob's sifted key: {sifted_bob}")
            
            # Calculate QBER
            if sifted_bob:
                errors = sum(a != b for a, b in zip(sifted_alice, sifted_bob))
                qber = errors / len(sifted_bob)
            else:
                qber = 0.0
                
            eve_detected = qber > 0.1
            shared_key = ''.join(map(str, sifted_bob))
            
            print(f"\n--- BOB'S FINAL RESULT ---")
            print(f"Quantum Bit Error Rate (QBER): {qber:.2%}")
            if eve_detected:
                print(f"🚨 Eavesdropper DETECTED! Key is discarded.")
            else:
                print(f"✅ No eavesdropper detected. Key is likely secure.")
                print(f"Final shared key: {shared_key}")
                
            print(f"\n--- ALICE'S FINAL RESULT ---")
            print(f"✅ Secure key established. QBER = {qber:.2%}.")
            print(f"Final shared key: {shared_key}")
            
            # Send results to both Alice and Bob
            result_payload = {
                'event': 'protocol_complete',
                'qber': qber,
                'sifted_key': sifted_bob,
                'shared_key': shared_key,
                'eve_detected': eve_detected,
                'matching_indices': matching_indices,
                'alice_bits': self.alice_bits.tolist(),
                'bob_bits': self.bob_bits,
                'final_circuit_image': circuit_png_data_uri(self.final_circuit),
                'stats_plot': bb84_stats_data_uri(qber, len(sifted_bob))
            }
            to_send.append(('alice', result_payload))
            to_send.append(('bob', result_payload))
            self.state = 'done'
            self.completed = True
            
        return to_send
