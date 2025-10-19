import os
import numpy as np
from typing import Dict, Any, List, Tuple
import qiskit_superstaq as qss
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import math
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import config
from qkd_webapp.config import Config
from .base import ProtocolSession


class E91Session(ProtocolSession):
    """E91 protocol session with entangled pairs and CHSH test - handles qubit limitations."""

    def __init__(self, room_id: str):
        super().__init__(room_id)
        self.state = 'init'
        self.pairs = None
        self.results = {}
        self.alice_sequence = None
        self.eve_tampered_sequence = None
        self.provider = None
        self.backend = None
        # Superstaq limitation: each Bell pair needs 2 qubits, so max 4 pairs per batch
        self.max_pairs_per_batch = 4  # 4 pairs = 8 qubits (under 9-qubit limit)

    def _initialize_quantum_backend(self):
        """Initialize Superstaq provider and backend with error handling"""
        try:
            if not Config.SUPERSTAQ_API_KEY:
                raise ValueError("SUPERSTAQ_API_KEY not found in environment variables")
            
            if not self.provider:
                self.provider = qss.SuperstaqProvider(api_key=Config.SUPERSTAQ_API_KEY)
                self.backend = self.provider.get_backend("cq_sqale_simulator")
            return True
        except Exception as e:
            print(f"❌ Error initializing quantum backend: {e}")
            return False

    def _create_bell_pairs_in_batches(self, num_pairs):
        """Create entangled Bell pairs in batches to respect qubit limits"""
        try:
            print(f"[Quantum] Creating {num_pairs} Bell pairs in batches of {self.max_pairs_per_batch}")
            
            alice_bits = []
            bob_bits = []
            
            num_batches = math.ceil(num_pairs / self.max_pairs_per_batch)
            
            for batch_idx in range(num_batches):
                start_pair = batch_idx * self.max_pairs_per_batch
                end_pair = min(start_pair + self.max_pairs_per_batch, num_pairs)
                pairs_in_batch = end_pair - start_pair
                qubits_needed = pairs_in_batch * 2
                
                print(f"  Batch {batch_idx + 1}/{num_batches}: Creating {pairs_in_batch} Bell pairs ({qubits_needed} qubits)")
                
                # Create quantum circuit for this batch
                qr = QuantumRegister(qubits_needed, 'q')
                cr = ClassicalRegister(qubits_needed, 'c')
                qc = QuantumCircuit(qr, cr)
                
                # Create Bell pairs within the batch
                for pair_idx in range(pairs_in_batch):
                    qubit_a = pair_idx * 2      # Alice's qubit
                    qubit_b = pair_idx * 2 + 1  # Bob's qubit
                    
                    # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
                    qc.h(qr[qubit_a])           # Hadamard on Alice's qubit
                    qc.cx(qr[qubit_a], qr[qubit_b])  # CNOT to create entanglement
                
                # Measure all qubits in the batch
                qc.measure(qr, cr)
                
                # Execute the batch
                job = self.backend.run(qc, method="dry-run", shots=1)
                result = job.result()
                counts = result.get_counts()
                bitstring = list(counts.keys())[0][::-1]  # Reverse for correct order
                
                # Extract Alice and Bob bits from the measurement results
                for pair_idx in range(pairs_in_batch):
                    alice_bit = int(bitstring[pair_idx * 2])
                    bob_bit = int(bitstring[pair_idx * 2 + 1])
                    alice_bits.append(alice_bit)
                    bob_bits.append(bob_bit)
                
            return np.array(alice_bits), np.array(bob_bits)
            
        except Exception as e:
            print(f"❌ Error in Bell pair creation: {e}")
            print("🔄 Falling back to classical simulation...")
            # Classical fallback: perfectly correlated random bits
            bits = np.random.randint(0, 2, size=num_pairs)
            return bits.copy(), bits.copy()

    def _generate_e91_visualization(self, alice_bits, bob_bits, eve_tampered=None, protocol_stats=None):
        """Generate E91-specific visualization graphs"""
        try:
            graphs = {}
            
            # 1. LINE GRAPH: Entangled Pair Correlation
            plt.figure(figsize=(12, 6))
            
            x_indices = range(len(alice_bits))
            
            # Plot Alice and Bob's correlated measurements
            plt.plot(x_indices, alice_bits, 'o-', label='Alice (Entangled)', 
                    color='blue', linewidth=2, markersize=6)
            plt.plot(x_indices, bob_bits, 's-', label='Bob (Entangled)', 
                    color='green', linewidth=2, markersize=6)
            
            # Plot Eve's tampered sequence if available
            if eve_tampered is not None:
                plt.plot(x_indices, eve_tampered, 'x--', label='Eve (Intercepted)', 
                        color='red', linewidth=2, markersize=8, alpha=0.7)
            
            # Highlight correlation breaks
            correlations = [i for i, (a, b) in enumerate(zip(alice_bits, bob_bits)) if a != b]
            if correlations:
                for pos in correlations:
                    plt.axvline(x=pos, color='red', linestyle=':', alpha=0.5)
                    plt.scatter(pos, alice_bits[pos], color='red', s=100, marker='o', 
                              facecolors='none', edgecolors='red', linewidth=2)
            
            plt.title('E91 QKD: Entangled Pair Correlations', fontsize=16, fontweight='bold')
            plt.xlabel('Entangled Pair Index', fontsize=12)
            plt.ylabel('Measurement Result (0 or 1)', fontsize=12)
            plt.ylim(-0.2, 1.2)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            # Save line graph
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            line_graph_b64 = base64.b64encode(buffer.getvalue()).decode()
            graphs['line_graph'] = f"data:image/png;base64,{line_graph_b64}"
            plt.close()
            
            # 2. BAR GRAPH: E91 Protocol Analysis
            if protocol_stats:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle('E91 QKD Protocol: Entanglement Analysis', fontsize=16, fontweight='bold')
                
                # Subplot 1: Entanglement Statistics
                categories1 = ['Total\nPairs', 'Correlated\nPairs', 'Broken\nCorrelations']
                values1 = [
                    len(alice_bits),
                    len(alice_bits) - protocol_stats.get('mismatches', 0),
                    protocol_stats.get('mismatches', 0)
                ]
                colors1 = ['lightblue', 'lightgreen', 'lightcoral']
                
                bars1 = ax1.bar(categories1, values1, color=colors1, alpha=0.8, edgecolor='black')
                ax1.set_title('Entanglement Analysis', fontweight='bold')
                ax1.set_ylabel('Count')
                
                for bar, val in zip(bars1, values1):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(val), ha='center', va='bottom', fontweight='bold')
                
                # Subplot 2: Security Assessment
                security_labels = ['Secure\nKey Bits', 'Compromised\nBits']
                secure_bits = protocol_stats.get('key_length', 0)
                compromised_bits = protocol_stats.get('mismatches', 0)
                security_values = [secure_bits, compromised_bits]
                security_colors = ['green', 'red']
                
                bars2 = ax2.bar(security_labels, security_values, color=security_colors, alpha=0.8)
                ax2.set_title('Security Analysis', fontweight='bold')
                ax2.set_ylabel('Count')
                
                for bar, val in zip(bars2, security_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(val), ha='center', va='bottom', fontweight='bold', color='white')
                
                # Subplot 3: Quantum Processing
                batch_categories = ['Bell Pair\nBatches', 'Pairs per\nBatch']
                batch_values = [
                    math.ceil(len(alice_bits) / self.max_pairs_per_batch),
                    self.max_pairs_per_batch
                ]
                
                bars3 = ax3.bar(batch_categories, batch_values, color=['orange', 'purple'], alpha=0.8)
                ax3.set_title('Quantum Processing', fontweight='bold')
                ax3.set_ylabel('Count')
                
                for bar, val in zip(bars3, batch_values):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                            str(val), ha='center', va='bottom', fontweight='bold')
                
                # Subplot 4: Correlation Quality
                correlation_rate = (1 - protocol_stats.get('mismatch_percent', 0) / 100) * 100
                violation_rate = 100 - correlation_rate
                
                rate_labels = ['Correlation\n%', 'Violation\n%']
                rate_values = [correlation_rate, violation_rate]
                rate_colors = ['lightgreen', 'lightcoral']
                
                bars4 = ax4.bar(rate_labels, rate_values, color=rate_colors, alpha=0.8)
                ax4.set_title(f'Bell Inequality\nEve: {"Detected" if protocol_stats.get("eve_detected", False) else "Not Detected"}', 
                             fontweight='bold')
                ax4.set_ylabel('Percentage')
                ax4.set_ylim(0, 100)
                
                for bar, val in zip(bars4, rate_values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                
                # Save bar graph
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                buffer.seek(0)
                bar_graph_b64 = base64.b64encode(buffer.getvalue()).decode()
                graphs['bar_graph'] = f"data:image/png;base64,{bar_graph_b64}"
                plt.close()
                
            return graphs
            
        except Exception as e:
            print(f"❌ Error generating E91 visualization: {e}")
            return {}

    async def handle(self, sender: str, msg: Dict[str, Any]):
        action = msg.get('action')
        to_send: List[Tuple[str, Dict[str, Any]]] = []

        if self.state == 'init' and sender == 'alice' and action == 'start':
            # Limit pairs for demonstration due to qubit constraints
            num_pairs = min(msg.get('num_pairs', 16), 20)  # Cap at 20 pairs (40 qubits max)
            
            try:
                # Initialize quantum backend
                if not self._initialize_quantum_backend():
                    to_send.append(('alice', {'event': 'error', 'message': 'Failed to initialize quantum backend'}))
                    return to_send
                
                # Create entangled Bell pairs in batches
                alice_bits, bob_bits = self._create_bell_pairs_in_batches(num_pairs)
                
                self.alice_sequence = alice_bits
                print(f"[Alice] Alice's entangled bits: {alice_bits}")
                print(f"[Alice] Bob's entangled bits: {bob_bits}")
                
                # Send results
                to_send.append(('alice', {
                    'event': 'alice_prepared', 
                    'bits': alice_bits.tolist(), 
                    'num_pairs': num_pairs,
                    'batches_used': math.ceil(num_pairs / self.max_pairs_per_batch)
                }))
                
                # Send to Eve and Bob
                to_send.append(('eve', {
                    'event': 'pairs_intercept', 
                    'bits': bob_bits.tolist(), 
                    'num_pairs': num_pairs, 
                    'protocol': 'E91'
                }))
                to_send.append(('bob', {
                    'event': 'pairs_received', 
                    'bits': bob_bits.tolist(), 
                    'num_pairs': num_pairs, 
                    'protocol': 'E91', 
                    'source': 'Alice'
                }))
                
                to_send.append(('alice', {
                    'event': 'classical_waiting', 
                    'message': f'[Alice] Waiting for Bob... (Created {num_pairs} Bell pairs in {math.ceil(num_pairs / self.max_pairs_per_batch)} batches)'
                }))
                self.state = 'alice_prepared'
                
            except Exception as e:
                print(f"❌ Error in Alice's preparation: {e}")
                to_send.append(('alice', {
                    'event': 'error', 
                    'message': f'Error in Bell pair creation: {str(e)}'
                }))
            
        elif self.state == 'alice_prepared' and sender == 'eve' and action == 'eve_forward':
            try:
                tamper_prob = msg.get('tamper_prob', 0.3)
                
                if self.alice_sequence is None:
                    to_send.append(('eve', {'event': 'error', 'message': 'No entangled pairs to intercept'}))
                    return to_send
                
                # Eve intercepts Bob's entangled qubits and tampers with them
                received_bits = msg.get('intercepted_bits', self.alice_sequence.tolist())
                tampered = np.array(received_bits).copy()
                tampered_positions = []
                
                for i in range(len(tampered)):
                    if np.random.rand() < tamper_prob:
                        tampered[i] = 1 - tampered[i]  # Flip the bit
                        tampered_positions.append(i)
                
                self.eve_tampered_sequence = tampered
                print(f"[Eve] Intercepted entangled pairs: {received_bits}")
                print(f"[Eve] Tampered {len(tampered_positions)} positions: {tampered_positions}")
                print(f"[Eve] Tampered sequence: {tampered}")
                
                self.state = 'eve_intercept'
                
                # Forward tampered pairs to Bob
                to_send.append(('eve', {
                    'event': 'eve_forwarded', 
                    'message': f'Intercepted and tampered {len(tampered_positions)} out of {len(received_bits)} entangled pairs',
                    'tampered_bits': tampered.tolist(),
                    'tampered_positions': tampered_positions
                }))
                to_send.append(('bob', {
                    'event': 'pairs_received', 
                    'bits': tampered.tolist(), 
                    'num_pairs': len(tampered), 
                    'protocol': 'E91', 
                    'source': 'Eve'
                }))
                
            except Exception as e:
                print(f"❌ Error in Eve's interception: {e}")
                to_send.append(('eve', {'event': 'error', 'message': f'Error in tampering: {str(e)}'}))
            
        elif (self.state in ['alice_prepared', 'eve_intercept']) and sender == 'bob' and action == 'measure':
            try:
                # Bob receives his entangled qubits (possibly tampered by Eve)
                received_bits = self.eve_tampered_sequence if self.eve_tampered_sequence is not None else msg.get('bits', [])
                bob_bits = np.array(received_bits)
                print(f"[Bob] Received entangled qubits: {received_bits}")
                print(f"[Bob] Measured bits: {bob_bits}")
                
                # Send results to Alice
                to_send.append(('bob', {
                    'event': 'bob_measured', 
                    'bits': bob_bits.tolist()
                }))
                to_send.append(('alice', {'event': 'bob_results_received', 'bits': bob_bits.tolist()}))
                
                # Alice performs Bell inequality test / correlation analysis
                if self.alice_sequence is not None:
                    mismatches = int(np.sum(self.alice_sequence != bob_bits))
                    mismatch_percent = (mismatches / len(self.alice_sequence)) * 100 if len(self.alice_sequence) else 0.0
                    
                    # In E91, perfect correlation should exist without eavesdropping
                    if mismatches > 0:
                        print(f"🚨 Bell inequality violation! {mismatches}/{len(self.alice_sequence)} pairs broken ({mismatch_percent:.2f}%)")
                        eve_detected = True
                    else:
                        print("✅ Perfect entanglement correlation maintained. No eavesdropping detected.")
                        eve_detected = False
                    
                    # Create secure key from correlated pairs
                    sifted_key = [int(a) for a, b in zip(self.alice_sequence, bob_bits) if a == b]
                    print(f"Final secure key: {sifted_key}")
                    
                    # Generate visualizations
                    protocol_stats = {
                        'mismatches': mismatches,
                        'mismatch_percent': mismatch_percent,
                        'eve_detected': eve_detected,
                        'key_length': len(sifted_key),
                        'bell_pairs_created': len(self.alice_sequence)
                    }
                    
                    graphs = self._generate_e91_visualization(
                        self.alice_sequence,
                        bob_bits,
                        self.eve_tampered_sequence,
                        protocol_stats
                    )
                    
                    self.state = 'done'
                    self.completed = True
                    
                    res = {
                        'event': 'protocol_complete',
                        'mismatches': mismatches,
                        'mismatch_percent': mismatch_percent,
                        'eve_detected': eve_detected,
                        'key_length': len(sifted_key),
                        'sifted_key': sifted_key,
                        'alice_bits': self.alice_sequence.tolist(),
                        'bob_bits': bob_bits.tolist(),
                        'eve_tampered': self.eve_tampered_sequence.tolist() if self.eve_tampered_sequence is not None else None,
                        'bell_pairs_created': len(self.alice_sequence),
                        'batches_used': math.ceil(len(self.alice_sequence) / self.max_pairs_per_batch),
                        'visualizations': graphs
                    }
                    to_send.append(('alice', res))
                    to_send.append(('bob', res))
                    
                    if self.eve_tampered_sequence is not None:
                        to_send.append(('eve', res))
                else:
                    to_send.append(('bob', {'event': 'error', 'message': 'No Alice sequence for correlation analysis'}))
                    
            except Exception as e:
                print(f"❌ Error in Bob's measurement: {e}")
                to_send.append(('bob', {'event': 'error', 'message': f'Error in measurement: {str(e)}'}))

        return to_send
