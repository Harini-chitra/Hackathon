import os
import numpy as np
from typing import Dict, Any, List, Tuple
import qiskit_superstaq as qss
from qiskit import QuantumCircuit
import math
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web applications
import matplotlib.pyplot as plt

# Import config
from qkd_webapp.config import Config
from .base import ProtocolSession


class COWSession(ProtocolSession):
    """Simplified Coherent One-Way protocol: Alice sends pulse sequence; Bob measures."""

    def __init__(self, room_id: str):
        super().__init__(room_id)
        self.state = 'init'
        self.sequence = None
        self.circuit = None
        self.eve_tampered_sequence = None
        self.provider = None
        self.backend = None
        self.max_qubits_per_batch = 8

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

    def _process_sequence_in_batches(self, sequence, operation_type="encode"):
        """Process large sequences in batches to respect qubit limits"""
        try:
            sequence = np.array(sequence)
            total_bits = len(sequence)
            batch_size = self.max_qubits_per_batch
            
            print(f"[Quantum] Processing {total_bits} bits in batches of {batch_size}")
            
            num_batches = math.ceil(total_bits / batch_size)
            processed_sequence = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_bits)
                batch_seq = sequence[start_idx:end_idx]
                batch_size_actual = len(batch_seq)
                
                print(f"  Processing batch {batch_idx + 1}/{num_batches}: bits {start_idx}-{end_idx-1}")
                
                qc = QuantumCircuit(batch_size_actual, batch_size_actual)
                
                if operation_type == "encode":
                    for i, bit in enumerate(batch_seq):
                        if bit == 1:
                            qc.x(i)
                
                qc.measure(range(batch_size_actual), range(batch_size_actual))
                
                job = self.backend.run(qc, method="dry-run", shots=1024)
                result = job.result()
                counts = result.get_counts()
                measured_bitstring = list(counts.keys())[0]
                batch_result = np.array([int(b) for b in measured_bitstring[::-1]])
                
                processed_sequence.extend(batch_result)
            
            return np.array(processed_sequence)
            
        except Exception as e:
            print(f"❌ Error in batch processing: {e}")
            print("🔄 Falling back to classical simulation...")
            return sequence.copy()

    def _generate_visualization_graphs(self, alice_seq, bob_seq, eve_tampered=None, protocol_stats=None):
        """Generate line graph and bar graph visualizations based on CLI code pattern"""
        try:
            graphs = {}
            
            # 1. LINE GRAPH: Sequence Comparison (similar to CLI cow_alice.py visualization)
            plt.figure(figsize=(12, 6))
            
            x_indices = range(len(alice_seq))
            
            # Plot Alice's sequence
            plt.plot(x_indices, alice_seq, 'o-', label='Alice (Original)', 
                    color='blue', linewidth=2, markersize=6)
            
            # Plot Bob's sequence
            plt.plot(x_indices, bob_seq, 'x-', label='Bob (Measured)', 
                    color='green', linewidth=2, markersize=8)
            
            # Plot Eve's tampered sequence if available
            if eve_tampered is not None:
                plt.plot(x_indices, eve_tampered, 's--', label='Eve (Tampered)', 
                        color='red', linewidth=2, markersize=4, alpha=0.7)
            
            # Highlight mismatched positions
            mismatches = [i for i, (a, b) in enumerate(zip(alice_seq, bob_seq)) if a != b]
            if mismatches:
                for pos in mismatches:
                    plt.axvline(x=pos, color='red', linestyle=':', alpha=0.5)
                    plt.scatter(pos, alice_seq[pos], color='red', s=100, marker='o', 
                              facecolors='none', edgecolors='red', linewidth=2)
            
            plt.title('COW QKD: Pulse Sequence Comparison', fontsize=16, fontweight='bold')
            plt.xlabel('Pulse Index', fontsize=12)
            plt.ylabel('Bit Value (0 or 1)', fontsize=12)
            plt.ylim(-0.2, 1.2)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            # Save line graph to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            line_graph_b64 = base64.b64encode(buffer.getvalue()).decode()
            graphs['line_graph'] = f"data:image/png;base64,{line_graph_b64}"
            plt.close()
            
            # 2. BAR GRAPH: Protocol Statistics and Analysis
            if protocol_stats:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
                
                # Subplot 1: Sequence Length Analysis
                categories = ['Total Bits', 'Matching Bits', 'Mismatched Bits']
                values = [
                    len(alice_seq),
                    len(alice_seq) - protocol_stats.get('mismatches', 0),
                    protocol_stats.get('mismatches', 0)
                ]
                colors = ['skyblue', 'lightgreen', 'lightcoral']
                
                bars1 = ax1.bar(categories, values, color=colors)
                ax1.set_title('Bit Analysis', fontweight='bold')
                ax1.set_ylabel('Number of Bits')
                
                # Add value labels on bars
                for bar, val in zip(bars1, values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                            str(val), ha='center', fontweight='bold')
                
                # Subplot 2: Security Analysis
                security_labels = ['Secure Bits', 'Tampered Bits']
                secure_bits = protocol_stats.get('key_length', 0)
                tampered_bits = protocol_stats.get('mismatches', 0)
                security_values = [secure_bits, tampered_bits]
                security_colors = ['green', 'red']
                
                bars2 = ax2.bar(security_labels, security_values, color=security_colors)
                ax2.set_title('Security Analysis', fontweight='bold')
                ax2.set_ylabel('Number of Bits')
                
                for bar, val in zip(bars2, security_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            str(val), ha='center', fontweight='bold', color='white')
                
                # Subplot 3: Protocol Performance
                perf_categories = ['Quantum Batches', 'Max Qubits/Batch', 'Total Qubits']
                perf_values = [
                    protocol_stats.get('quantum_batches_used', 1),
                    self.max_qubits_per_batch,
                    len(alice_seq)
                ]
                
                bars3 = ax3.bar(perf_categories, perf_values, color=['orange', 'purple', 'cyan'])
                ax3.set_title('Quantum Processing', fontweight='bold')
                ax3.set_ylabel('Count')
                ax3.tick_params(axis='x', rotation=15)
                
                for bar, val in zip(bars3, perf_values):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                            str(val), ha='center', fontweight='bold')
                
                # Subplot 4: Success Rate
                success_rate = (1 - protocol_stats.get('mismatch_percent', 0) / 100) * 100
                eve_detection = 'Detected' if protocol_stats.get('eve_detected', False) else 'Not Detected'
                
                # Pie chart for success rate
                ax4.pie([success_rate, 100 - success_rate], 
                       labels=[f'Successful\n{success_rate:.1f}%', f'Failed\n{100-success_rate:.1f}%'],
                       colors=['lightgreen', 'lightcoral'],
                       autopct='%1.1f%%',
                       startangle=90)
                ax4.set_title(f'Protocol Success Rate\nEve: {eve_detection}', fontweight='bold')
                
                plt.suptitle('COW QKD Protocol: Comprehensive Analysis', 
                           fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout()
                
                # Save bar graph to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                bar_graph_b64 = base64.b64encode(buffer.getvalue()).decode()
                graphs['bar_graph'] = f"data:image/png;base64,{bar_graph_b64}"
                plt.close()
            
            return graphs
            
        except Exception as e:
            print(f"❌ Error generating visualization graphs: {e}")
            return {}

    async def handle(self, sender: str, msg: Dict[str, Any]):
        action = msg.get('action')
        to_send: List[Tuple[str, Dict[str, Any]]] = []

        if self.state == 'init' and sender == 'alice' and action == 'start':
            num_bits = min(msg.get('num_bits', 16), 32)
            
            try:
                if not self._initialize_quantum_backend():
                    to_send.append(('alice', {'event': 'error', 'message': 'Failed to initialize quantum backend'}))
                    return to_send
                
                raw_seq = np.random.randint(0, 2, size=num_bits)
                print(f"[Alice] Raw pulses: {raw_seq}")
                
                alice_seq = self._process_sequence_in_batches(raw_seq, "encode")
                
                self.sequence = alice_seq
                print(f"[Alice] Encoded sequence: {alice_seq}")
                
                to_send.append(('alice', {
                    'event': 'alice_prepared', 
                    'sequence': alice_seq.tolist(), 
                    'num_bits': num_bits,
                    'batches_used': math.ceil(num_bits / self.max_qubits_per_batch)
                }))
                
                to_send.append(('eve', {
                    'event': 'sequence_intercept', 
                    'sequence': alice_seq.tolist(), 
                    'num_bits': num_bits, 
                    'protocol': 'COW'
                }))
                to_send.append(('bob', {
                    'event': 'sequence_received', 
                    'sequence': alice_seq.tolist(), 
                    'num_bits': num_bits, 
                    'protocol': 'COW', 
                    'source': 'Alice'
                }))
                
                to_send.append(('alice', {
                    'event': 'classical_waiting', 
                    'message': f'[Alice] Waiting for Bob on classical channel... (Processed in {math.ceil(num_bits / self.max_qubits_per_batch)} quantum batches)'
                }))
                self.state = 'alice_prepared'
                
            except Exception as e:
                print(f"❌ Error in Alice's preparation: {e}")
                to_send.append(('alice', {
                    'event': 'error', 
                    'message': f'Error in quantum preparation: {str(e)}'
                }))
            
        elif self.state == 'alice_prepared' and sender == 'eve' and action == 'eve_forward':
            try:
                tamper_prob = msg.get('tamper_prob', 0.25)
                
                if self.sequence is None:
                    to_send.append(('eve', {'event': 'error', 'message': 'No sequence to tamper with'}))
                    return to_send
                
                tampered = self.sequence.copy()
                tampered_positions = []
                
                for i in range(len(tampered)):
                    if np.random.rand() < tamper_prob:
                        tampered[i] = 1 - tampered[i]
                        tampered_positions.append(i)
                
                self.eve_tampered_sequence = tampered
                print(f"[Eve] Intercepted sequence: {self.sequence}")
                print(f"[Eve] Tampered {len(tampered_positions)} positions: {tampered_positions}")
                print(f"[Eve] Tampered sequence: {tampered}")
                
                self.state = 'eve_intercept'
                
                to_send.append(('eve', {
                    'event': 'eve_forwarded', 
                    'message': f'Tampered {len(tampered_positions)} out of {len(self.sequence)} qubits ({tamper_prob*100:.1f}% probability)',
                    'tampered_sequence': tampered.tolist(),
                    'tampered_positions': tampered_positions
                }))
                to_send.append(('bob', {
                    'event': 'sequence_received', 
                    'sequence': tampered.tolist(), 
                    'num_bits': len(tampered), 
                    'protocol': 'COW', 
                    'source': 'Eve'
                }))
                
            except Exception as e:
                print(f"❌ Error in Eve's tampering: {e}")
                to_send.append(('eve', {'event': 'error', 'message': f'Error in tampering: {str(e)}'}))
            
        elif (self.state in ['alice_prepared', 'eve_intercept']) and sender == 'bob' and action == 'measure':
            try:
                if not self._initialize_quantum_backend():
                    to_send.append(('bob', {'event': 'error', 'message': 'Failed to initialize quantum backend'}))
                    return to_send
                
                received_seq = self.eve_tampered_sequence if self.eve_tampered_sequence is not None else self.sequence
                print(f"[Bob] Received: {received_seq}")
                
                bob_seq = self._process_sequence_in_batches(received_seq, "measure")
                
                print(f"[Bob] Measured: {bob_seq}")
                
                to_send.append(('bob', {
                    'event': 'bob_measured', 
                    'sequence': bob_seq.tolist(),
                    'batches_used': math.ceil(len(received_seq) / self.max_qubits_per_batch)
                }))
                to_send.append(('alice', {'event': 'bob_results_received', 'sequence': bob_seq.tolist()}))
                
                if self.sequence is not None:
                    mismatches = int(np.sum(self.sequence != bob_seq))
                    mismatch_percent = (mismatches / len(self.sequence)) * 100 if len(self.sequence) else 0.0
                    
                    if mismatches > 0:
                        print(f"🚨 Eve detected! {mismatches}/{len(self.sequence)} pulses tampered ({mismatch_percent:.2f}%)")
                        eve_detected = True
                    else:
                        print("✅ No interference detected. All pulses matched.")
                        eve_detected = False
                    
                    sifted_key = [int(a) for a, b in zip(self.sequence, bob_seq) if a == b]
                    print(f"Sifted key: {sifted_key}")
                    
                    # Generate visualization graphs
                    protocol_stats = {
                        'mismatches': mismatches,
                        'mismatch_percent': mismatch_percent,
                        'eve_detected': eve_detected,
                        'key_length': len(sifted_key),
                        'quantum_batches_used': math.ceil(len(self.sequence) / self.max_qubits_per_batch)
                    }
                    
                    graphs = self._generate_visualization_graphs(
                        self.sequence, 
                        bob_seq, 
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
                        'alice_sequence': self.sequence.tolist(),
                        'bob_sequence': bob_seq.tolist(),
                        'eve_tampered': self.eve_tampered_sequence.tolist() if self.eve_tampered_sequence is not None else None,
                        'quantum_batches_used': math.ceil(len(self.sequence) / self.max_qubits_per_batch),
                        'simulator_limit': f'{self.max_qubits_per_batch} qubits per batch',
                        # Add visualization graphs to results
                        'visualizations': graphs
                    }
                    to_send.append(('alice', res))
                    to_send.append(('bob', res))
                    
                    if self.eve_tampered_sequence is not None:
                        to_send.append(('eve', res))
                else:
                    to_send.append(('bob', {'event': 'error', 'message': 'No reference sequence for comparison'}))
                    
            except Exception as e:
                print(f"❌ Error in Bob's measurement: {e}")
                to_send.append(('bob', {'event': 'error', 'message': f'Error in measurement: {str(e)}'}))

        return to_send
