from qiskit import QuantumCircuit
from typing import List

def build_bb84_circuit(bits: List[int], bases: List[int]) -> QuantumCircuit:
    n = len(bits)
    qc = QuantumCircuit(n, n)
    for i, (bit, basis) in enumerate(zip(bits, bases)):
        if bit == 1:
            qc.x(i)
        if basis == 1:
            qc.h(i)
    return qc

from qiskit.visualization import circuit_drawer
from io import BytesIO


def circuit_to_png_bytes(qc: QuantumCircuit) -> bytes:
    """Render a circuit to PNG image bytes."""
    fig = circuit_drawer(qc, output="mpl")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


import matplotlib.pyplot as plt
import base64


def circuit_png_data_uri(qc: QuantumCircuit) -> str:
    import base64, io
    png_bytes = circuit_to_png_bytes(qc)
    b64 = base64.b64encode(png_bytes).decode()
    return f"data:image/png;base64,{b64}"


def bb84_stats_data_uri(qber: float, key_len: int) -> str:
    """Generate the same bar chart as CLI and return data URI."""
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(["QBER (%)", "Key Len"], [qber*100, key_len], color=["red","blue"])
    ax.set_ylabel("Value")
    ax.set_title("BB84 Protocol Statistics")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def build_cow_circuit(sequence):
    n = len(sequence)
    qc = QuantumCircuit(n, n)
    for i, bit in enumerate(sequence):
        if bit == 1:
            qc.x(i)
    qc.measure(range(n), range(n))
    return qc


def build_e91_pairs(num_pairs: int):
    # For simplicity just return num_pairs placeholder values
    return list(range(num_pairs))


def measure_bb84(qc: QuantumCircuit, bases_bob):
    qc2 = qc.copy()
    for i, basis in enumerate(bases_bob):
        if basis == 1:
            qc2.h(i)
    qc2.measure_all()
    return qc2
