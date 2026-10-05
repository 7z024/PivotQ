print(f"量子比特: {circuit.num_qubits}")
print(f"未赋值参数: {len(circuit.parameters)}")
print(f"基础门总数: {circuit.size()}")
print("门计数: " + ", ".join(
    f"{name.upper()}={count}" for name, count in sorted(circuit.count_ops().items())
))
print(f"电路深度: {circuit.depth()}")
