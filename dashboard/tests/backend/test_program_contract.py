import unittest
from backend.program import source_request, H2O_SOURCE, CIRCUIT_SOURCE, seal_program
from backend.registry import validate_and_plan
from backend.circuit_parser import parse_quantum_circuit
from backend.circuit_runner import simulate, verify

class ProgramContractTests(unittest.TestCase):
    def test_source_overrides_stale_form_parameters(self):
        request=source_request({'task_id':'h2o-hybrid-aimd','source':H2O_SOURCE.replace('steps=10','steps=2'),'inputs':{'steps':999}})
        errors,plan=validate_and_plan(request)
        self.assertEqual(errors,[])
        self.assertEqual(plan.normalized_inputs['steps'],2)

    def test_fake_h2o_circuit_is_rejected(self):
        with self.assertRaises(ValueError):
            source_request({'task_id':'h2o-hybrid-aimd','source':CIRCUIT_SOURCE})

    def test_unsupported_python_cannot_be_silently_ignored(self):
        for source in [CIRCUIT_SOURCE.replace('circuit.h(0)','if False:\n    circuit.h(0)'),
                       CIRCUIT_SOURCE+'print(1)\n', CIRCUIT_SOURCE.replace('h(0)','h(9)'),
                       CIRCUIT_SOURCE.replace('cx(0, 1)','cx(0, 0)'),
                       CIRCUIT_SOURCE.replace('h(0)','ry(float("nan"), 0)')]:
            self.assertFalse(parse_quantum_circuit(source)['valid'])

    def test_snapshot_tampering_rejected(self):
        req=source_request({'task_id':'quantum-circuit','source':CIRCUIT_SOURCE})
        _,plan=validate_and_plan(req)
        program=seal_program(req,plan)
        verify(program)
        program['circuit']['gates'][0]['name']='X'
        with self.assertRaises(ValueError): verify(program)

    def test_gate_semantics_against_independent_qiskit_oracle(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        source='circuit = QuantumCircuit(3)\n'
        qc=QuantumCircuit(3)
        for name,args in [('h',[0]),('x',[1]),('y',[2]),('z',[0]),('rx',[.37,0]),('ry',[-.71,1]),('rz',[1.24,2]),('cx',[0,2]),('cz',[2,1])]:
            source+=f'circuit.{name}({", ".join(map(str,args))})\n'
            getattr(qc,name)(*args)
        source+='circuit.measure([0,1,2])\n'
        circuit=parse_quantum_circuit(source)
        actual,counts=simulate(circuit,1000,42,device='cpu')
        expected=Statevector.from_instruction(qc).probabilities_dict()
        for bitstring,p in expected.items(): self.assertAlmostEqual(actual.get(bitstring[::-1],0),p,places=11)
        self.assertEqual(sum(counts.values()),1000)

    def test_changed_gate_changes_measured_outcome(self):
        for gate,expected in [('', '000'),('circuit.x(0)\n','100')]:
            circuit=parse_quantum_circuit('circuit = QuantumCircuit(3)\n'+gate+'circuit.measure([0,1,2])')
            p,c=simulate(circuit,100,2,device='cpu')
            self.assertEqual(p,{expected:1.0})
            self.assertEqual(c,{expected:100})
