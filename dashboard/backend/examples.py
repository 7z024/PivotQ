from .program import H2O_SOURCE, CIRCUIT_SOURCE
EXAMPLES = [
    {'id':'h2o-aimd','title':'H₂O AIMD','task_id':'h2o-hybrid-aimd','language':'python','description':'冻结模型的混合 AIMD','code':H2O_SOURCE},
    {'id':'circuit-ghz','title':'GHZ 量子电路','task_id':'quantum-circuit','language':'python','description':'按编辑器门序列执行','code':CIRCUIT_SOURCE},
]
def all_examples():
    return EXAMPLES
