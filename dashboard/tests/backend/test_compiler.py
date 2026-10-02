import unittest
from unittest.mock import patch

from backend.compiler import compile_project
from backend.project_store import default_project


class CompilerTests(unittest.TestCase):
    def test_default_h2o_project_compiles(self):
        with patch('backend.program.model_circuit',return_value=({'valid':True,'qubits':3,'gates':[]},'test-checkpoint-hash')):
            result = compile_project(default_project())
        self.assertTrue(result["valid"])
        self.assertEqual(result['program']['inputs']['steps'],10)
        self.assertEqual(result['circuit']['qubits'],3)
        self.assertEqual(result['program']['source'],default_project().files['main.py'])

    def test_python_syntax_error_is_reported(self):
        project = default_project()
        project.files["main.py"] = "def broken(:\n"
        result = compile_project(project)
        self.assertFalse(result["valid"])
        self.assertTrue(any(item["path"] == "main.py" for item in result["diagnostics"]))


if __name__ == "__main__":
    unittest.main()
