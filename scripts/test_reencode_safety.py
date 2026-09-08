"""CPU-only tests for hardlink-safe metadata and training-only selection."""
import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent


class Safety(unittest.TestCase):
    def test_hardlink_metadata_replacement(self):
        spec=importlib.util.spec_from_file_location('assembler',HERE/'assemble_k275.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)/'base'
            target=Path(folder)/'target'
            base.write_text('immutable\n')
            os.link(base,target)
            module.atomic_text(target,'candidate\n')
            self.assertEqual(base.read_text(),'immutable\n')
            self.assertEqual(target.read_text(),'candidate\n')
            self.assertNotEqual(base.stat().st_ino,target.stat().st_ino)

    def test_selection_ignores_evaluation_error_and_preserves_budget(self):
        tree=ast.parse((HERE/'reencode_k2_pilot.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize')
        ns={}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'selection','exec'),ns)
        rows=[]
        for expert,train,check in [(0,20,1),(64,2,100)]:
            rows.append(dict(expert=expert,k3=dict(train_error=1,eval_error=1),
                identity=dict(train_error=train,eval_error=check),hessian=dict(train_error=train,eval_error=check)))
        result=ns['summarize'](rows,list(range(64)))
        for mode in result.values():
            self.assertEqual(mode['pilot_k2'],[64])
            self.assertEqual(len(mode['full_layer_k2']),64)
            self.assertFalse(mode['local_gate_passed'])


if __name__=='__main__': unittest.main()
