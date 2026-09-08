"""CPU tests: resident telemetry must not inspect GPU tensors."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class Tests(unittest.TestCase):
    def test_resident_observer_is_noop(self):
        path=Path(__file__).resolve().parents[1]/'runtime/lazy_k3_patch.py'
        tree=ast.parse(path.read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='LazyExpertStore')
        fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='observe_ids')
        fn.args.args[1].annotation=None
        ns={'_execution_mode':lambda:'resident_uva'}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'observer','exec'),ns)
        ns['observe_ids'](SimpleNamespace(policy='lfu'),object())
        ns['_execution_mode']=lambda:'resident_fused'
        ns['observe_ids'](SimpleNamespace(policy='lfu'),object())
        ns['_execution_mode']=lambda:'chunked_fused'
        with self.assertRaises(AttributeError):
            ns['observe_ids'](SimpleNamespace(policy='lfu'),object())


if __name__=='__main__':unittest.main()
