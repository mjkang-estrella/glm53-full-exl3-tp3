import ast
from pathlib import Path
import unittest

tree = ast.parse(Path(__file__).with_name('upload_models.py').read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_hub_attributes')
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<attributes-test>', 'exec'), ns)
validate = ns['validate_hub_attributes']
BASE = b'*.safetensors filter=lfs diff=lfs merge=lfs -text\n'


class AttributesTests(unittest.TestCase):
    def test_original_and_known_json_rule(self):
        self.assertEqual(validate(BASE, set()), ['*.safetensors'])
        body = BASE + b'checkpoint/part-0000/index.json filter=lfs diff=lfs merge=lfs -text\n'
        self.assertEqual(len(validate(body, {'checkpoint/part-0000/index.json'})), 2)

    def test_unknown_path_or_wildcard_rejected(self):
        for name in ('unknown.json', '*.json', '**'):
            with self.assertRaises(RuntimeError):
                validate(BASE + f'{name} filter=lfs diff=lfs merge=lfs -text\n'.encode(), {'known.json'})

    def test_altered_filter_or_missing_base_rejected(self):
        for body in (BASE.replace(b'filter=lfs', b'filter=other'), b'', BASE + BASE):
            with self.assertRaises(RuntimeError):
                validate(body, set())


if __name__ == '__main__':
    unittest.main()
