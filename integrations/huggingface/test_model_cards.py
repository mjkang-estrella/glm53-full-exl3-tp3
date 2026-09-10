"""CPU-only regression: published releases are quantizations, not fine-tunes."""
import ast
from pathlib import Path
import unittest

tree = ast.parse(Path(__file__).with_name('upload_models.py').read_text())
nodes = [n for n in tree.body if
         (isinstance(n, ast.FunctionDef) and n.name == 'card') or
         (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SPECS' for t in n.targets))]
ns = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<model-cards>', 'exec'), ns)


class ModelCards(unittest.TestCase):
    def test_both_are_explicit_quantizations(self):
        for spec in ns['SPECS']:
            with self.subTest(repo=spec['repo']):
                text = ns['card'](spec)
                metadata = text.split('---', 2)[1]
                self.assertIn('base_model: zai-org/GLM-5.3-BF16', metadata)
                self.assertIn('base_model_relation: quantized', metadata)
                self.assertNotIn('base_model_relation: finetune', metadata)
                self.assertIn('no additional language-model fine-tuning was performed', text)
                self.assertNotIn('Upload status: incomplete', text)
                self.assertIn('restore_checkpoint.py', text)


if __name__ == '__main__':
    unittest.main()
