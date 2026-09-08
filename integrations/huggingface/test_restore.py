import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from restore_checkpoint import restore


class RestoreTests(unittest.TestCase):
    def fixture(self, root):
        src = root / 'download'
        src.mkdir()
        part = src / 'checkpoint/part-0000'
        part.mkdir(parents=True)
        payload = b'unchanged tensor bytes'
        (part / '00000-weight.safetensors').write_bytes(payload)
        manifest = {'files': [{'original': 'nested/weight.safetensors',
            'published': 'checkpoint/part-0000/00000-weight.safetensors',
            'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}]}
        (src / 'TRANSPORT_MANIFEST.json').write_text(json.dumps(manifest))
        self.seal(src)
        return src, payload

    def seal(self, src):
        (src / 'UPLOAD_COMPLETE.json').write_text(json.dumps({'transport_manifest_sha256':
            hashlib.sha256((src / 'TRANSPORT_MANIFEST.json').read_bytes()).hexdigest()}))

    def test_exact_restore_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            src, payload = self.fixture(root)
            restore(src, root / 'restored', verify=True)
            self.assertEqual((root / 'restored/nested/weight.safetensors').read_bytes(), payload)
            with self.assertRaises(ValueError):
                restore(src, root / 'restored')

    def test_incomplete_and_corrupt_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            src, _ = self.fixture(root)
            (src / 'checkpoint/part-0000/00000-weight.safetensors').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                restore(src, root / 'restored', verify=True)
            self.assertFalse((root / 'restored').exists())

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            src, _ = self.fixture(root)
            p = src / 'TRANSPORT_MANIFEST.json'
            data = json.loads(p.read_text())
            data['files'][0]['original'] = '../escape'
            p.write_text(json.dumps(data))
            self.seal(src)
            with self.assertRaises(ValueError):
                restore(src, root / 'restored')


if __name__ == '__main__':
    unittest.main()
