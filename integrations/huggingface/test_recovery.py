import ast
import json
from pathlib import Path
import tempfile
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import unittest
from unittest.mock import patch, Mock
import supervise_upload as supervisor

tree = ast.parse(Path(__file__).with_name('upload_models.py').read_text())
fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'bounded_batches')
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<batch-test>', 'exec'), ns)
delay_fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'rate_limit_delay')
ns.update(re=re, time=time, parsedate_to_datetime=parsedate_to_datetime, timezone=timezone)
exec(compile(ast.Module(body=[delay_fn], type_ignores=[]), '<rate-test>', 'exec'), ns)


class Recovery(unittest.TestCase):
    def test_rate_reset_headers_and_fallback(self):
        delay = ns['rate_limit_delay']
        self.assertEqual(delay({'Retry-After': '3600'}, 0), 3605)
        self.assertEqual(delay({'RateLimit': '"api";r=0;t=900'}, 0), 905)
        self.assertEqual(delay({}, 0), 600)
        self.assertEqual(delay({}, 10), 3600)
        self.assertEqual(delay({'Retry-After': 'Thu, 01 Jan 1970 01:00:00 GMT'}, 0, now=0), 3605)

    def test_limits_and_single_large_stream(self):
        rows = [{'bytes': 10, 'id': i} for i in range(20)] + [{'bytes': 1000, 'id': 20}]
        batches = list(ns['bounded_batches'](rows, max_files=8, max_bytes=45))
        self.assertEqual([f for b in batches for f in b], rows)
        for b in batches:
            self.assertLessEqual(len(b), 8)
            self.assertTrue(sum(f['bytes'] for f in b) <= 45 or len(b) == 1)

    def test_file_limit(self):
        batches = list(ns['bounded_batches']([{'bytes': 1}] * 20))
        self.assertEqual([len(b) for b in batches], [8, 8, 4])

    def test_external_oom_updates_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'STATUS.json').write_text(json.dumps({'phase': 'uploading', 'committed_bytes': 123}))
            child = Mock()
            child.poll.return_value = -9
            with patch.object(supervisor, 'ROOT', root), patch.object(supervisor, 'scope', return_value={'Result': 'oom-kill'}), patch.object(supervisor.subprocess, 'Popen', return_value=child):
                self.assertEqual(supervisor.main(), -9)
            result = json.loads((root / 'STATUS.json').read_text())
            self.assertEqual(result['phase'], 'failed')
            self.assertEqual(result['scope_result'], 'oom-kill')
            self.assertEqual(result['committed_bytes'], 123)


if __name__ == '__main__':
    unittest.main()
