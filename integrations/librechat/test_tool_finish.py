"""CPU contract tests against an explicitly supplied live-adapter source copy.

Run: python3 test_tool_finish.py /path/to/deepseek_reasoning_adapter.py
Only the normalizer functions/classes are extracted; no server is started.
"""
import ast
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys
from typing import Any
import unittest

source = Path(sys.argv.pop(1))
names = {'normalize_reasoning', 'closest_offered_tool', 'normalize_tool_arguments',
         'normalize_tool_calls', 'transform_json_bytes', 'ToolCallStreamNormalizer'}
tree = ast.parse(source.read_text())
body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
namespace = {'Any': Any, 'json': json, 'SequenceMatcher': SequenceMatcher}
exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), namespace)
call = {'index': 0, 'id': 'call-test', 'type': 'function',
        'function': {'name': 'echo_marker', 'arguments': '{"marker":"K275_TOOL_OK"}'}}


class Contract(unittest.TestCase):
    def test_nonstream_call_finish(self):
        result = namespace['normalize_tool_calls']({'choices': [{'message': {'tool_calls': [call]}, 'finish_reason': 'stop'}]}, {})
        self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')

    def test_length_is_not_success(self):
        result = namespace['normalize_tool_calls']({'choices': [{'message': {'tool_calls': [call]}, 'finish_reason': 'length'}]}, {})
        self.assertEqual(result['choices'][0]['finish_reason'], 'length')

    def test_normal_text_is_unchanged(self):
        result = namespace['normalize_tool_calls']({'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]}, {})
        self.assertEqual(result['choices'][0]['finish_reason'], 'stop')

    def events(self, normalizer, delta, finish=None):
        raw = b'data: ' + json.dumps({'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}).encode() + b'\n'
        return [json.loads(line[6:]) for line in normalizer.feed(raw) if line.startswith(b'data: {')]

    def test_buffered_calls_precede_terminal(self):
        normalizer = namespace['ToolCallStreamNormalizer']({})
        self.events(normalizer, {'tool_calls': [call]})
        events = self.events(normalizer, {}, 'stop')
        self.assertEqual(events[0]['choices'][0]['delta']['tool_calls'][0]['function'], call['function'])
        self.assertEqual(events[-1]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertFalse(normalizer.pending)

    def test_same_chunk_call_and_stop_retains_terminal(self):
        events = self.events(namespace['ToolCallStreamNormalizer']({}), {'tool_calls': [call]}, 'stop')
        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1]['choices'][0]['finish_reason'], 'tool_calls')

    def test_stream_plain_stop_is_unchanged(self):
        events = self.events(namespace['ToolCallStreamNormalizer']({}), {'content': 'OK'}, 'stop')
        self.assertEqual(events[-1]['choices'][0]['finish_reason'], 'stop')


if __name__ == '__main__':
    unittest.main()
