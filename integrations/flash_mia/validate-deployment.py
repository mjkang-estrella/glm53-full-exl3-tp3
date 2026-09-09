#!/usr/bin/env python3
"""Zima CPU client: bounded checks of the Flash service, no generated code execution."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

MODEL = 'GLM-5.3-Flash-EXL3'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='http://192.168.0.238:8888')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--long-tokens', type=int, default=0)
    args = p.parse_args()
    result = {'model': MODEL, 'base': args.base, 'checks': {}, 'passed': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.out.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(args.out)
    def post(path, body, timeout=900):
        req = urllib.request.Request(args.base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            value = json.load(r)
        return value, time.monotonic() - start
    def chat(text, thinking=False, **extra):
        return post('/v1/chat/completions', dict(model=MODEL, temperature=0,
            messages=[{'role': 'user', 'content': text}], max_tokens=512 if thinking else 128,
            chat_template_kwargs={'enable_thinking': thinking}, **extra))
    for name, text, expected, thinking in [
        ('chat', 'Reply exactly FLASH_E3_OK without explanation.', 'FLASH_E3_OK', False),
        ('reasoning', 'Calculate 17 times 19. Final answer must be only the integer.', '323', True),
        ('korean', '다른 설명 없이 정확히 다음 글자만 답하세요: 한국어정상', '한국어정상', False)]:
        value, elapsed = chat(text, thinking)
        c = value['choices'][0]
        result['checks'][name] = {'passed': c['message']['content'].strip() == expected and c['finish_reason'] == 'stop',
            'content': c['message']['content'], 'finish_reason': c['finish_reason'], 'seconds': elapsed, 'usage': value.get('usage')}
        save()
    tool = {'type': 'function', 'function': {'name': 'report_value', 'description': 'Report an integer',
        'parameters': {'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value']}}}
    value, elapsed = chat('Call report_value with value 7.', tools=[tool], tool_choice='auto')
    c = value['choices'][0]
    calls = c['message'].get('tool_calls') or []
    ok = bool(calls) and calls[0]['function']['name'] == 'report_value' and json.loads(calls[0]['function']['arguments']) == {'value': 7}
    result['checks']['tool'] = {'passed': ok, 'finish_reason': c['finish_reason'], 'calls': calls, 'seconds': elapsed}
    save()
    if not all(x['passed'] for x in result['checks'].values()):
        raise SystemExit('Short quality gate failed; inspect receipt before long-context test')
    if args.long_tokens:
        unit = 'archive cedar delta ember fjord granite harbor iris juniper kinetic lunar maple. '
        def count(text):
            value, _ = post('/tokenize', {'model': MODEL, 'prompt': text})
            return value['count'] if 'count' in value else len(value['tokens'])
        per = count(unit * 128) / 128
        repeats = int((args.long_tokens - 120) / per)
        text = unit * repeats
        pivot = len(text) * 2 // 3
        needle = 'FLASH-E3-CEDAR-92741'
        text = ('Read the document and return only its unique retrieval code.\n<document>\n' +
            text[:pivot] + '\nThe unique retrieval code is ' + needle + '.\n' + text[pivot:] + '\n</document>')
        result['long_request'] = {'phase': 'generating', 'requested_target': args.long_tokens,
            'raw_prompt_tokens': count(text), 'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}
        save()
        value, elapsed = chat(text)
        c = value['choices'][0]
        result['checks']['long'] = {'passed': c['message']['content'].strip() == needle and c['finish_reason'] == 'stop',
            'content': c['message']['content'], 'finish_reason': c['finish_reason'], 'usage': value.get('usage'), 'seconds': elapsed}
    result['passed'] = all(x['passed'] for x in result['checks'].values())
    save()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
