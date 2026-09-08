"""Run bounded sequential generation checks against an explicitly chosen adapter."""
import argparse
import json
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--base', required=True)
p.add_argument('--full', action='store_true')
a = p.parse_args()
model = 'GLM-5.3-K3-TP3-CANDIDATE'


def post(payload):
    request = urllib.request.Request(a.base + '/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=180)


base = dict(model=model, temperature=0, max_tokens=512,
            chat_template_kwargs={'enable_thinking': True})
payload = dict(base, messages=[{'role': 'user', 'content': 'Reply exactly K275_ROUTE_OK with no explanation.'}])
with post(payload) as r:
    result = json.load(r)
choice = result['choices'][0]
assert choice['message']['content'].strip() == 'K275_ROUTE_OK', result
assert choice['finish_reason'] == 'stop', result
print(json.dumps({'check': 'chat', 'passed': True, 'model': result['model'], 'usage': result.get('usage')}), flush=True)
if a.full:
    payload['stream'] = True
    text, finish, done = '', None, False
    with post(payload) as r:
        for line in r:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            for c in event.get('choices', []):
                text += c.get('delta', {}).get('content') or ''
                finish = c.get('finish_reason') or finish
    assert done and finish == 'stop' and text.strip() == 'K275_ROUTE_OK', (text, finish, done)
    print(json.dumps({'check': 'streaming', 'passed': True}), flush=True)
    tool = {'type': 'function', 'function': {'name': 'echo_marker', 'description': 'Return the marker',
        'parameters': {'type': 'object', 'properties': {'marker': {'type': 'string'}}, 'required': ['marker']}}}
    payload = dict(base, messages=[{'role': 'user', 'content': 'Call echo_marker with marker K275_TOOL_OK.'}],
                   tools=[tool], tool_choice={'type': 'function', 'function': {'name': 'echo_marker'}})
    with post(payload) as r:
        result = json.load(r)
    choice = result['choices'][0]
    call = choice['message']['tool_calls'][0]['function']
    assert call['name'] == 'echo_marker' and json.loads(call['arguments']) == {'marker': 'K275_TOOL_OK'}, result
    assert choice['finish_reason'] == 'tool_calls', result
    print(json.dumps({'check': 'tool_call', 'passed': True}), flush=True)
    payload['stream'] = True
    streamed_calls, terminal, done = [], None, False
    with post(payload) as r:
        for line in r:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            for c in event.get('choices', []):
                streamed_calls.extend(c.get('delta', {}).get('tool_calls') or [])
                terminal = c.get('finish_reason') or terminal
    assert done and terminal == 'tool_calls' and streamed_calls, (terminal, streamed_calls)
    fn = streamed_calls[0]['function']
    assert fn['name'] == 'echo_marker' and json.loads(fn['arguments']) == {'marker': 'K275_TOOL_OK'}, streamed_calls
    print(json.dumps({'check': 'streaming_tool_call', 'passed': True}), flush=True)
