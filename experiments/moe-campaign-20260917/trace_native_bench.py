"""Record native generated inputs before its benchmark timer, without changing them.

The endpoint sends SampleRequest.prompt directly. Strings are captured as exact
UTF-8 and token lists as exact JSON; no extra tokenization or request wrapper is
added to the timed path. Historical baseline inputs remain unrecorded.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_requests(requests):
    records = []
    for index, request in enumerate(requests):
        prompt = request.prompt
        if isinstance(prompt, str):
            encoding, data = 'utf-8', prompt.encode('utf-8')
        else:
            assert isinstance(prompt, list) and all(type(x) is int and x >= 0 for x in prompt)
            encoding, data = 'json-token-ids', json.dumps(prompt, separators=(',', ':')).encode()
        assert request.prompt_len == 3072 and request.expected_output_len == 1024
        records.append(dict(index=index, prompt=prompt, encoding=encoding,
                            prompt_sha256=hashlib.sha256(data).hexdigest(),
                            declared_input_len=request.prompt_len,
                            requested_output_len=request.expected_output_len))
    assert len(records) == 64
    return records


def main():
    output = Path(sys.argv[1])
    argv = sys.argv[2:]
    assert argv[:3] == ['vllm', 'bench', 'serve'] and not output.exists()
    sys.argv = argv
    from vllm.entrypoints.cli.main import main as native_main
    import vllm.benchmarks.serve as serve
    import vllm.benchmarks.datasets as datasets
    import vllm.benchmarks.lib.endpoint_request_func as endpoint
    original = serve.get_samples
    calls = 0

    def traced_samples(args, tokenizer):
        nonlocal calls
        assert calls == 0, 'Unexpected repeated native dataset generation'
        requests = original(args, tokenizer)
        calls += 1
        record = dict(seed=args.seed, command=argv, requests=record_requests(requests),
                      trace_timing='after native get_samples, before native benchmark function',
                      historical_baseline_inputs_recovered=False,
                      native_modules_sha256={m.__name__: digest(inspect.getfile(m))
                                             for m in (serve, datasets, endpoint)},
                      tokenizer_sha256=digest('/model/tokenizer.json'),
                      wrapper_sha256=digest(__file__))
        with output.open('x') as stream:
            json.dump(record, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        return requests

    serve.get_samples = traced_samples
    try:
        native_main()
        assert calls == 1 and output.is_file(), 'Native input trace did not run'
    finally:
        serve.get_samples = original


if __name__ == '__main__':
    main()
