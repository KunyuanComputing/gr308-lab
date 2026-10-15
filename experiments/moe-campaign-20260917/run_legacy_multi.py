"""Sequential old-vLLM model coverage, preserving MTP identity and GPU isolation."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
IMAGE = 'sha256:a153d4c8890880157e6860908e5f3db75d78d471589222fbdec193a0ab52c5dc'
CLIENT = ROOT.parent/'gr308-qwen36-graph-mtp-20260911/run.py'
QUALITY = ROOT.parent/'gr308-tp2-sweep-20260911/run_sweep.py'
MODELS = ['Qwen3.6-35B-A3B', 'Qwen3.8-27B', 'Qwen3.6-27B',
          'Qwen3-32B', 'Qwen3-VL-30B-A3B-Instruct', 'Qwen3-VL-32B-Instruct',
          'DeepSeek-R1-Distill-Qwen-7B', 'Qwen3.8-27B-GPTQ-Int4-V100']
PORT = 18182
DEVICES = [4, 5]
CASES = [(128,64,1),(128,64,4),(128,64,16),(1024,128,4)]
FORMAL_CASES = [(3072,1024,16)]


def command(args, timeout=40, check=True):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f'{args[:3]} rc={p.returncode}: {p.stderr[-3000:]}')
    return p


def save(root, name, value):
    (root/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def emit(**value):
    print(json.dumps(dict(time=time.time(), **value), ensure_ascii=False), flush=True)


def smi():
    raw = command(['sudo','-n','fantgpu-smi'], timeout=15).stdout
    mem = [int(x) for x in re.findall(r'(\d+)MiB / 57344MiB', raw)]
    temp = [int(x) for x in re.findall(r'\b(\d+)C\s', raw)]
    assert len(mem) == len(temp) == 8, 'Unexpected GPU inventory'
    return dict(time=time.time(), memories_mib=mem, temperatures_c=temp, raw=raw)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unexpected_containers(running, owned_id):
    # An empty list means the model exited, not that another workload appeared.
    return sorted(set(running)-{owned_id})


def formal_attention_kv_gib(config, tp=2, context=4096, sequences=16):
    text=config.get('text_config',config)
    heads=max(1,(text['num_key_value_heads']+tp-1)//tp)
    head_dim=text.get('head_dim',text['hidden_size']//text['num_attention_heads'])
    per_token=text['num_hidden_layers']*heads*head_dim*2*2
    target=per_token*context*sequences+512*1024**2
    return max(4,(target+1024**3-1)//1024**3)


def plan_for(model, variant):
    hybrid = model['model_type'] in ('qwen3_5','qwen3_5_moe')
    mtp = 3 if hybrid and model.get('mtp_layers') and variant != 'baseline' else 0
    return dict(mtp=mtp, hybrid=hybrid, kv_gib=8 if hybrid else 4,
                block_size=1088 if model['name']=='Qwen3.6-35B-A3B' else (800 if hybrid else None),
                dtype='float16' if 'GPTQ' in model['name'] else 'bfloat16',
                vision=model['model_type'] in ('qwen3_vl','qwen3_vl_moe'),
                reasoning=model['name'].startswith('DeepSeek-R1'),
                method='mtp3' if mtp else 'none',
                mtp_applicable=bool(hybrid and model.get('mtp_layers')))


def quality(client, helper, out, cfg):
    rows = helper.semantics(client, out, vision=cfg['vision'], reasoning=cfg['reasoning'])
    if not all(r['passed'] for r in rows) or cfg['reasoning']:
        return rows
    records = '\n'.join(f'Record {i:04d}: item code P-{i*7+3:05d}; quantity {i%31+1}.' for i in range(80))
    prompt = records+'\nTARGET_ORDER=ZX-8492\nReply with only the value of TARGET_ORDER.'
    body = dict(model=client.MODEL,messages=[dict(role='user',content=prompt)],temperature=0,
                seed=2026,max_tokens=128,chat_template_kwargs={'enable_thinking':False})
    response = client.api('/v1/chat/completions',body,180)
    choice = response['choices'][0]
    text = choice['message'].get('content') or ''
    passed = helper.answer_matches(text,'ZX-8492',choice['finish_reason'])
    rows.append(dict(case='long-extract',request=body,response=response,expected='ZX-8492',
                     final=text,passed=passed))
    save(out,'correctness.json',rows)
    emit(stage='correctness',model=client.MODEL,case='long-extract',passed=passed,answer=text)
    return rows


def performance(client, out, cases):
    source = ('Continue this inventory report in the same format.\n' +
              '\n'.join(f'Record {i:04d}: item P-{i*7+3:05d}, quantity {i%31+1}, '
                        f'warehouse {chr(65+i%8)}, status checked.' for i in range(400)))
    tokens = client.api('/tokenize',dict(model=client.MODEL,prompt=source,add_special_tokens=False))['tokens']
    save(out,'prompt-tokens.json',tokens)
    save(out,'workload.json',dict(kind='synthetic_exact_token_throughput_not_quality_eval',
                                source=source,ignore_eos=True,seed=2026,cases=cases))
    summaries=[]
    for inp,length,concurrency in cases:
        label=f'i{inp}-o{length}-c{concurrency}'
        prompt=tokens[:inp]
        assert len(prompt)==inp
        n=max(2,concurrency)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            warm=list(pool.map(lambda i:client.stream(label+'-warmup',i,prompt,8),range(concurrency)))
            assert all(r['ok'] for r in warm), label+': warmup contract failed'
            (out/f'{label}-metrics-before.prom').write_text(client.api('/metrics'))
            begin=time.monotonic()
            emit(stage='benchmark_started',model=client.MODEL,case=label,requests=n)
            rows=list(pool.map(lambda i:client.stream(label,i,prompt,length),range(n)))
            elapsed=time.monotonic()-begin
        (out/f'{label}-metrics-after.prom').write_text(client.api('/metrics'))
        good=[r for r in rows if r['ok']]
        row=dict(case=label,input_tokens=inp,output_tokens=length,concurrency=concurrency,requests=n,
                 passed=len(good),qualified=len(good)==n,wall_s=elapsed,
                 raw_completed_output_tokens_per_s=sum(r['usage']['completion_tokens'] for r in good)/elapsed)
        for metric in ('ttft_ms','tpot_ms','e2e_ms'):
            values=[r[metric] for r in good if r[metric] is not None]
            if values:
                row[metric]=dict(mean=sum(values)/len(values),p50=client.percentile(values,.5),p95=client.percentile(values,.95))
        summaries.append(row)
        save(out,'performance.json',summaries)
        emit(stage='performance',model=client.MODEL,**row)
        assert len(good)==n, label+': request failed, no throughput promotion'
    return summaries


def run_model(model, variant, attempt, cases, suite='coverage'):
    cfg=plan_for(model,variant)
    slug=re.sub('[^a-z0-9]+','-',model['name'].lower()).strip('-')
    suffix='' if suite=='coverage' else '-'+suite
    name=f'gr308-legacy-multi-20260911-{slug}-{cfg["method"]}{suffix}-r{attempt}'
    out=ROOT/(slug+'-'+cfg['method']+suffix+f'-r{attempt}')
    out.mkdir(exist_ok=False)
    (out/'requests').mkdir()
    result=dict(status='failed',model=model['name'],suite=suite,started_epoch_s=time.time(),container=name,
                configuration=cfg,logical_functions=DEVICES,physical_cards=sorted({x//2 for x in DEVICES}))
    owned=False
    stop=threading.Event()
    failures=[]
    watcher=None
    start=time.monotonic()
    try:
        assert not command(['docker','ps','-q']).stdout.strip(), 'Running container exists'
        assert command(['docker','inspect',name],check=False).returncode!=0, 'Container already exists'
        before=smi();save(out,'hardware-before.json',before)
        assert before['memories_mib']==[540]*8, 'Devices not fully idle'
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind(('127.0.0.1',PORT))
        root=Path(model['directory'])
        for file,key in [('config.json','config_sha256'),('model.safetensors.index.json','index_sha256'),('tokenizer.json','tokenizer_sha256')]:
            assert hashlib.sha256((root/file).read_bytes()).hexdigest()==model[key], 'Model metadata changed'
        inventory=load(ROOT/'inventory.py','current_inventory')
        current=inventory.inspect_model(root)
        assert current['weights_header_qualified'], 'Weight header admission failed'
        if suite!='coverage' and not cfg['hybrid']:
            config=json.loads((root/'config.json').read_text())
            cfg['kv_gib']=formal_attention_kv_gib(config)
            cfg['kv_sizing_note']='16 full-context BF16/FP16 attention sequences plus 512 MiB, rounded up; at least 4 GiB'
        save(out,'model-manifest.json',current)
        for source,destination in [(CLIENT,'client-frozen.py'),(QUALITY,'quality-frozen.py'),
                                   (Path(__file__),'runner.py')]:
            shutil.copyfile(source,out/destination)
        client=load(out/'client-frozen.py','client_'+slug)
        client.ROOT=out;client.MODEL=slug;client.PORT=PORT
        helper=load(out/'quality-frozen.py','quality_'+slug)
        runtime=Path('/etc/fant/fantHipSettings.cfg')
        shutil.copyfile(runtime,out/'fantHipSettings.cfg')
        graph=dict(mode='NONE',cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=[4,8,16,32,64])
        serve=['vllm','serve','/model','--host','127.0.0.1','--port',str(PORT),'--served-model-name',slug,
               '--tensor-parallel-size','2','--dtype',cfg['dtype'],'--gpu-memory-utilization','0.9',
               '--max-model-len','4096','--kv-cache-memory-bytes',str(cfg['kv_gib']*1024**3),
               '--max-num-seqs','16','--max-num-batched-tokens','2048','--no-enable-prefix-caching',
               '--no-async-scheduling','--disable-custom-all-reduce','--cudagraph-metrics',
               '--trust-remote-code','--no-enable-log-requests','--compilation-config',json.dumps(graph)]
        if cfg['hybrid']:
            serve+=['--block-size',str(cfg['block_size']),'--language-model-only']
        if cfg['vision']:
            serve+=['--limit-mm-per-prompt','{"image":1,"video":0}',
                    '--mm-processor-kwargs','{"max_pixels":262144}']
        if cfg['mtp']:
            serve+=['--speculative-config',json.dumps(dict(method='mtp',num_speculative_tokens=cfg['mtp']))]
        launch=['docker','run','-d','--name',name,'--restart','no','--network','host','--ipc','private',
                '--shm-size','8g','--privileged','--security-opt','label=disable','--cpuset-cpus','56-63',
                '--memory','256g','--log-opt','max-size=40m','--log-opt','max-file=2',
                '-e','CUDA_VISIBLE_DEVICES='+','.join(map(str,DEVICES)),
                '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','OMP_NUM_THREADS=2',
                '-e','XDG_CACHE_HOME=/cache/xdg','-e','TORCH_EXTENSIONS_DIR=/cache/torch_extensions']
        for path in ('/usr/lib/x86_64-linux-gnu/fantgpu-fh3','/usr/local/fant/include','/usr/local/fant/lib/llvm'):
            launch+=['-v',path+':'+path+':ro']
        cache='/data/infra/benchmarks/.grserve-qwen36-work/runtime-cache'
        launch+=['-v',cache+':/cache','-v',str(root)+':/model:ro',
                 '-v',str(out/'fantHipSettings.cfg')+':/etc/fant/fantHipSettings.cfg:ro',
                 IMAGE,'timeout','--signal=TERM','--kill-after=20','1240',*serve]
        hashes={file:hashlib.sha256((out/file).read_bytes()).hexdigest()
                for file in ('runner.py','client-frozen.py','quality-frozen.py','fantHipSettings.cfg')}
        save(out,'launch.json',dict(docker_args=launch,vllm_args=serve,mtp=cfg['mtp'],graph=graph,
                                  kv_bytes_per_worker=cfg['kv_gib']*1024**3,context=4096,
                                  logical_functions=DEVICES,physical_cards=sorted({x//2 for x in DEVICES}),
                                  runner_sha256=hashes['runner.py'],source_hashes=hashes,
                                  cpu_offload=False,deadline_s=1200,non_target_memory_ceilings_mib=[540]*8))
        cid=command(launch).stdout.strip();owned=True
        result['container_id']=cid
        emit(stage='started',model=model['name'],method=cfg['method'],container=name,kv_gib=cfg['kv_gib'])
        def monitor():
            while not stop.is_set():
                try:
                    row=smi()
                    with (out/'telemetry.jsonl').open('a') as handle:
                        handle.write(json.dumps(row)+'\n')
                    assert max(row['temperatures_c'])<80, 'Temperature >=80C'
                    assert all(v==540 for i,v in enumerate(row['memories_mib']) if i not in DEVICES), 'Non-target allocation changed'
                    assert max(row['memories_mib'][i] for i in DEVICES)<56000, 'Target allocation >=56000 MiB'
                    assert time.monotonic()-start<1200, 'Model run deadline'
                    running=command(['docker','ps','-q','--no-trunc']).stdout.split()
                    assert not unexpected_containers(running,cid), 'Unexpected concurrent container'
                except Exception as exc:
                    failures.append(repr(exc));emit(stage='safety_stop',error=repr(exc))
                    command(['docker','stop','--timeout','15',cid],check=False)
                    return
                stop.wait(10)
        watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
        boot=time.monotonic()
        while time.monotonic()-boot<420 and not failures:
            assert command(['docker','inspect','--format','{{.State.Running}}',cid]).stdout.strip()=='true', 'Server exited during startup'
            try:
                response=client.api('/v1/models',timeout=2)
                assert response['data'][0]['id']==slug, 'Wrong service identity'
                save(out,'models.json',response)
                break
            except (OSError,client.http.client.HTTPException):
                time.sleep(5)
        else:
            raise RuntimeError('Readiness deadline/safety stop')
        result['startup_s']=time.monotonic()-boot
        emit(stage='ready',model=model['name'],startup_s=result['startup_s'])
        result['correctness']=quality(client,helper,out,cfg)
        assert all(r['passed'] for r in result['correctness']), 'Correctness gate failed'
        result['performance']=performance(client,out,cases)
        assert not failures, failures
        result['status']='passed'
    except Exception as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc())
        emit(stage='failed',model=model['name'],error=repr(exc))
    finally:
        stop.set()
        if watcher:
            watcher.join(timeout=20)
        if owned:
            logs=command(['docker','logs','--timestamps',name],timeout=60,check=False)
            (out/'server.log').write_text(logs.stdout+logs.stderr)
            save(out,'container-before-stop.json',json.loads(command(['docker','inspect',name]).stdout))
            command(['docker','stop','--timeout','20',name],check=False)
            save(out,'container-after-stop.json',json.loads(command(['docker','inspect',name]).stdout))
        polls=[];deadline=time.monotonic()+60
        while True:
            after=smi();polls.append(after)
            save(out,'cleanup-polls.json',polls)
            if after['memories_mib']==[540]*8 or time.monotonic()>=deadline:
                break
            time.sleep(2)
        save(out,'hardware-after.json',after)
        result.update(memory_released=after['memories_mib']==[540]*8,elapsed_s=time.monotonic()-start,
                      finished_epoch_s=time.time(),safety_failures=failures)
        save(out,'result.json',result)
        emit(stage='complete',model=model['name'],status=result['status'],elapsed_s=result['elapsed_s'],
             memory_released=result['memory_released'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    parser.add_argument('--variant',choices=['coverage','baseline'],default='coverage')
    parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--suite',choices=['coverage','formal','both'],default='coverage')
    args=parser.parse_args()
    inventory=json.loads((ROOT/'model-inventory.json').read_text())
    models={row['name']:row for row in inventory['models']}
    progress=[]
    batch_slug=re.sub('[^a-z0-9]+','-',args.models[0].lower()).strip('-')
    for name in args.models:
        model=models[name]
        assert model['weights_header_qualified'], name+': weights incomplete'
        cases=CASES if args.suite=='coverage' else (FORMAL_CASES if args.suite=='formal' else CASES+FORMAL_CASES)
        result=run_model(model,args.variant,args.attempt,cases,args.suite)
        progress.append(result)
        save(ROOT,f'legacy-{args.variant}-{args.suite}-r{args.attempt}-{batch_slug}-progress.json',progress)
        if not result['memory_released'] or result['safety_failures']:
            raise SystemExit('Stop sweep: hardware/resource isolation gate did not pass')
    emit(stage='sweep_complete',passed=sum(row['status']=='passed' for row in progress),attempted=len(progress))
