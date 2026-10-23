"""CPU production-method contracts; not a full serving validation."""
import ast
import copy
import importlib.util
import json
import statistics
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
from enum import Enum

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / 'python/sglang/srt'
spec = importlib.util.spec_from_file_location('policy_diag', SRT/'observability/policy_diagnostics.py')
diag = importlib.util.module_from_spec(spec); spec.loader.exec_module(diag)


def methods():
    tree = ast.parse((SRT/'managers/schedule_policy.py').read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'SchedulePolicy')
    wanted = {'_shortest_prefill_work','_uncached_len','_sort_by_shortest_prefill','_sort_by_hrrn','shortest_prefill_chunk_limit','_determine_active_policy'}
    cls.body = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in wanted]
    ns = {'CacheAwarePolicy': NS(SHORTEST_PREFILL_FIRST='shortest-prefill-first',HRRN='hrrn',LPM='lpm'), 'CacheAgnosticPolicy': NS(FCFS='fcfs'), 'WAITING_QUEUE_PREFIX_MATCH_MAX':128, '_ceil_div': lambda a,b: -(-a//b)}
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<production-policy>','exec'),ns)
    return ns['SchedulePolicy']

Policy = methods()


def request(rid, work=1000, ts=10, output=0, matched=0):
    return NS(rid=rid,origin_input_ids=range(work),output_ids=range(output),
              num_matched_prefix_tokens=matched,arrival_processed_tokens=5,
              time_stats=NS(wait_queue_entry_time=ts,forward_entry_time=0),
              full_untruncated_fill_ids=range(work),prefix_len=matched,extend_len=work-matched,
              token_indices_to_pool=None,beam_group=None,lora_id=None,
              cache_request_handle=rid,init_next_round_input=lambda cache: None)


def sample(policy='shortest-prefill-first', n=0):
    rows=[]
    p=diag.PolicySampler(10,clock=lambda:100,emit=rows.append).begin(policy,n)
    return p,rows


class Contracts(unittest.TestCase):
    def test_disabled_and_cadence(self):
        clock=Mock(side_effect=AssertionError('clock'))
        self.assertIsNone(diag.PolicySampler(clock=clock).begin('hrrn',0));clock.assert_not_called()
        now=[100]; s=diag.PolicySampler(10,clock=lambda:now[0])
        self.assertIsNotNone(s.begin('hrrn',0)); self.assertIsNone(s.begin('hrrn',0))
        now[0]=109;self.assertIsNone(s.begin('hrrn',0))
        now[0]=110;self.assertIsNotNone(s.begin('hrrn',0))
        self.assertIsNone(s.begin('fcfs',0))

    def test_sort_order_and_key_coverage(self):
        # Ties, duplicate deferral, fully cached and output-bearing work.
        cases=[[],[request('b'),request('a')],
               [request('cached',2000,matched=2000),request('long',10000,1),request('short',1000,90)],
               [request('output',10000,output=5000,matched=12000),request('fresh',1000)]]
        for policy in ('shortest-prefill-first','hrrn'):
            for requests in cases:
                for deferred in (set(),{r.rid for r in requests[:1]}):
                    a,b=list(requests),list(requests);p,rows=sample(policy,len(b))
                    fn=Policy._sort_by_shortest_prefill if policy=='shortest-prefill-first' else Policy._sort_by_hrrn
                    args=() if policy=='shortest-prefill-first' else (500,)
                    before=[copy.deepcopy(vars(r.time_stats)) for r in b]
                    fn(a,deferred,*args);fn(b,deferred,*args,diagnostics=p)
                    self.assertEqual([r.rid for r in a],[r.rid for r in b])
                    self.assertEqual([vars(r.time_stats) for r in requests],before)
                    if b:p.head(b[0])
                    p.finish();self.assertEqual(rows[0]['queue'].get('keys_seen',0),len(b))
                    self.assertNotIn('rid',json.dumps(rows))

    def test_original_sort_semantics(self):
        reqs=[request('b',2000,5),request('a',2000,5),request('z',1000,90),request('c',0,1)]
        for policy in ('shortest-prefill-first','hrrn'):
            deferred={'c'};p,_=sample(policy,len(reqs)); actual=list(reqs)
            if policy=='shortest-prefill-first':
                expected=sorted(reqs,key=lambda r:(r.rid in deferred,max(1,len(r.origin_input_ids)+len(r.output_ids)-r.num_matched_prefix_tokens),r.time_stats.wait_queue_entry_time))
                Policy._sort_by_shortest_prefill(actual,deferred,p)
            else:
                def original(r):
                    if r.rid in deferred:return(float('inf'),r.rid)
                    work=max(0,len(r.origin_input_ids)-r.num_matched_prefix_tokens)
                    return(-float('inf'),r.rid) if work==0 else(-max(0,500-r.arrival_processed_tokens)/work,r.rid)
                expected=sorted(reqs,key=original);Policy._sort_by_hrrn(actual,deferred,500,p)
            self.assertEqual(actual,expected)

    def test_episode_reset_unknown_cancel_and_pairs(self):
        a=request('old',5000,1);b=request('new',1000,90);p,rows=sample(n=2)
        Policy._sort_by_shortest_prefill([a,b],set(),p);p.head(b);p.finish()
        self.assertEqual(rows[0]['oldest_episode'],dict(age_seconds=99,work_tokens=5000,prefix_deferred=False))
        a.time_stats.wait_queue_entry_time=99
        p,rows=sample(n=1);Policy._sort_by_shortest_prefill([a],set(),p);p.finish()
        self.assertEqual(rows[0]['oldest_episode']['age_seconds'],1)
        for stamp in (0,None,float('nan'),101):
            a.time_stats.wait_queue_entry_time=stamp;p,rows=sample('hrrn',1)
            Policy._sort_by_hrrn([a],set(),500,p);p.finish()
            self.assertEqual(rows[0]['queue']['age_unknown'],1);self.assertNotIn('oldest_episode',rows[0])
        p,rows=sample(n=0);Policy._sort_by_shortest_prefill([],set(),p);p.finish();self.assertNotIn('oldest_episode',rows[0])

    def test_continuation_preserves_limits(self):
        obj=Policy();chunk=request('chunk',100000)
        for policy in ('shortest-prefill-first','hrrn'):
            obj.policy=policy
            for budget in (0,256,512,20480,20500):
                for queue in ([],[request('a',2000),request('b',3000)],[request('large',200000)]):
                    p,rows=sample(policy,len(queue))
                    off=obj.shortest_prefill_chunk_limit(chunk,queue,budget,256)
                    on=obj.shortest_prefill_chunk_limit(chunk,queue,budget,256,p)
                    self.assertEqual(off,on)
                    p.continuation_admitted(chunk,False);p.finish()
                    v=rows[0]['continuation'];self.assertEqual(v['policy_cap_returned'],on is not None)
                    self.assertEqual(v['admitted_tokens'],0)
                    if on is not None:self.assertEqual(v['cap_tokens'],on)
        obj.policy='shortest-prefill-first';p,rows=sample()
        self.assertEqual(obj.shortest_prefill_chunk_limit(chunk,[request('a',2000),request('b',3000)],20480,256,p),15360)
        self.assertEqual(p.record['continuation']['reserved_tokens'],5120)

    def test_effective_policy_threshold(self):
        obj=Policy()
        for configured in ('hrrn','lpm','shortest-prefill-first'):
            obj.policy=configured
            for n in (0,128,129,384):
                actual=obj._determine_active_policy([None]*n)
                expected='fcfs' if configured in ('hrrn','lpm') and n>128 else configured
                self.assertEqual(actual,expected)

    def test_unvisited_population_and_missing_scan_are_explicit(self):
        p,rows=sample(n=3)
        p.visit(request('first'), 'admitted')
        p.end('adder_result', adder_result='other', candidate_admitted=True)
        p.finish(scan_complete=True)
        self.assertEqual(rows[0]['pass_end']['unvisited_candidates'],2)
        self.assertEqual(rows[0]['visits']['admitted']['count'],1)
        p,rows=sample(n=3);p.finish()
        self.assertNotIn('pass_end',rows[0])

    def test_sink_failure_isolated(self):
        p=diag.PolicySampler(10,clock=lambda:100,emit=Mock(side_effect=OSError)).begin('hrrn',0)
        p.finish()

    def test_pass_budget_residuals_preserve_coexhaustion_and_unknown(self):
        class NoConversion:
            def __float__(self):
                raise AssertionError('No device or custom scalar conversion')
        for values, expected in (
            ({'rem_input_tokens': -256, 'rem_chunk_tokens': 0, 'rem_dllm_tokens': None},
             {'rem_input_tokens': -256, 'rem_chunk_tokens': 0, 'rem_dllm_tokens': None}),
            ({'rem_input_tokens': 1024, 'rem_chunk_tokens': None},
             {'rem_input_tokens': 1024, 'rem_chunk_tokens': None}),
            ({'rem_input_tokens': float('nan'), 'rem_chunk_tokens': NoConversion(), 'rem_dllm_tokens': True}, {}),
        ):
            adder=NS(**values);adder.budget_state=Mock(side_effect=AssertionError('budget re-evaluation'))
            p,rows=sample(n=2);p.visit(request('a'),'admitted')
            p.end('adder_result',adder_result='other',candidate_admitted=True)
            p.finish(scan_complete=True,adder=adder)
            self.assertEqual(rows[0]['pass_end']['budget_remaining'],expected)
            self.assertEqual(rows[0]['pass_end']['unvisited_candidates'],1)
            adder.budget_state.assert_not_called()
            for name,value in values.items():self.assertIs(vars(adder)[name],value)
        p,rows=sample();p.finish(scan_complete=True)
        self.assertNotIn('budget_remaining',rows[0]['pass_end'])

    def test_actual_admission_loop_no_extra_readiness_or_order_changes(self):
        # Execute the unmodified production loop, with CPU resources. This covers
        # every existing early stop/skip and successful/rejected adder results.
        tree=ast.parse((SRT/'managers/scheduler.py').read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Scheduler')
        loop=next(x for x in ast.walk(cls) if isinstance(x,ast.For) and isinstance(x.iter,ast.Attribute) and x.iter.attr=='waiting_queue' and any(isinstance(y,ast.Name) and y.id=='policy_diagnostics' for y in ast.walk(x)))
        E=Enum('AddReqResult','CONTINUE NO_TOKEN OTHER')
        finish=next(x for x in ast.walk(cls) if isinstance(x,ast.If) and isinstance(x.body[0],ast.Expr) and isinstance(x.body[0].value,ast.Call) and isinstance(x.body[0].value.func,ast.Attribute) and x.body[0].value.func.attr=='finish' and any(k.arg=='scan_complete' for k in x.body[0].value.keywords))
        code=compile(ast.fix_missing_locations(ast.Module(body=[loop,finish],type_ignores=[])),'<production-loop>','exec')
        for scenario in ('admit','admit_no_token','admit_other','pass_budget','lora','pooling','slots','prefetch','reissue','staging','reject','continue_reject'):
            outputs=[]
            for enabled in (False,True):
                calls=[];r=request('candidate');r.kv=NS(holds_mamba=False,mamba_cow_src_index=None,mamba_needs_clear=False)
                def readiness(name,result):
                    calls.append(name);return result
                can_run=[NS(token_indices_to_pool=True,kv=NS(holds_kv=False))] if scenario=='pooling' else []
                adder=NS(rem_input_tokens=0 if scenario=='pass_budget' else 20480,rem_chunk_tokens=20480,rem_dllm_tokens=None,can_run_list=can_run,budget_state=lambda:readiness('budget',E.OTHER if scenario=='pass_budget' else E.CONTINUE))
                def add(req,**kwargs):
                    calls.append('add')
                    if scenario not in ('reject','continue_reject'):adder.can_run_list.append(req)
                    if scenario=='admit_other':adder.rem_input_tokens=-128;adder.rem_chunk_tokens=0
                    return E.NO_TOKEN if scenario in ('reject','admit_no_token') else E.OTHER if scenario=='admit_other' else E.CONTINUE
                adder.add_one_req=add
                tc=NS(check_prefetch_progress=lambda _:readiness('prefetch',scenario!='prefetch'),pop_prefetch_loaded_span=lambda _:(0,0))
                r.time_stats.observe_storage_prefetch_ready=lambda _:None;r.time_stats.observe_storage_staging_ready=lambda _:None
                scheduler=NS(waiting_queue=[r],enable_lora=scenario=='lora',can_schedule_lora_req=lambda *_:readiness('lora',False),get_num_allocatable_reqs=lambda *a,**k:0 if scenario=='slots' else 96,enable_priority_preemption=False,enable_hicache_storage=True,enable_lmcache=False,tree_cache=tc,_prefetch_after_device_hit_loss=lambda _:readiness('reissue',scenario=='reissue'),chunked_req=None,truncation_align_size=None)
                batch=NS(reqs=[],batch_is_full=False,is_empty=lambda:True)
                pipeline=NS(prepare_staged_prefetch=lambda _:readiness('staging',scenario!='staging'))
                p,rows=sample(n=1)
                ns=dict(self=scheduler,adder=adder,running_batch=batch,running_loras=set(),buffer_pipeline=pipeline,AddReqResult=E,tuning_recorder=NS(observe=lambda *a,**k:None),policy_diagnostics=p if enabled else None)
                exec(code,ns)
                outputs.append((calls,[getattr(q,'rid','existing') for q in can_run],batch.batch_is_full))
                if enabled:
                    self.assertEqual(len(rows),1);self.assertEqual(sum(v['count'] for v in rows[0]['visits'].values()),1)
                    end=rows[0]['pass_end']
                    self.assertEqual(end['budget_remaining'],{k:vars(adder)[k] for k in ('rem_input_tokens','rem_chunk_tokens','rem_dllm_tokens')})
                    expected='pass_budget_stop' if scenario=='pass_budget' else 'request_slots_stop' if scenario=='slots' else 'adder_result' if scenario in ('reject','admit_no_token','admit_other') else 'queue_scan_exhausted'
                    self.assertEqual(end['cause'],expected)
                    if expected=='adder_result':
                        self.assertEqual(end['candidate_admitted'],scenario!='reject')
                        self.assertEqual(end['adder_result'],'other' if scenario=='admit_other' else 'no_token')
                    self.assertEqual(end['visited_candidates'],1)
                    self.assertEqual(end['unvisited_candidates'],0)
            self.assertEqual(*outputs,scenario)


if __name__=='__main__':
    unittest.main()
