"""Read-only window throughput; budget ledger is the sole cost authority."""


def query(boundary):
    # boundary is a trusted SQL timestamp expression or a bound placeholder.
    return f"""WITH bounds AS (SELECT {boundary} AS cutoff),
    groups AS (SELECT kind,action FROM commulingo_pipeline_jobs GROUP BY 1,2),
    publications AS (
        SELECT j.kind,j.action,count(DISTINCT a.job_id) AS applied,
            count(DISTINCT a.job_id) FILTER (WHERE EXISTS (
                SELECT 1 FROM commulingo_pipeline_attempts t WHERE t.job_id=a.job_id
                    AND t.started_at>(SELECT cutoff FROM bounds) AND t.started_at<=a.created_at)) AS measured_applied
        FROM commulingo_pipeline_artifacts a JOIN commulingo_pipeline_jobs j ON j.id=a.job_id
        WHERE a.stage='submit' AND a.value->>'status'='approved'
          AND a.created_at>(SELECT cutoff FROM bounds) GROUP BY 1,2),
    costs AS (
        SELECT j.kind,j.action,sum(b.actual) AS actual,
            coalesce(sum(b.reserved) FILTER (WHERE b.actual IS NULL),0) AS reserved,
            count(*) FILTER (WHERE b.actual IS NULL) AS unsettled
        FROM commulingo_pipeline_budget b JOIN commulingo_pipeline_jobs j ON j.id=b.job_id
        WHERE b.created_at>(SELECT cutoff FROM bounds) GROUP BY 1,2),
    attempts AS (
        SELECT j.kind,j.action,count(*) AS attempts,
            count(*) FILTER (WHERE a.finished_at IS NULL) AS unfinished,
            sum(coalesce(a.duration_seconds,extract(epoch FROM now()-a.started_at))) AS seconds,
            sum(coalesce((a.metrics->>'rounds_used')::integer,0)) AS rounds,
            count(*) FILTER (WHERE a.metrics ? 'input_tokens') AS token_measured,
            sum((a.metrics->>'input_tokens')::bigint) AS input_tokens,
            sum((a.metrics->>'output_tokens')::bigint) AS output_tokens,
            sum((a.metrics->>'cache_read_tokens')::bigint) AS cache_read_tokens,
            count(*) FILTER (WHERE a.metrics->>'preflight_no_model'='true') AS preflight_no_model,
            sum((a.metrics->>'fetch_backoff_hits')::integer) AS fetch_backoff_hits,
            sum((a.metrics->>'review_context_original_chars')::bigint) AS review_original_chars,
            sum((a.metrics->>'review_context_chars')::bigint) AS review_chars,
            count(*) FILTER (WHERE a.stage='research') AS research,
            count(*) FILTER (WHERE a.next_stage='research' AND a.stage IN ('draft','validate','review')) AS rework
        FROM commulingo_pipeline_attempts a JOIN commulingo_pipeline_jobs j ON j.id=a.job_id
        WHERE a.started_at>(SELECT cutoff FROM bounds) GROUP BY 1,2),
    first_checks AS (
        SELECT DISTINCT ON (job_id) job_id,started_at,metrics
        FROM commulingo_pipeline_attempts WHERE (stage='draft' OR metrics->>'workflow'='editor')
            AND (metrics ? 'preflight_passed' OR metrics ? 'preflight_failures' OR coalesce((metrics->>'terminal_calls')::integer,0)>0)
        ORDER BY job_id,started_at,id),
    validation AS (
        SELECT j.kind,j.action,count(*) AS checked,
            count(*) FILTER (WHERE f.metrics->>'preflight_passed'='true'
                AND coalesce((f.metrics->>'preflight_failures')::integer,0)=0
                AND coalesce((f.metrics->>'terminal_calls')::integer,0)<=1) AS first_pass
        FROM first_checks f JOIN commulingo_pipeline_jobs j ON j.id=f.job_id
        WHERE f.started_at>(SELECT cutoff FROM bounds) GROUP BY 1,2),
    editorial AS (
        SELECT j.kind,j.action,
            coalesce(sum(jsonb_array_length(coalesce(a.value->'resolved_issues','[]'::jsonb)))
                FILTER (WHERE a.stage='submit' AND a.value->>'status'='approved'),0) AS resolved_issues,
            coalesce(sum(jsonb_array_length(coalesce(a.value->'deferred_issues','[]'::jsonb)))
                FILTER (WHERE a.stage='submit' AND a.value->>'status'='approved'),0) AS deferred_issues,
            count(*) FILTER (WHERE a.value ? 'hold_reason') AS no_progress_holds,
            count(*) FILTER (WHERE a.stage='review' AND a.value->>'decision'='revise') AS factual_revisions
        FROM commulingo_pipeline_artifacts a JOIN commulingo_pipeline_jobs j ON j.id=a.job_id
        WHERE a.created_at>(SELECT cutoff FROM bounds) GROUP BY 1,2),
    dispositions AS (
        SELECT j.kind,j.action,
            count(*) FILTER (WHERE a.stage='discover') AS discoveries,
            count(*) FILTER (WHERE a.stage='judge') AS judgments
        FROM commulingo_pipeline_artifacts a JOIN commulingo_pipeline_jobs j ON j.id=a.job_id
        WHERE a.created_at>(SELECT cutoff FROM bounds) GROUP BY 1,2)
    SELECT coalesce(json_agg(row_to_json(r)),'[]'::json) FROM (
        SELECT g.kind,g.action,coalesce(p.applied,0) AS applied,coalesce(p.measured_applied,0) AS measured_applied,
            c.actual,coalesce(c.reserved,0) AS reserved,coalesce(c.unsettled,0) AS unsettled,
            coalesce(a.attempts,0) AS attempts,coalesce(a.unfinished,0) AS unfinished,
            a.seconds,a.rounds,a.research,a.rework,v.checked,v.first_pass,
            a.token_measured,a.input_tokens,a.output_tokens,a.cache_read_tokens,
            a.preflight_no_model,a.fetch_backoff_hits,a.review_original_chars,a.review_chars,
            d.discoveries,d.judgments,e.resolved_issues,e.deferred_issues,e.no_progress_holds,e.factual_revisions
        FROM groups g LEFT JOIN publications p USING(kind,action)
        LEFT JOIN costs c USING(kind,action) LEFT JOIN attempts a USING(kind,action)
        LEFT JOIN validation v USING(kind,action) LEFT JOIN dispositions d USING(kind,action)
        LEFT JOIN editorial e USING(kind,action)
        ORDER BY g.kind,g.action) r"""


def render(rows):
    lines = ['생산율 (기간 비용/반영; 신규 시도 계측 시작 전 실행시간은 포함하지 않음):']
    for r in rows:
        cost = r['actual']
        rate = f"${cost/r['applied']:.4f}/반영" if cost is not None and r['applied'] else '반영당 비용 산출 불가'
        if r['unsettled']:
            rate += f" (미정산 {r['unsettled']}건, 예약 ${r['reserved']:.4f}; 비용 미완결)"
        lines.append(f"  {r['kind']}/{r['action']}: 반영 {r['applied']} · {rate}")
        if r['seconds'] and not r['unfinished']:
            lines.append(f"    계측 범위 {r['measured_applied']*3600/float(r['seconds']):.2f} 반영/실행시간 · 재조사 {r['rework'] or 0}/{r['attempts']} 시도")
        lines.append(f"    계측 시도 {r['attempts']} · 미종료 {r['unfinished']} · "
                     f"실행 {float(r['seconds'] or 0):.0f}초 · 재조사 전환 {r['rework'] or 0}")
        if r['checked']:
            lines.append(f"    첫 저장 검증 {r['first_pass']}/{r['checked']} · "
                         f"후보 발견 {r['discoveries'] or 0} · 무편집 판단 {r['judgments'] or 0}")
        if r.get('token_measured'):
            lines.append(f"    토큰 계측 {r['token_measured']}시도: 입력 {r['input_tokens']} · 출력 {r['output_tokens']} · 캐시 읽기 {r.get('cache_read_tokens') or 0}")
        if r.get('review_original_chars'):
            lines.append(f"    검토 문맥 {r['review_original_chars']}→{r['review_chars']}자 (비용 절감률 아님)")
        if r.get('preflight_no_model') or r.get('fetch_backoff_hits'):
            lines.append(f"    유료 예약 전 종료 {r.get('preflight_no_model') or 0} · 실패 URL 재시도 억제 {r.get('fetch_backoff_hits') or 0}")
        if r.get('resolved_issues') or r.get('deferred_issues') or r.get('no_progress_holds'):
            lines.append(f"    승인·반영한 결함 해결 {r.get('resolved_issues') or 0} · 미해결 {r.get('deferred_issues') or 0} · "
                         f"진전 없는 반복 보류 {r.get('no_progress_holds') or 0} · 사실 수정 요청 {r.get('factual_revisions') or 0}")
    return lines
