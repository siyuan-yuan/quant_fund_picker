"""Read-only A1 v2 checkpoint census; writes only the separate audit directory."""
from pathlib import Path
import csv
import hashlib
import json
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/v6/g0_full_pit/csrc_stratified'
V2 = BASE / 'a1/v2'
OUT = BASE / 'a1/root_cause_audit'
ARTIFACTS = ['required_classification_checkpoints.csv', 'document_dispositions.csv',
             'classification_clause_clusters.csv', 'classification_state_timeline.csv',
             'g0_amendment_a1_gate_metrics.csv', 'g0_amendment_a1_gate_review.md']

def csvread(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def jsonread(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def writecsv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def hashes():
    return {name: hashlib.sha256((V2 / name).read_bytes()).hexdigest() for name in ARTIFACTS}

def main():
    before = hashes()
    checkpoints = csvread(V2 / 'document_dispositions.csv')
    aliases = csvread(BASE.parent / 'csrc_stratified_aliases.csv')
    samples = {x['family_key']: x for x in csvread(BASE.parent / 'csrc_stratified_sample.csv')}
    codes = defaultdict(set)
    for a in aliases:
        codes[a['family_key']].add(a['query_code'])
    metadata = jsonread(BASE / 'metadata.jsonl')
    latest = {x['source_document']: x for x in jsonread(BASE / 'parsed_documents.jsonl')}
    sections = {x['source_document']: x for x in jsonread(V2 / 'section_cache.jsonl')}
    docs = {}
    familydocs = defaultdict(list)
    reverse = {code: family for family, members in codes.items() for code in members}
    for m in metadata:
        code, upload = str(m['fundCode']).zfill(6), str(m['uploadInfoId'])
        source = f'{code}_{upload}.pdf'
        p, s = latest.get(source, {}), sections.get(source, {})
        detail = p.get('document') or p.get('failure') or {}
        d = dict(source_document=source, share_code=code, upload_info_id=upload,
                 report_name=m.get('reportName', ''), report_code=m.get('reportCode', ''),
                 known_at=m.get('reportSendDate', ''), metadata_upload_date=m.get('uploadDate', ''),
                 pdf_exists=(BASE / 'pdf' / source).is_file(),
                 section_cache_status=s.get('extraction_status', ''),
                 section_cache_error=s.get('extraction_error', ''),
                 section_clause=s.get('section_text', ''),
                 parser_status=p.get('status', ''), parser_version=p.get('parser_version', ''),
                 parser_reason=detail.get('reason', ''), parser_clause=detail.get('evidence', ''),
                 effective_date=detail.get('effective_date') or s.get('effective_date') or '',
                 official_url=m.get('attachFilePath', ''),
                 metadata_locator=f'metadata.jsonl:fundCode={code},uploadInfoId={upload}')
        docs[source] = d
        if code in reverse:
            familydocs[reverse[code]].append(d)
    rows, candidates = [], []
    for c in checkpoints:
        family, source = c['family_key'], c['source_document']
        synthetic = source.startswith('fund_master:')
        available = sorted(familydocs[family], key=lambda d: (d['known_at'], d['source_document']))
        relevant = [d for d in available if '招募' in d['report_name']] if c['trigger_type'] == 'INCEPTION' else available
        direct = docs.get(source)
        prior = [d for d in relevant if d['known_at'] and c['known_at'] and d['known_at'] <= c['known_at']]
        # Primary buckets identify the earliest *observed* blocker, not exclusive causal mechanisms.
        if c['disposition'] != 'UNRESOLVED':
            bucket, why = 'RESOLVED_BASELINE', 'v2 disposition retained'
        elif synthetic:
            bucket = 'LIFECYCLE_EVIDENCE_JOIN_NOT_IMPLEMENTED'
            why = 'Synthetic upload id has no matching official upload; adjudicate_checkpoints joins only exact upload_info_id'
        elif direct is None:
            bucket, why = 'NO_CANDIDATE_IN_LOCAL_METADATA', 'Frozen source absent from local metadata; official absence not established'
        elif not direct['pdf_exists']:
            bucket, why = 'LOCAL_PDF_MISSING', 'Exact metadata candidate exists but cached PDF absent'
        elif c['failure_reason'] == 'MISSING_PREDECESSOR':
            bucket, why = 'ADJUDICATION_PREDECESSOR_MISSING', 'Clause extracted but v2 cannot establish state or predecessor required for NOOP'
        elif 'effective date' in direct['parser_reason']:
            bucket, why = 'DATE_OR_STAGE_UNDETERMINED', direct['parser_reason']
        elif direct['parser_reason'] == 'No explicit equity allocation constraint found':
            bucket, why = 'UNDETERMINED_SECTION_VS_SEMANTICS', 'Combined extractor/parser reported no equity constraint; cache lacks full section/text diagnostics'
        else:
            bucket, why = 'UNDETERMINED', 'Existing local evidence does not isolate cause'
        flags = []
        if synthetic and not prior: flags.append('NO_PRE_CHECKPOINT_CANDIDATE_IN_LOCAL_METADATA')
        if synthetic and any(d['parser_clause'] for d in prior): flags.append('PRIOR_OFFICIAL_CANDIDATE_HAS_PARSED_CLAUSE')
        if direct and direct['section_clause']: flags.append('DENOMINATOR_CACHE_HAS_CLAUSE')
        if direct and not direct['parser_clause']: flags.append('ADJUDICATION_INPUT_HAS_NO_CLAUSE')
        row = dict(c, primary_root_cause=bucket, root_cause_evidence=why,
                   secondary_flags=';'.join(flags), stratum_status=samples[family]['status'],
                   stratum_era=samples[family]['inception_era'],
                   alias_codes=';'.join(sorted(codes[family])), synthetic_checkpoint=synthetic,
                   exact_metadata_match=bool(direct), family_metadata_document_count=len(available),
                   relevant_candidate_count=len(relevant), pre_checkpoint_candidate_count=len(prior),
                   pre_checkpoint_parsed_clause_count=sum(bool(d['parser_clause']) for d in prior),
                   first_candidate_document=relevant[0]['source_document'] if relevant else '',
                   first_candidate_known_at=relevant[0]['known_at'] if relevant else '',
                   trace_status='LOCAL_AUTOMATED_TRACE;LEGAL_HUMAN_REVIEW_PENDING',
                   official_absence_established=False,
                   current_section_status=(direct or {}).get('section_cache_status', ''),
                   report_code=(direct or {}).get('report_code', ''),
                   report_name=(direct or {}).get('report_name', ''),
                   current_section_error=(direct or {}).get('section_cache_error', ''),
                   current_parser_status=(direct or {}).get('parser_status', ''),
                   current_parser_reason=(direct or {}).get('parser_reason', ''),
                   current_clause=(direct or {}).get('parser_clause', ''))
        rows.append(row)
        if c['trigger_type'] in {'INCEPTION', 'TRANSFORMATION'}:
            chosen = relevant if synthetic else ([direct] if direct else [])
            for d in chosen:
                candidates.append(dict(checkpoint_id=c['checkpoint_id'], family_key=family,
                                       trigger_type=c['trigger_type'], checkpoint_known_at=c['known_at'],
                                       match_method='family_alias_candidate_not_adjudicated' if synthetic else 'exact_frozen_source',
                                       known_by_checkpoint=bool(d['known_at'] and c['known_at'] and d['known_at'] <= c['known_at']), **d))
    unresolved = [r for r in rows if r['disposition'] == 'UNRESOLVED']
    inception = [r for r in rows if r['trigger_type'] == 'INCEPTION']
    transformation = [r for r in rows if r['trigger_type'] == 'TRANSFORMATION']
    assert (len(rows), len(unresolved), len(inception), len(transformation)) == (1371, 1251, 60, 49)
    assert len({r['checkpoint_id'] for r in unresolved}) == 1251
    OUT.mkdir(parents=True, exist_ok=True)
    writecsv('unresolved_checkpoint_census.csv', unresolved)
    writecsv('inception_trace.csv', inception)
    writecsv('transformation_trace.csv', transformation)
    writecsv('lifecycle_candidate_documents.csv', candidates)
    summary = dict(total_checkpoints=len(rows), unresolved=len(unresolved),
                   primary_root_causes=dict(Counter(r['primary_root_cause'] for r in unresolved)),
                   inception_with_prior_prospectus=sum(r['pre_checkpoint_candidate_count'] > 0 for r in inception),
                   inception_with_prior_parsed_clause=sum(r['pre_checkpoint_parsed_clause_count'] > 0 for r in inception),
                   transformation_buckets=dict(Counter(r['primary_root_cause'] for r in transformation)),
                   unresolved_trigger_counts=dict(Counter(r['trigger_type'] for r in unresolved)),
                   section_uncomparable_report_codes=dict(Counter(r['report_code'] for r in unresolved if r['trigger_type']=='SECTION_UNCOMPARABLE')),
                   methodology='Primary earliest observed blocker, secondary flags; local caches only. No official absence inference. No new adjudication.',
                   frozen_artifact_hashes_before=before, frozen_artifact_hashes_after=hashes())
    assert summary['frozen_artifact_hashes_before'] == summary['frozen_artifact_hashes_after']
    (OUT / 'census_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if not k.startswith('frozen')}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
