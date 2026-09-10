import csv,json,subprocess,re
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
sample={r['family_key']:r for r in rows(BASE.parent/'csrc_stratified_sample.csv')}
meta={str(r['uploadInfoId']):r for r in rows(BASE/'metadata.csv')}
cache={r['source_document']:r for r in map(json.loads,(BASE/'a1/v2/section_cache.jsonl').read_text(encoding='utf-8').splitlines())}
parsed={r['source_document']:r for r in map(json.loads,(BASE/'parsed_documents.jsonl').read_text(encoding='utf-8').splitlines())}
pop=[r for r in rows(BASE/'a1/v2/document_dispositions.csv') if r['trigger_type']=='SECTION_UNCOMPARABLE' and r['disposition']=='UNRESOLVED']
for r in pop:
    s=sample[r['family_key']];r['stratum']=s['status']+'|'+s['inception_era']
counts=Counter(r['stratum'] for r in pop)
selected=[r for s in sorted(counts) for r in sorted((r for r in pop if r['stratum']==s),key=lambda x:x['checkpoint_id'])[:3]]
out=[]
for r in selected:
    doc=r['source_document'];m=meta.get(r['upload_info_id'],{});c=cache.get(doc,{})
    o={k:r[k] for k in ['checkpoint_id','family_key','stratum','source_document','known_at']}
    o.update(population_count=counts[r['stratum']],report_name=m.get('reportName',''),report_code=m.get('reportCode',''),cache_error=c.get('extraction_error',''),parser_record=json.dumps(parsed.get(doc,{}),ensure_ascii=False))
    try:
        p=subprocess.run(['pdftotext','-layout',str(BASE/'pdf'/doc),'-'],capture_output=True,timeout=30)
        text=p.stdout.decode('utf-8',errors='replace');pages=text.split('\f')
        hits=[]
        for i,page in enumerate(pages,1):
            for match in re.finditer(r'投资范围|资产配置|投资目标|投资策略|投资限制|股票.{0,30}(?:比例|投资)|(?:不低于|不高于).{0,30}[%％]',page):
                hits.append({'page':i,'excerpt':page[max(0,match.start()-100):match.end()+500]})
        o.update(text_chars=len(text),extraction_status='success' if p.returncode==0 else 'error',evidence=json.dumps(hits[:45],ensure_ascii=False))
    except subprocess.TimeoutExpired:o.update(text_chars=0,extraction_status='timeout',evidence='[]')
    out.append(o)
    print(doc,o['text_chars'],flush=True)
with (HERE/'section_uncomparable_sample_raw.json').open('w',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(counts),ensure_ascii=True))
