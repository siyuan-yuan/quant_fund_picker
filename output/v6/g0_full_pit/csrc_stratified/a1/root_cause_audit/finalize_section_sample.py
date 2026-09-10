import csv,json,re
from pathlib import Path
H=Path(__file__).resolve().parent
data=json.loads((H/'section_uncomparable_sample_raw.json').read_text(encoding='utf-8'))
notes={
'163001_155371.pdf':(51,'CLAUSE_PRESENT_IMPLEMENTATION_GAP','股票资产90%-95%；基金的投资、投资目标、投资范围和投资策略均可读。'),
'017624_1385081.pdf':(45,'JOINT_ASSET_CONSTRAINT','股票、可交换债券、可转换债券合计10%-30%；并非股票单独10%-30%。'),
'017464_1089218.pdf':(2,'JOINT_ASSET_CONSTRAINT','股票、可转债及可交换债券合计10%～30%；概要表格把30拆为跨行3/0，需规范化。'),
'017464_893158.pdf':(50,'JOINT_ASSET_CONSTRAINT','基金合同第十二部分投资范围存在；股票、可转债及可交换债券合计10%～30%。'),
'166009_208469.pdf':(2,'JOINT_ASSET_CONSTRAINT_WITH_HISTORY','股票、权证等权益类资产60%～95%；同页区分2011成立与2015-07-17转混合。不能将历史股票分类写回当前。'),
'850006_678107.pdf':(2,'CLAUSE_PRESENT_IMPLEMENTATION_GAP','股票（含存托凭证）60%-95%，港股占股票0%-50%；概要投资范围跨页。'),
'850006_678069.pdf':(50,'CLAUSE_PRESENT_IMPLEMENTATION_GAP','投资限制明确股票（含存托凭证）60%-95%；不是法律文本缺失。'),
'850006_825156.pdf':(2,'CLAUSE_PRESENT_IMPLEMENTATION_GAP','新版概要亦股票（含存托凭证）60%-95%；与678107核心比例相同，未证明全章节相同。'),
}
out=[]
for r in data:
    o={k:r[k] for k in ['checkpoint_id','family_key','stratum','population_count','source_document','known_at','report_name','report_code','cache_error','text_chars','extraction_status','parser_record']}
    if r['source_document'] in notes:
        page,category,note=notes[r['source_document']]
        hits=[x for x in json.loads(r['evidence']) if x['page']==page]
        h=next((x for x in hits if re.search(r'10%|60%|90%|10%～|10%～3',x['excerpt'])),hits[0] if hits else {})
        o.update(audit_bucket=category,pdf_page=page,evidence_excerpt=h.get('excerpt',''),finding=note,section_observation='投资章节或投资条款存在；正式cache失败标签不足以定位真实层级')
    else:
        note='本件是'+r['report_name']+'；属于事项公告，当前样本未发现投资分类条款，不能据此证明全家族官方历史无证据。'
        if r['source_document']=='519649_111656.pdf':note+=' PDF第2页审议事项为终止基金合同；第1页称第一次提示，metadata称第二次，需核对文档身份。'
        o.update(audit_bucket='NON_CLASSIFICATION_NOTICE',pdf_page=1,evidence_excerpt=r['report_name'],finding=note,section_observation='标题/正文事项非投资章节；evidence_excerpt为metadata标题，非PDF原文')
    o['disposition']='UNRESOLVED'
    o['noop_audit']='未作NOOP裁决；核心比例相同不足以证明同一完整章节与合法predecessor相同'
    out.append(o)
with (H/'section_uncomparable_sample.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print('wrote',len(out),'rows')
