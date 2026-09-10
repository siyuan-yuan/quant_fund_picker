$ErrorActionPreference = 'Stop'
$base = Split-Path $PSScriptRoot -Parent
$v2 = Join-Path $base 'v2'
$rows = @(Import-Csv (Join-Path $v2 'classification_state_timeline.csv'))
$dispositions = @{}
Import-Csv (Join-Path $v2 'document_dispositions.csv') | ForEach-Object { $dispositions[$_.checkpoint_id] = $_ }
$pdfdir = Join-Path (Split-Path $base -Parent) 'pdf'
$audit = foreach ($row in $rows) {
    $causal = $row.causal_violation -eq 'True'
    $zero = $row.effective_to -and $row.effective_to -le $row.effective_from
    if (-not ($causal -or $zero)) { continue }
    $family = @($rows | Where-Object { $_.family_key -eq $row.family_key -and $_.effective_from } | Sort-Object known_at,effective_date,checkpoint_id)
    $ids = @($family | ForEach-Object checkpoint_id)
    $i = [Array]::IndexOf($ids, $row.checkpoint_id)
    $prev = if ($i -gt 0) { $family[$i-1] } else { $null }
    $next = if ($i -lt $family.Count-1) { $family[$i+1] } else { $null }
    $same = @($family | Where-Object effective_from -eq $row.effective_from)
    $rawRanges = @($same | ForEach-Object { $d=$dispositions[$_.checkpoint_id]; "$($d.fund_type)|$($d.equity_min_pct)|$($d.equity_max_pct)" } | Select-Object -Unique)
    $text = ((& pdftotext -layout (Join-Path $pdfdir $row.source_document) -) -join "`n") -replace '\s',''
    $dateEvidence = ''
    $root = 'SAME_DAY_SAME_STATE_SERIAL_CLOSURE'
    $note = '同日同状态多个checkpoint顺次关闭，形成零长区间；应保留节点，状态区间按同时点合并。'
    if ($zero -and $rawRanges.Count -gt 1) {
        $root = 'SAME_DAY_COMPATIBLE_RANGES_SERIAL_CLOSURE'
        $note = '同日原始区间40-100与40-95兼容，timeline先收紧为40-95，再顺次关闭造成零长；不是原始证据完全相同，也不是互斥状态冲突。'
    }
    if ($causal) {
        $parts = $row.effective_date.Split('-')
        $datePattern = "$([int]$parts[0])年0?$([int]$parts[1])月0?$([int]$parts[2])日"
        $dateEvidence = ([regex]::Matches($text, ".{0,120}$datePattern.{0,260}") | Select-Object -First 4 | ForEach-Object Value) -join ' || '
        $root = 'HISTORICAL_LEGAL_DATE_FLAGGED_AS_CAUSAL'
        $note = '源PDF正文支持历史生效日期；effective_from等于known_at，无PIT起点回填。尚未独立核验更早首发公告。'
        if ($row.source_document -eq '400013_321232.pdf') {
            $root = 'DATE_EXTRACTION_CROSSES_DEFINITION_BLOCK'
            $note = '2004-06-01是证券投资基金法实施日；日期regex跨多项定义误绑定基金合同生效，非基金法律生效日。'
        }
        if ($row.source_document -eq '400013_321700.pdf') {
            $root = 'PROPOSED_STATE_BOUND_TO_PRIOR_TRANSFORMATION_DATE'
            $note = '2017-05-11为既往转型日期；本文件为修改合同议案且出现2018年XX月XX占位生效日，0-95条款疑属拟议新状态，必须分栏定位后判断，不能视为当前已生效。'
        }
        if ($row.source_document -eq '400013_321691.pdf') {
            $note += ' 本条恢复40-80，与前一2017-12-14议案的0-95不同；优先排除前一议案拟议条款误用。'
        }
    }
    [pscustomobject]@{
        family_key=$row.family_key; checkpoint_id=$row.checkpoint_id; anomaly_type=$(if($causal){'CAUSAL_FLAG'}else{'NONPOSITIVE_INTERVAL'});
        known_at=$row.known_at; effective_date=$row.effective_date; effective_from=$row.effective_from; effective_to=$row.effective_to;
        predecessor=$prev.checkpoint_id; successor=$next.checkpoint_id; predecessor_source_doc=$prev.source_document; successor_source_doc=$next.source_document;
        source_doc=$row.source_document; trigger_type=$row.trigger_type; disposition=$row.disposition; fund_type=$row.fund_type;
        equity_min_pct=$row.equity_min_pct; equity_max_pct=$row.equity_max_pct;
        same_day_checkpoint_count=$same.Count; same_day_raw_state_count=$rawRanges.Count;
        same_day_raw_states=($rawRanges -join ' || ');
        predecessor_state="$($prev.fund_type)|$($prev.equity_min_pct)|$($prev.equity_max_pct)";
        successor_state="$($next.fund_type)|$($next.equity_min_pct)|$($next.equity_max_pct)";
        actual_pit_backfill=($row.effective_from -lt $row.known_at);
        root_cause=$root; evidence_raw=$row.evidence_raw; date_evidence_from_pdf=$dateEvidence; assessment=$note;
        verification_method='local official PDF pdftotext -layout text inspection; source-page visual/legal signoff pending'
    }
}
$audit | Export-Csv (Join-Path $PSScriptRoot 'timeline_anomalies.csv') -NoTypeInformation -Encoding utf8
$audit | Group-Object root_cause | Select-Object Name,Count | Format-Table -AutoSize
if ($audit.Count -ne 21) { throw "Expected 21 anomalies, found $($audit.Count)" }
if (@($audit | Where-Object actual_pit_backfill -eq True).Count -ne 0) { throw 'Unexpected actual PIT backfill' }
$audit | Where-Object anomaly_type -eq NONPOSITIVE_INTERVAL | Group-Object same_day_raw_state_count | Select-Object Name,Count
