#!/usr/bin/env bash
# 早停演示脚本：与 README「在 /docs 演示：乱序上报触发早停」一一对应。
# 场景：验证端乱序上报损失（step3/step4 先于 step2），判定在缺口处等待；
# step2 补齐后连续 2 个检查点未刷新最低损失，触发早停——本次上报、
# 自动回退到最优检查点、停训在同一事务生效。
# 用法：bash docs/early_stop_demo.sh   （先 docker compose up --build -d）
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
JOB="${JOB:-es-demo-job}"

hdr() { echo; echo "== $* =="; }
# 先清理同名作业（若存在）无法删除——换个 job_id 即可：JOB=es-demo-job-2 bash ...

hdr "1. 创建作业：20 样本 / 批量 4 / 2 进程 / 共 5 步，patience=2 启用早停"
curl -sS -X POST "$BASE/jobs" -H 'Content-Type: application/json' -d '{
  "job_id": "'"$JOB"'",
  "sample_ids": ["s00","s01","s02","s03","s04","s05","s06","s07","s08","s09",
                 "s10","s11","s12","s13","s14","s15","s16","s17","s18","s19"],
  "batch_size": 4,
  "num_processes": 2,
  "patience": 2
}' | python3 -m json.tool

hdr "2. 两进程跑完 5 步，每步登记检查点（round 1）"
for step in 1 2 3 4 5; do
  for rank in 0 1; do
    samples=$(curl -sS "$BASE/jobs/$JOB/prefetch?step=$step&process_rank=$rank&round_seq=1" \
      | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['assigned_sample_ids']))")
    if [ "$samples" != "[]" ]; then
      code=$(curl -sS -o /tmp/es_ack.json -w "%{http_code}" -X POST \
        "$BASE/jobs/$JOB/ack?round_seq=1" -H 'Content-Type: application/json' \
        -d "{\"step\":$step,\"process_rank\":$rank,\"sample_ids\":$samples}")
      [ "$code" = "200" ] || { echo "ack failed: $(cat /tmp/es_ack.json)"; exit 1; }
    fi
  done
  curl -sS -X POST "$BASE/jobs/$JOB/checkpoints?round_seq=1" \
    -H 'Content-Type: application/json' \
    -d "{\"step\":$step,\"checkpoint_uri\":\"ckpt://$JOB/step-$step\"}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"  已登记检查点 step={d['step']}\")"
done

hdr "3. 上报 step1 损失 1.0（判定推进到 step1，最低损失 1.0@step1）"
curl -sS -X POST "$BASE/jobs/$JOB/losses" -H 'Content-Type: application/json' \
  -d '{"step":1,"checkpoint_round":1,"loss":1.0}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin)['judgment']; print(f\"  judged_through={d['judged_through_step']} best={d['best_loss']}@step{d['best_checkpoint_step']} no_improve={d['no_improve_count']} triggered={d['triggered']}\")"

hdr "4. 乱序上报：step3=0.9、step4=0.85 先到——step2 缺报，判定等待；同值重报只算一次"
for s in 3 4; do
  loss=$([ "$s" = 3 ] && echo 0.9 || echo 0.85)
  echo "-- step$s 首次上报 --"
  curl -sS -X POST "$BASE/jobs/$JOB/losses" -H 'Content-Type: application/json' \
    -d "{\"step\":$s,\"checkpoint_round\":1,\"loss\":$loss}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); j=d['judgment']; print(f\"  newly_recorded={d['newly_recorded']} judged_through={j['judged_through_step']} waiting_for={j['waiting_for_step']} triggered={j['triggered']}\")"
  echo "-- step$s 同值重报（幂等） --"
  curl -sS -X POST "$BASE/jobs/$JOB/losses" -H 'Content-Type: application/json' \
    -d "{\"step\":$s,\"checkpoint_round\":1,\"loss\":$loss}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); j=d['judgment']; print(f\"  newly_recorded={d['newly_recorded']} judged_through={j['judged_through_step']} waiting_for={j['waiting_for_step']} triggered={j['triggered']}\")"
done

hdr "5. 补齐 step2=0.8：连续判定 2(刷新)->3(未刷新 1)->4(未刷新 2)，触发早停"
curl -sS -X POST "$BASE/jobs/$JOB/losses" -H 'Content-Type: application/json' \
  -d '{"step":2,"checkpoint_round":1,"loss":0.8}' | python3 -m json.tool

hdr "6. 终态：status=stopped，自动回退到损失最低的 step2，轮次推进到 2"
curl -sS "$BASE/jobs/$JOB" | python3 -m json.tool
echo "-- 早停状态 --"
curl -sS "$BASE/jobs/$JOB/early_stop" | python3 -m json.tool
echo "-- 仍可用检查点（step3..5 已作废） --"
curl -sS "$BASE/jobs/$JOB/checkpoints" | python3 -m json.tool
echo "-- 保留的损失记录（step3/step4 的记录随检查点一并作废） --"
curl -sS "$BASE/jobs/$JOB/losses" | python3 -m json.tool
echo "-- 回退历史（kind=early_stop 为自动回退） --"
curl -sS "$BASE/jobs/$JOB/rollbacks" | python3 -m json.tool

hdr "7. 停训后写操作一律 409：预取 / 确认 / 检查点登记 / 手动回退"
echo "-- 预取 --"
curl -sS -o /tmp/es_resp.json -w "HTTP %{http_code}\n" \
  "$BASE/jobs/$JOB/prefetch?step=3&process_rank=0&round_seq=2"; cat /tmp/es_resp.json; echo
echo "-- 样本确认 --"
curl -sS -o /tmp/es_resp.json -w "HTTP %{http_code}\n" -X POST \
  "$BASE/jobs/$JOB/ack?round_seq=2" -H 'Content-Type: application/json' \
  -d '{"step":3,"process_rank":0,"sample_ids":["s08"]}'; cat /tmp/es_resp.json; echo
echo "-- 检查点登记 --"
curl -sS -o /tmp/es_resp.json -w "HTTP %{http_code}\n" -X POST \
  "$BASE/jobs/$JOB/checkpoints?round_seq=2" -H 'Content-Type: application/json' \
  -d '{"step":3,"checkpoint_uri":"ckpt://late"}'; cat /tmp/es_resp.json; echo
echo "-- 手动回退 --"
curl -sS -o /tmp/es_resp.json -w "HTTP %{http_code}\n" -X POST \
  "$BASE/jobs/$JOB/rollback?round_seq=2" -H 'Content-Type: application/json' \
  -d '{"checkpoint_step":1}'; cat /tmp/es_resp.json; echo

hdr "8. 已作废检查点（step3, round1）的迟到上报：拒收"
curl -sS -o /tmp/es_resp.json -w "HTTP %{http_code}\n" -X POST \
  "$BASE/jobs/$JOB/losses" -H 'Content-Type: application/json' \
  -d '{"step":3,"checkpoint_round":1,"loss":0.9}'; cat /tmp/es_resp.json; echo

echo
echo "演示完成。可执行 docker compose restart 后重新查询 "
echo "  GET /jobs/$JOB 与 GET /jobs/$JOB/early_stop"
echo "停训状态与选中的检查点（step2）仍然保留。"
