#!/usr/bin/env bash
# 端到端手工试跑脚本：对应 /docs (Swagger UI) 里的样例请求。
# 用法：bash docs/smoke_test.sh
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
JOB="${JOB:-smoke-job}"

jq_get() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))" "$1"; }
jq_json() { python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(eval(sys.argv[1])))" "$1"; }

echo "== 1. 创建作业：12 个样本，每步批量 4，初始 2 进程（共 3 步） =="
curl -sS -X POST "$BASE/jobs" -H 'Content-Type: application/json' -d '{
  "job_id": "'"$JOB"'",
  "sample_ids": ["s00","s01","s02","s03","s04","s05","s06",
                 "s07","s08","s09","s10","s11"],
  "batch_size": 4,
  "num_processes": 2
}' | python3 -m json.tool

echo "== 2. 两进程分别预取 step1 的分片 =="
curl -sS "$BASE/jobs/$JOB/prefetch?step=1&process_rank=0&round_seq=1" | python3 -m json.tool
curl -sS "$BASE/jobs/$JOB/prefetch?step=1&process_rank=1&round_seq=1" | python3 -m json.tool

echo "== 3. 两进程确认 step1；rank0 再重复确认一次（只算一次） =="
curl -sS -X POST "$BASE/jobs/$JOB/ack?round_seq=1" -H 'Content-Type: application/json' \
  -d '{"step":1,"process_rank":0,"sample_ids":["s00","s01"]}' | python3 -m json.tool
curl -sS -X POST "$BASE/jobs/$JOB/ack?round_seq=1" -H 'Content-Type: application/json' \
  -d '{"step":1,"process_rank":1,"sample_ids":["s02","s03"]}' | python3 -m json.tool
echo "-- 重复确认，newly_recorded 应为 0 --"
curl -sS -X POST "$BASE/jobs/$JOB/ack?round_seq=1" -H 'Content-Type: application/json' \
  -d '{"step":1,"process_rank":0,"sample_ids":["s00","s01"]}' | python3 -m json.tool

echo "== 4. 未收齐时登记检查点会被 409 拒绝（直接登记 step3） =="
curl -sS -o /tmp/resp.json -w "HTTP %{http_code}\n" -X POST \
  "$BASE/jobs/$JOB/checkpoints?round_seq=1" -H 'Content-Type: application/json' \
  -d '{"step":3,"checkpoint_uri":"ckpt://too-early"}'; cat /tmp/resp.json; echo

echo "== 5. 登记 step1 检查点 =="
curl -sS -X POST "$BASE/jobs/$JOB/checkpoints?round_seq=1" \
  -H 'Content-Type: application/json' \
  -d '{"step":1,"checkpoint_uri":"ckpt://'"$JOB"'/step-1"}' | python3 -m json.tool

echo "== 6. 回退到 step0（从头）并把进程数改成 3，进入 round 2 =="
curl -sS -X POST "$BASE/jobs/$JOB/rollback?round_seq=1" \
  -H 'Content-Type: application/json' \
  -d '{"checkpoint_step":0,"num_processes":3}' | python3 -m json.tool

echo "== 7. 旧轮次(round=1)确认被 409 拒绝 =="
curl -sS -o /tmp/resp.json -w "HTTP %{http_code}\n" -X POST \
  "$BASE/jobs/$JOB/ack?round_seq=1" -H 'Content-Type: application/json' \
  -d '{"step":1,"process_rank":0,"sample_ids":["s00"]}'; cat /tmp/resp.json; echo

echo "== 8. 新轮次 3 进程预取 step1：4 个样本切成 2/1/1，顺序不变 =="
for r in 0 1 2; do
  curl -sS "$BASE/jobs/$JOB/prefetch?step=1&process_rank=$r&round_seq=2" \
    | jq_get "f'rank={d[\"process_rank\"]} assigned={d[\"assigned_sample_ids\"]}'"
done

echo "== 9. 新轮次跑完全部 3 步（每步 3 进程预取+确认），登记 step3 检查点 =="
for step in 1 2 3; do
  for rank in 0 1 2; do
    samples=$(curl -sS "$BASE/jobs/$JOB/prefetch?step=$step&process_rank=$rank&round_seq=2" \
      | jq_json "d['assigned_sample_ids']")
    # 末步/空分片时允许空数组：空分片无需确认
    if [ "$samples" != "[]" ]; then
      code=$(curl -sS -o /tmp/ack.json -w "%{http_code}" -X POST \
        "$BASE/jobs/$JOB/ack?round_seq=2" -H 'Content-Type: application/json' \
        -d "{\"step\":$step,\"process_rank\":$rank,\"sample_ids\":$samples}")
      [ "$code" = "200" ] || { echo "ack failed: $(cat /tmp/ack.json)"; exit 1; }
    fi
  done
done
curl -sS -X POST "$BASE/jobs/$JOB/checkpoints?round_seq=2" \
  -H 'Content-Type: application/json' \
  -d '{"step":3,"checkpoint_uri":"ckpt://'"$JOB"'/step-3-r2"}' | python3 -m json.tool

echo "== 10. 查看作业状态、检查点列表、回退历史 =="
curl -sS "$BASE/jobs/$JOB" | python3 -m json.tool
curl -sS "$BASE/jobs/$JOB/checkpoints" | python3 -m json.tool
curl -sS "$BASE/jobs/$JOB/rollbacks" | python3 -m json.tool

echo
echo "全部接口试跑完成。接下来可重启服务验证持久化（见 README）。"
