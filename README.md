# 训练样本分派 / 检查点协调服务

多进程训练场景下的样本分派与检查点协调 API。训练进程会**提前预取**样本，
回退后需要从检查点位置继续取数；本服务用 FastAPI 分派样本、SQLite 持久化进度，
保证回退、重启后状态一致。

## 核心规则

1. **作业登记**：创建作业时固定样本全局顺序（数组索引即样本序号）、每步总批量
   `batch_size` 和进程数 `num_processes`。第 *n* 步取
   `sample_ids[(n-1)*batch_size : n*batch_size]`，末步允许不满批。
2. **按步预取**：进程按 `(step, process_rank)` 预取自己的分片。分片为连续区间、
   保持全局顺序，余数样本分给靠前的进程；末步样本数少于进程数时，靠后的进程取到空分片。
3. **确认去重**：进程读完后上报确认，SQLite 以
   `(job_id, round_seq, step, sample_index)` 为主键，重复确认只算一次；
   上报样本必须属于该步且属于该进程的分片，否则 400 拒绝。
4. **检查点门控**：只有“上次保存检查点的下一步 → 本次上报末步”之间**每一步都收齐
   全部样本确认**，才登记训练端上报的已落盘检查点，并把 `current_step` 推进到该步。
5. **回退**：只能回退到**已登记的检查点**（或步 0 从头开始），可同时修改进程数。
   样本全局顺序与每步划分不变，仅按新进程数重新切分分片；回退后 `round_seq` 加 1，
   旧轮次的确认被清空，**晚于回退目标的检查点一并作废**（目标步及更早的保留），
   旧轮次的预取/确认/检查点上报/回退请求一律 409 拒绝。
6. **重启续跑**：全部状态在 SQLite 单文件（WAL 模式）中，服务重启后作业、检查点、
   轮次完整恢复，进程先 `GET /jobs/{id}` 拿到当前 `current_step / round_seq /
   num_processes`，从 `current_step + 1` 继续预取即可。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/jobs` | 登记作业（样本顺序、批量、进程数） |
| GET | `/jobs/{job_id}` | 查询作业当前状态（重启后续跑的入口） |
| GET | `/jobs/{job_id}/steps` | 当前轮次各步确认收齐情况 |
| GET | `/jobs/{job_id}/prefetch?step&process_rank&round_seq` | 按步预取本进程分片 |
| POST | `/jobs/{job_id}/ack?round_seq` | 确认已读样本（幂等去重） |
| POST | `/jobs/{job_id}/checkpoints?round_seq` | 登记已落盘检查点（门控） |
| GET | `/jobs/{job_id}/checkpoints` | 已登记检查点列表（回退可选值） |
| POST | `/jobs/{job_id}/rollback?round_seq` | 回退到检查点，可更改进程数 |
| GET | `/jobs/{job_id}/rollbacks` | 回退历史 |
| GET | `/health` | 健康检查 |

所有写接口都带 `round_seq` 查询参数：进程须持有服务端当前轮次号，
回退生效后旧轮次请求会被拒绝，防止“回退后旧进程的迟到上报”污染新进度。

## 用 Docker Compose 启动

```bash
docker compose up --build -d
# 数据保存在命名卷 dispatcher-data 的 /data/dispatcher.db
docker compose logs -f
docker compose down          # 停止；加 -v 会同时删除数据卷
```

启动后打开交互文档：<http://localhost:8000/docs> （Swagger UI，请求体自带样例）。

## 在 /docs 里试跑：预取 → 回退 → 换进程数

也可以直接跑等价的命令行脚本：

```bash
bash docs/smoke_test.sh              # 默认 http://127.0.0.1:8000
JOB=my-job BASE=http://localhost:8000 bash docs/smoke_test.sh
```

脚本覆盖：创建作业（12 样本 / 批量 4 / 2 进程）→ 两进程预取并确认 step1
（含重复确认幂等演示）→ 未收齐时检查点被 409 拒绝 → 登记 step1 检查点 →
回退到步 0 并改为 **3 进程** → 旧轮次确认被 409 拒绝 → 新轮次按 2/1/1 重新分片
（样本顺序不变）→ 跑完全部步并登记检查点。

重启续跑验证（Docker 场景即 `docker compose restart`）：

```bash
# 本地 uvicorn 场景
kill $(cat /tmp/uvicorn.pid)
DISPATCHER_DB=/tmp/demo-data/dispatcher.db uvicorn app.main:app --port 8000
curl http://127.0.0.1:8000/jobs/<job_id>   # current_step / round_seq 均保留
```

## 本地开发（不用 Docker）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt httpx pytest
export DISPATCHER_DB=./data/dispatcher.db
uvicorn app.main:app --reload
PYTHONPATH=. pytest -q          # 12 个端到端用例
```

## 存储结构（SQLite）

- `jobs`：样本顺序（JSON）、批量、进程数、当前检查点步 `current_step`、轮次 `round_seq`
- `confirmations`：按 `(作业, 轮次, 步, 样本索引)` 去重的确认记录
- `checkpoints`：已登记检查点（步号、URI、登记时轮次）
- `rollback_events`：回退审计记录（原步、目标检查点、新旧进程数、轮次）

## 说明与边界

- 单实例 + 进程内写锁 + SQLite WAL，适合训练集群内单协调者的规模；
  多副本部署需换用带事务的外部数据库。
- 样本 ID 为字符串，路径/数字 key 均可；创建作业时不允许重复样本。
- 预取不做“占用”登记：分派是确定性函数（步号 + rank + 进程数），
  进度以确认为准，因此进程重启后重新预取结果一致，无需会话状态。
