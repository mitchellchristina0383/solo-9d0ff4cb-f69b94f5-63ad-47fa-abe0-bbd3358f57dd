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
7. **早停（可选）**：创建作业时传 `patience = N` 即启用。验证端对每个已登记检查点
   调用 `POST /jobs/{id}/losses`，按**检查点步号 + 登记轮次**上报同一验证集的损失：
   - 同一 `(step, checkpoint_round)` 同值重报只算一次（幂等），改报不同值 409 拒绝；
   - 判定从**最早仍可回退的检查点**开始按步推进，**遇到缺报就等**（更晚到达的
     上报先存着，缺口补齐后自动续判）；
   - 损失严格刷新最低值则计数清零，否则计数 +1；**连续 N 个已判定检查点未刷新
     最低损失**即触发早停：本次上报、**自动回退到损失最低的检查点**（并列取最早）、
     **停训**三者在同一事务生效，任一步失败整体回滚，调用方可原样重报；
   - 手动回退后，仍可用检查点的损失保留并继续参与判定；已作废检查点的迟到上报
     一律 409 拒收——即使同步号后来在新轮次重新登记（上报携带的登记轮次对不上
     现存检查点）；
   - 停训后预取、样本确认、检查点登记、手动回退一律 409；重启后停训状态与
     选中的检查点仍可通过 `GET /jobs/{id}` / `GET /jobs/{id}/early_stop` 查询。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/jobs` | 登记作业（样本顺序、批量、进程数，可选 `patience` 启用早停） |
| GET | `/jobs/{job_id}` | 查询作业当前状态（重启后续跑的入口；含停训状态与选中检查点） |
| GET | `/jobs/{job_id}/steps` | 当前轮次各步确认收齐情况 |
| GET | `/jobs/{job_id}/prefetch?step&process_rank&round_seq` | 按步预取本进程分片 |
| POST | `/jobs/{job_id}/ack?round_seq` | 确认已读样本（幂等去重） |
| POST | `/jobs/{job_id}/checkpoints?round_seq` | 登记已落盘检查点（门控） |
| GET | `/jobs/{job_id}/checkpoints` | 已登记检查点列表（回退可选值） |
| POST | `/jobs/{job_id}/rollback?round_seq` | 回退到检查点，可更改进程数 |
| GET | `/jobs/{job_id}/rollbacks` | 回退历史（`kind` 区分 manual / early_stop） |
| POST | `/jobs/{job_id}/losses` | 上报某检查点在同一验证集上的损失（按步号+登记轮次，可触发早停） |
| GET | `/jobs/{job_id}/losses` | 已登记的验证损失列表 |
| GET | `/jobs/{job_id}/early_stop` | 早停判定状态（最低损失、连续未刷新次数、缺报等待、停训选中检查点） |
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

## 在 /docs 演示：乱序上报触发早停

启动服务（`docker compose up --build -d`）后打开 <http://localhost:8000/docs>，
按下面顺序操作（与 `docs/early_stop_demo.sh` 逐步对应，可直接在 Swagger UI
里粘贴执行）：

1. **`POST /jobs`** 创建作业并启用早停（20 样本 / 批量 4 / 2 进程 / 共 5 步）：
   ```json
   {"job_id": "es-demo-job",
    "sample_ids": ["s00","s01","s02","s03","s04","s05","s06","s07","s08","s09",
                   "s10","s11","s12","s13","s14","s15","s16","s17","s18","s19"],
    "batch_size": 4, "num_processes": 2, "patience": 2}
   ```
2. **跑完 5 步并登记检查点**：每步对 rank 0/1 各做一次
   `GET /jobs/es-demo-job/prefetch?step=<s>&process_rank=<r>&round_seq=1`，
   把返回的 `assigned_sample_ids` 通过 `POST /jobs/es-demo-job/ack?round_seq=1`
   确认，然后 `POST /jobs/es-demo-job/checkpoints?round_seq=1`
   登记 `{"step": <s>, "checkpoint_uri": "ckpt://es-demo-job/step-<s>"}`。
3. **`POST /jobs/es-demo-job/losses`** 上报 step1 损失：
   `{"step": 1, "checkpoint_round": 1, "loss": 1.0}`
   → 响应 `judgment` 显示 `judged_through_step=1`，最低损失 `1.0@step1`。
4. **乱序上报**（后两步的检查点结果先返回）：
   `{"step": 3, "checkpoint_round": 1, "loss": 0.9}`、
   `{"step": 4, "checkpoint_round": 1, "loss": 0.85}`
   → 两次响应都是 `waiting_for_step=2`、`triggered=false`：**step2 缺报，判定等待**；
   重复发送同一请求体可以看到 `newly_recorded=false`（同值重报只算一次），
   把 `loss` 改成别的值则 409 拒绝。
5. **补齐缺口**，上报 `{"step": 2, "checkpoint_round": 1, "loss": 0.8}`
   → 判定连续推进：step2（0.8 刷新最低）→ step3（未刷新，计 1）→ step4
   （未刷新，计 2 = patience），**触发早停**。响应中 `early_stop` 与本次上报
   同事务生效：自动回退到损失最低的 **step2**、`round_seq` 推进到 2、
   `status=stopped`。
6. **核对终态**：`GET /jobs/es-demo-job` 显示 `status=stopped`、
   `stop_checkpoint_step=2`；`GET /jobs/es-demo-job/checkpoints` 只剩 step1/2
   （step3..5 已作废）；`GET /jobs/es-demo-job/losses` 只剩 step1/2 的损失；
   `GET /jobs/es-demo-job/rollbacks` 有一条 `kind=early_stop` 的记录。
7. **停训后的拒绝**：再试预取 / 确认 / 登记检查点 / 手动回退（round_seq=2）
   均 409；补报已作废的 step3（`checkpoint_round=1`）损失也 409。
8. **重启验证**：`docker compose restart` 后重新 `GET /jobs/es-demo-job` 与
   `GET /jobs/es-demo-job/early_stop`，停训状态与选中的 step2 仍然保留。

命令行等价演示（服务已启动时直接跑）：

```bash
bash docs/early_stop_demo.sh           # 默认 http://127.0.0.1:8000
JOB=es-demo-job-2 bash docs/early_stop_demo.sh   # 换个作业名重跑
```

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
PYTHONPATH=. pytest -q          # 23 个端到端用例（含 10 个早停用例）
```

## 存储结构（SQLite）

- `jobs`：样本顺序（JSON）、批量、进程数、当前检查点步 `current_step`、轮次
  `round_seq`、`status`（running/stopped）、早停耐心值 `patience`、
  停训选中的检查点 `stop_checkpoint_step`、停训时间 `stopped_at`
- `confirmations`：按 `(作业, 轮次, 步, 样本索引)` 去重的确认记录
- `checkpoints`：已登记检查点（步号、URI、登记时轮次）
- `rollback_events`：回退审计记录（原步、目标检查点、新旧进程数、轮次、
  `kind` = manual / early_stop）
- `validation_reports`：验证损失记录，按 `(作业, 检查点步号, 检查点登记轮次)`
  去重；检查点被回退作废时其损失记录一并删除

老版本创建的库文件启动时会自动 `ALTER TABLE` 补齐新增列，无需手工迁移。

## 说明与边界

- 单实例 + 进程内写锁 + SQLite WAL，适合训练集群内单协调者的规模；
  多副本部署需换用带事务的外部数据库。
- 样本 ID 为字符串，路径/数字 key 均可；创建作业时不允许重复样本。
- 预取不做“占用”登记：分派是确定性函数（步号 + rank + 进程数），
  进度以确认为准，因此进程重启后重新预取结果一致，无需会话状态。
- 早停判定完全由 `validation_reports` 表现存记录推导（不存中间计数器），
  重启、回退后重算结果一致；损失按 IEEE754 双精度比较，同值重报的“同值”
  指 JSON 数值解析后相等。

## 本次交付说明（早停功能）

### 改动清单

- `app/schemas.py`
  - `CreateJobRequest` 新增可选 `patience`（≥1，不传则不启用早停）；
  - `JobOut` 新增 `patience / stop_checkpoint_step / stopped_at`，`status`
    增加 stopped 语义；
  - 新增 `LossReportRequest / LossReportOut / EarlyStopJudgment /
    EarlyStopTriggerOut / LossRecordOut / EarlyStopStatusOut`。
- `app/db.py`
  - `jobs` 表新增 `patience / stop_checkpoint_step / stopped_at` 列，
    `rollback_events` 新增 `kind` 列（manual / early_stop），
    新增 `validation_reports` 表（主键 `(job_id, step, round_seq)`）；
    老库文件启动时自动 `ALTER TABLE` 迁移；
  - `report_loss()`：同值重报幂等、改值 409、作废/未登记检查点 409、
    停训后新上报 409；新上报后从最早仍可回退的检查点按步判定，缺报即等；
    连续 N 次未刷新最低损失时，**上报 + 自动回退（`_apply_rollback`，
    与手动回退共用）+ 停训在同一事务提交**，异常整体回滚；
  - `_judge_early_stop()`：纯查询推导判定状态（最低损失并列取最早）；
  - 手动 `rollback()` 保留仍可用检查点的损失、删除作废检查点的损失，
    并拒绝停训后的调用；`prefetch / record_ack / register_checkpoint`
    统一加停训守卫；
  - 新增 `list_losses() / early_stop_status()` 查询。
- `app/main.py`
  - `POST /jobs` 透传 `patience`；新增 `POST/GET /jobs/{id}/losses`、
    `GET /jobs/{id}/early_stop`；版本升至 1.1.0。
- `tests/test_early_stop.py`（新增 10 个用例）、`docs/early_stop_demo.sh`
  （新增）、`README.md`（本文件）。原有接口行为不变（`patience` 缺省不启用）。

### 自测结果

- `PYTHONPATH=. pytest -q`：**23 passed**（原 13 个回归用例 + 新 10 个早停用例）。
  新用例覆盖：乱序上报触发早停（含原子终态核对）、缺报等待、同值重报幂等 /
  改值拒绝 / NaN 拒绝、并列取最早、手动回退后保留可用损失并拒收旧轮次迟到
  上报（同步号新轮次重新登记后旧轮次仍拒收）、停训后四类写操作 409、
  触发事务中途失败（mock 存储故障）后原样重报成功、重启后停训状态与选中
  检查点可查。
- 本地 uvicorn 实跑 `docs/early_stop_demo.sh`：乱序上报（3、4 先于 2）→
  判定等待 → 补齐 step2 触发早停 → 自动回退 step2 + 停训同事务生效；
  随后 kill 进程重启，停训状态、`stop_checkpoint_step=2`、写操作拒绝均保持。
- 原 `docs/smoke_test.sh` 回归通过；`/docs`（Swagger UI）与 `/openapi.json`
  正常渲染，新增接口带样例。

### 未完成 / 已知限制

- 本环境没有 Docker，`docker compose up --build -d` 未在本机实际执行；
  Dockerfile / docker-compose.yml 未改动（服务形态与启动命令不变），
  上述演示流程已用等价的本机 uvicorn 验证，Compose 启动后行为一致。
- 早停触发后不停发“通知”，训练端/验证端通过上报响应与
  `GET /jobs/{id}` 轮询感知停训；如需 webhook 可后续扩展。
- 损失比较为严格小于刷新最低值；未支持“最小改善幅度（min_delta）”参数。
