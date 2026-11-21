"""FastAPI 样本分派服务。

进程视角的典型流程：
1. ``POST /jobs`` 登记作业（样本全局顺序、每步总批量、进程数，可选早停耐心值）；
2. 每个进程在每一步调用 ``GET /jobs/{job_id}/prefetch`` 预取自己的分片；
3. 读完后调用 ``POST /jobs/{job_id}/ack`` 确认，重复上报只记一次；
4. 训练端落盘检查点后调用 ``POST /jobs/{job_id}/checkpoints``，
   仅当“上次保存后至末步”各步全部收齐确认时才登记；
5. 验证端对每个已登记检查点调用 ``POST /jobs/{job_id}/losses`` 上报同一验证集
   的损失；连续 N 个已判定检查点未刷新最低损失即触发早停：上报、自动回退到
   损失最低的检查点、停训在同一事务生效；
6. 需要回退时 ``POST /jobs/{job_id}/rollback``，可换进程数，
   样本与每步顺序不变，开启新一轮确认；
7. 服务重启后从 SQLite 恢复，仍可从当前检查点续跑；已停训作业保持停训状态。
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from . import schemas
from .db import ConflictError, Database, NotFoundError, StoreError

DB_PATH = os.environ.get("DISPATCHER_DB", "/data/dispatcher.db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    app.state.db = Database(DB_PATH)
    yield
    app.state.db.close()


app = FastAPI(
    title="训练样本分派 / 检查点协调服务",
    version="1.1.0",
    description=(
        "按步为多进程训练分派样本，用 SQLite 持久化确认进度与检查点，"
        "支持回退到任意已登记检查点并调整进程数；创建作业时可指定早停耐心值，"
        "验证端按检查点上报验证损失，连续未刷新最低损失即自动回退到最优检查点并停训。"
    ),
    lifespan=lifespan,
)


def get_db() -> Database:
    return app.state.db


@app.exception_handler(NotFoundError)
async def not_found_handler(_request, exc: NotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ConflictError)
async def conflict_handler(_request, exc: ConflictError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(StoreError)
async def store_error_handler(_request, exc: StoreError):
    # 其余业务错误（如样本不属于该步/该进程）属于请求内容问题
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# --------------------------------------------------------------------- jobs
@app.post("/jobs", response_model=schemas.JobOut, status_code=201,
          tags=["作业"], summary="登记训练作业（可指定早停耐心值）")
def create_job(req: schemas.CreateJobRequest):
    job = get_db().create_job(
        req.job_id, req.sample_ids, req.batch_size, req.num_processes, req.patience
    )
    return job


@app.get("/jobs/{job_id}", response_model=schemas.JobOut,
         tags=["作业"], summary="查询作业状态（重启后可据此续跑）")
def get_job(job_id: str):
    return get_db().get_job(job_id)


@app.get("/jobs/{job_id}/steps", response_model=List[schemas.StepStatusOut],
         tags=["作业"], summary="查看当前轮次各步的确认收齐情况")
def list_step_status(job_id: str):
    db = get_db()
    job = db.get_job(job_id)
    return [db.step_status(job_id, s) for s in range(1, job["total_steps"] + 1)]


# ----------------------------------------------------------------- prefetch
@app.get(
    "/jobs/{job_id}/prefetch",
    response_model=schemas.PrefetchOut,
    tags=["样本分派"],
    summary="进程按步预取自己的样本分片",
)
def prefetch(
    job_id: str,
    step: int = Query(..., ge=1, description="步号，从 1 开始",
                      json_schema_extra={"example": 1}),
    process_rank: int = Query(..., ge=0, description="进程序号，从 0 开始",
                              json_schema_extra={"example": 0}),
    round_seq: int = Query(
        ...,
        ge=1,
        description="进程当前持有的轮次号；与服务端不一致会在确认/上报时被拒绝",
        json_schema_extra={"example": 1},
    ),
):
    db = get_db()
    job = db.get_job(job_id)
    if round_seq != job["round_seq"]:
        raise HTTPException(
            status_code=409,
            detail=(
                f"轮次已过期：请求 round={round_seq}，服务端当前 round={job['round_seq']}，"
                "请先 GET /jobs/{job_id} 获取最新状态后再预取"
            ),
        )
    return db.prefetch(job_id, step, process_rank)


# ---------------------------------------------------------------------- ack
@app.post("/jobs/{job_id}/ack", response_model=schemas.AckOut,
          tags=["确认"], summary="进程确认本步已读样本（重复确认只算一次）")
def acknowledge(
    job_id: str,
    req: schemas.AckRequest,
    round_seq: int = Query(..., ge=1, description="进程当前轮次号；回退后旧轮次会被拒绝",
                           json_schema_extra={"example": 1}),
):
    db = get_db()
    result = db.record_ack(
        job_id, round_seq, req.step, req.process_rank, req.sample_ids
    )
    job = db.get_job(job_id)
    return {
        "job_id": job_id,
        "step": req.step,
        "round_seq": job["round_seq"],
        "accepted": True,
        "newly_recorded": result["newly_recorded"],
        "confirmed_sample_count": result["confirmed"],
        "step_total": result["total"],
        "step_complete": result["complete"],
    }


# --------------------------------------------------------------- checkpoint
@app.post(
    "/jobs/{job_id}/checkpoints",
    response_model=schemas.CheckpointOut,
    tags=["检查点"],
    summary="登记训练端已落盘的检查点（各步收齐确认后才受理）",
)
def register_checkpoint(
    job_id: str,
    req: schemas.CheckpointRequest,
    round_seq: int = Query(..., ge=1, description="进程当前轮次号",
                           json_schema_extra={"example": 1}),
):
    return get_db().register_checkpoint(
        job_id, round_seq, req.step, req.checkpoint_uri
    )


@app.get("/jobs/{job_id}/checkpoints",
         response_model=List[schemas.CheckpointOut],
         tags=["检查点"], summary="列出全部已登记检查点（回退时可选）")
def list_checkpoints(job_id: str):
    return get_db().list_checkpoints(job_id)


# ----------------------------------------------------------------- rollback
@app.post("/jobs/{job_id}/rollback",
          response_model=schemas.RollbackOut,
          tags=["回退"],
          summary="回退到已有检查点，可同时更改进程数并开启新一轮确认")
def rollback(
    job_id: str,
    req: schemas.RollbackRequest,
    round_seq: int = Query(..., ge=1, description="进程当前轮次号",
                           json_schema_extra={"example": 1}),
):
    return get_db().rollback(
        job_id, round_seq, req.checkpoint_step, req.num_processes
    )


@app.get("/jobs/{job_id}/rollbacks",
         tags=["回退"], summary="查看回退历史")
def list_rollbacks(job_id: str):
    return get_db().list_rollbacks(job_id)


# -------------------------------------------------------------- early stop
@app.post(
    "/jobs/{job_id}/losses",
    response_model=schemas.LossReportOut,
    tags=["早停"],
    summary="验证端上报某检查点在同一验证集上的损失（按步号+登记轮次定位）",
)
def report_loss(job_id: str, req: schemas.LossReportRequest):
    return get_db().report_loss(job_id, req.step, req.checkpoint_round, req.loss)


@app.get(
    "/jobs/{job_id}/losses",
    response_model=List[schemas.LossRecordOut],
    tags=["早停"],
    summary="列出已登记的验证损失（回退作废的检查点记录不保留）",
)
def list_losses(job_id: str):
    return get_db().list_losses(job_id)


@app.get(
    "/jobs/{job_id}/early_stop",
    response_model=schemas.EarlyStopStatusOut,
    tags=["早停"],
    summary="查询早停判定状态（最低损失、连续未刷新次数、停训选中的检查点）",
)
def early_stop_status(job_id: str):
    return get_db().early_stop_status(job_id)


@app.get("/health", tags=["运维"], summary="健康检查")
def health():
    return {"status": "ok"}
