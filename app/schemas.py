"""Pydantic 请求/响应模型。

样本 ID 使用字符串表达（路径、整数 ID 或自定义 key 均可），
因此整套 API 对样本类型没有假设。
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class CreateJobRequest(BaseModel):
    job_id: str = Field(
        ...,
        min_length=1,
        description="训练作业 ID，自定义且需全局唯一",
        json_schema_extra={"example": "job-demo-001"},
    )
    sample_ids: List[str] = Field(
        ...,
        min_length=1,
        description="全局样本顺序，索引即样本序号；回退后该顺序保持不变",
        json_schema_extra={"example": ["s00", "s01", "s02", "s03", "s04", "s05",
                                       "s06", "s07", "s08", "s09", "s10", "s11"]},
    )
    batch_size: int = Field(
        ...,
        ge=1,
        description="每步总批量（一步包含多少个样本）",
        json_schema_extra={"example": 4},
    )
    num_processes: int = Field(
        ...,
        ge=1,
        description="初始训练进程数",
        json_schema_extra={"example": 2},
    )
    patience: Optional[int] = Field(
        None,
        ge=1,
        description="早停耐心值 N：连续 N 个已判定检查点未刷新最低验证损失即自动回退并停训；"
                    "不传则不启用早停",
        json_schema_extra={"example": 2},
    )

    @field_validator("sample_ids")
    @classmethod
    def _unique_samples(cls, v: List[str]) -> List[str]:
        if len(set(v)) != len(v):
            raise ValueError("sample_ids 中存在重复样本")
        return v


class JobOut(BaseModel):
    job_id: str
    sample_count: int
    total_steps: int
    batch_size: int
    num_processes: int
    current_step: int = Field(description="当前检查点对应的步号；0 表示尚无检查点")
    round_seq: int = Field(description="当前确认轮次；每次回退 +1，重启不变")
    status: str = Field(description="running=训练中；stopped=已早停停训")
    patience: Optional[int] = Field(description="早停耐心值；null 表示未启用早停")
    stop_checkpoint_step: Optional[int] = Field(
        description="早停选中的检查点步号（损失最低、并列取最早）；未停训为 null"
    )
    stopped_at: Optional[str] = Field(description="停训时间；未停训为 null")
    created_at: str
    updated_at: str


class PrefetchOut(BaseModel):
    job_id: str
    step: int
    process_rank: int
    num_processes: int
    round_seq: int
    assigned_sample_ids: List[str] = Field(description="该进程在本步应读的样本（保持全局顺序）")
    assigned_indexes: List[int] = Field(description="对应样本在全局顺序中的索引")
    all_sample_ids: List[str] = Field(description="本步全量样本（按全局顺序），便于训练端核对")


class AckRequest(BaseModel):
    step: int = Field(..., ge=1, json_schema_extra={"example": 1})
    process_rank: int = Field(..., ge=0, json_schema_extra={"example": 0})
    sample_ids: List[str] = Field(
        ...,
        min_length=1,
        description="该进程本步已读样本，重复上报只记一次",
        json_schema_extra={"example": ["s00", "s01"]},
    )


class AckOut(BaseModel):
    job_id: str
    step: int
    round_seq: int
    accepted: bool = Field(description="是否属于当前轮次（旧轮次的上报会被拒绝）")
    newly_recorded: int = Field(description="本次新登记的确认数；重复确认为 0")
    confirmed_sample_count: int = Field(description="本步当前已收齐的样本确认数")
    step_total: int = Field(description="本步应确认的样本总数")
    step_complete: bool = Field(description="本步样本是否已全部确认")


class CheckpointRequest(BaseModel):
    step: int = Field(..., ge=1, description="检查点覆盖的末步，须大于当前检查点步号",
                      json_schema_extra={"example": 2})
    checkpoint_uri: str = Field(
        ...,
        min_length=1,
        description="训练端已落盘的检查点标识（路径/URI/版本名均可）",
        json_schema_extra={"example": "s3://ckpt/job-demo-001/step-2"},
    )


class CheckpointOut(BaseModel):
    job_id: str
    step: int
    checkpoint_uri: str
    round_seq: int
    registered_at: str


class RollbackRequest(BaseModel):
    checkpoint_step: int = Field(
        ...,
        ge=0,
        description="回退到的已有检查点步号；0 表示从头开始",
        json_schema_extra={"example": 1},
    )
    num_processes: Optional[int] = Field(
        None,
        ge=1,
        description="新的进程数；不传则保持不变。样本全局顺序与每步划分均不变，仅重新切分",
        json_schema_extra={"example": 3},
    )


class RollbackOut(BaseModel):
    job_id: str
    checkpoint_step: int
    num_processes: int
    round_seq: int = Field(description="回退后的新一轮次（旧轮次的确认/上报将被拒绝）")
    next_step: int
    rolled_back_at: str


class StepStatusOut(BaseModel):
    step: int
    round_seq: int
    total: int
    confirmed: int
    missing_sample_ids: List[str]
    complete: bool


# ------------------------------------------------------------------ early stop
class LossReportRequest(BaseModel):
    step: int = Field(
        ...,
        ge=1,
        description="被验证的检查点步号（检查点须已登记）",
        json_schema_extra={"example": 2},
    )
    checkpoint_round: int = Field(
        ...,
        ge=1,
        description="该检查点登记时的轮次；用于区分回退后同步号重新登记的新检查点，"
                    "已作废检查点的迟到结果会被拒收",
        json_schema_extra={"example": 1},
    )
    loss: float = Field(
        ...,
        description="同一验证集上的损失（须为有限数值）；同一 (step, checkpoint_round) "
                    "同值重报只算一次，改报不同值会被拒绝",
        json_schema_extra={"example": 0.42},
    )


class EarlyStopJudgment(BaseModel):
    patience: int
    judged_through_step: int = Field(description="已连续判定到的检查点步号；尚无判定为 0")
    waiting_for_step: Optional[int] = Field(
        description="第一个缺报损失的检查点步号（判定在此等待）；无缺报为 null"
    )
    best_loss: Optional[float] = Field(description="已判定检查点中的最低损失")
    best_checkpoint_step: Optional[int] = Field(
        description="最低损失对应的检查点步号（并列取最早），即触发时的回退目标"
    )
    no_improve_count: int = Field(description="当前连续未刷新最低损失的次数")
    triggered: bool = Field(description="本次判定是否触发早停")


class EarlyStopTriggerOut(BaseModel):
    """触发早停时与上报同一事务生效的自动回退 + 停训结果。"""
    status: str = Field(description="停训后为 stopped")
    checkpoint_step: int = Field(description="自动回退到的检查点步号（损失最低、并列取最早）")
    num_processes: int = Field(description="自动回退不改变进程数")
    round_seq: int = Field(description="回退后的新轮次")
    next_step: int
    rolled_back_at: str


class LossReportOut(BaseModel):
    job_id: str
    step: int
    checkpoint_round: int
    loss: float
    accepted: bool = Field(description="是否属于当前有效检查点（作废检查点的迟到上报会被拒绝）")
    newly_recorded: bool = Field(description="本次是否新登记；同值重报为 false（只算一次）")
    reported_at: str
    job_status: str = Field(description="上报生效后的作业状态")
    judgment: EarlyStopJudgment = Field(description="本次上报后的早停判定状态")
    early_stop: Optional[EarlyStopTriggerOut] = Field(
        None, description="仅当本次上报触发早停时返回：自动回退与停训已一并生效"
    )


class LossRecordOut(BaseModel):
    job_id: str
    step: int
    round_seq: int = Field(description="检查点的登记轮次")
    loss: float
    reported_at: str


class EarlyStopStatusOut(BaseModel):
    job_id: str
    enabled: bool = Field(description="创建作业时是否指定了 patience")
    status: str
    patience: Optional[int] = None
    judged_through_step: Optional[int] = None
    waiting_for_step: Optional[int] = None
    best_loss: Optional[float] = None
    best_checkpoint_step: Optional[int] = None
    no_improve_count: Optional[int] = None
    stop_checkpoint_step: Optional[int] = Field(
        description="早停选中的检查点步号；重启后仍可查"
    )
    stopped_at: Optional[str] = None
