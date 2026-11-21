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
    status: str
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
