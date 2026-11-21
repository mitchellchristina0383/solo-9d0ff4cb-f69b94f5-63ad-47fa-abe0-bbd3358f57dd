"""SQLite 持久化层。

设计要点：
- 单文件 SQLite，服务重启后状态完整保留，可从当前检查点续跑；
- 所有写操作在进程内锁内串行执行，SQLite 自身再用
  ``UNIQUE`` 约束兜底，保证“重复确认只算一次”；
- 确认按“轮次(round_seq)”隔离，回退开启新一轮后，
  旧轮次的确认与检查点上报一律拒绝。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import List, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StoreError(Exception):
    """存储层业务错误，message 可直接返回给调用方。"""


class NotFoundError(StoreError):
    pass


class ConflictError(StoreError):
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    sample_ids    TEXT NOT NULL,          -- JSON 数组，数组顺序即全局样本顺序
    batch_size    INTEGER NOT NULL,
    num_processes INTEGER NOT NULL,
    current_step  INTEGER NOT NULL DEFAULT 0,  -- 当前已登记检查点的末步
    round_seq     INTEGER NOT NULL DEFAULT 1,  -- 当前确认轮次
    status        TEXT NOT NULL DEFAULT 'running',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS confirmations (
    job_id       TEXT NOT NULL,
    round_seq    INTEGER NOT NULL,        -- 该确认属于哪一轮
    step         INTEGER NOT NULL,
    sample_index INTEGER NOT NULL,
    sample_id    TEXT NOT NULL,
    process_rank INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (job_id, round_seq, step, sample_index)  -- 重复确认天然去重
);

CREATE TABLE IF NOT EXISTS checkpoints (
    job_id         TEXT NOT NULL,
    step           INTEGER NOT NULL,
    checkpoint_uri TEXT NOT NULL,
    round_seq      INTEGER NOT NULL,
    registered_at  TEXT NOT NULL,
    PRIMARY KEY (job_id, step)
);

CREATE TABLE IF NOT EXISTS rollback_events (
    job_id          TEXT NOT NULL,
    from_step       INTEGER NOT NULL,
    checkpoint_step INTEGER NOT NULL,
    old_processes   INTEGER NOT NULL,
    new_processes   INTEGER NOT NULL,
    round_seq       INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: str):
        self.path = path
        # check_same_thread=False：FastAPI 线程池中使用，配合 self._lock 串行化写入
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # 纯计算工具：步 -> 样本切片；进程 -> 步内分片
    # ------------------------------------------------------------------
    @staticmethod
    def step_slice(total_samples: int, batch_size: int, step: int) -> Tuple[int, int]:
        """返回 step（1-based）对应的样本半开区间 [start, end)。末步可能不满批。"""
        start = (step - 1) * batch_size
        end = min(start + batch_size, total_samples)
        return start, end

    @staticmethod
    def total_steps(total_samples: int, batch_size: int) -> int:
        return (total_samples + batch_size - 1) // batch_size

    @staticmethod
    def rank_shard(length: int, num_processes: int, rank: int) -> Tuple[int, int]:
        """长度为 length 的步内样本，按进程数尽量均分给 rank 的半开区间。

        采用连续分片（保持全局顺序），余数样本分给前面的进程。
        """
        base, extra = divmod(length, num_processes)
        start = rank * base + min(rank, extra)
        end = start + base + (1 if rank < extra else 0)
        return start, end

    # ------------------------------------------------------------------ jobs
    def create_job(
        self,
        job_id: str,
        sample_ids: List[str],
        batch_size: int,
        num_processes: int,
    ) -> dict:
        now = utc_now_iso()
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if exists:
                raise ConflictError(f"作业 {job_id} 已存在")
            self._conn.execute(
                "INSERT INTO jobs (job_id, sample_ids, batch_size, num_processes,"
                " current_step, round_seq, status, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, 0, 1, 'running', ?, ?)",
                (job_id, json.dumps(sample_ids), batch_size, num_processes, now, now),
            )
            self._conn.commit()
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"作业 {job_id} 不存在")
        return self._job_row_to_dict(row)

    def list_checkpoints(self, job_id: str) -> List[dict]:
        self.get_job(job_id)
        rows = self._conn.execute(
            "SELECT * FROM checkpoints WHERE job_id = ? ORDER BY step", (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_rollbacks(self, job_id: str) -> List[dict]:
        self.get_job(job_id)
        rows = self._conn.execute(
            "SELECT * FROM rollback_events WHERE job_id = ? ORDER BY rowid", (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def _job_row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["sample_ids"] = json.loads(d["sample_ids"])
        d["sample_count"] = len(d["sample_ids"])
        d["total_steps"] = self.total_steps(d["sample_count"], d["batch_size"])
        return d

    # -------------------------------------------------------------- prefetch
    def prefetch(self, job_id: str, step: int, rank: int) -> dict:
        job = self.get_job(job_id)
        total_steps = job["total_steps"]
        if step <= job["current_step"]:
            raise ConflictError(
                f"step={step} 不晚于当前检查点步 {job['current_step']}，"
                "该步已落盘，无需再取"
            )
        if step > total_steps:
            raise ConflictError(f"step={step} 超出总步数 {total_steps}")
        if not (0 <= rank < job["num_processes"]):
            raise ConflictError(
                f"process_rank={rank} 越界，当前进程数 {job['num_processes']}"
            )
        start, end = self.step_slice(job["sample_count"], job["batch_size"], step)
        step_samples = job["sample_ids"][start:end]
        shard_start, shard_end = self.rank_shard(
            len(step_samples), job["num_processes"], rank
        )
        return {
            "job_id": job_id,
            "step": step,
            "process_rank": rank,
            "num_processes": job["num_processes"],
            "round_seq": job["round_seq"],
            "assigned_sample_ids": step_samples[shard_start:shard_end],
            "assigned_indexes": list(
                range(start + shard_start, start + shard_end)
            ),
            "all_sample_ids": step_samples,
        }

    # ------------------------------------------------------------------- ack
    def record_ack(
        self, job_id: str, round_seq: int, step: int, rank: int, sample_ids: List[str]
    ) -> dict:
        with self._lock:
            job = self.get_job(job_id)
            if round_seq != job["round_seq"]:
                raise ConflictError(
                    f"确认来自旧轮次 round={round_seq}，当前轮次为 {job['round_seq']}；"
                    "回退生效后只接受新一轮确认"
                )
            if step <= job["current_step"]:
                raise ConflictError(
                    f"step={step} 已被检查点覆盖（current_step={job['current_step']}）"
                )
            if step > job["total_steps"]:
                raise ConflictError(f"step={step} 超出总步数 {job['total_steps']}")
            if not (0 <= rank < job["num_processes"]):
                raise ConflictError(
                    f"process_rank={rank} 越界，当前进程数 {job['num_processes']}"
                )

            start, end = self.step_slice(
                job["sample_count"], job["batch_size"], step
            )
            step_samples = job["sample_ids"][start:end]
            step_index = {sid: start + i for i, sid in enumerate(step_samples)}

            # 严格校验：上报样本必须属于该步，且必须是该进程被分派到的分片，
            # 防止进程漏读/串片后被静默接受。
            shard_start, shard_end = self.rank_shard(
                len(step_samples), job["num_processes"], rank
            )
            allowed = set(step_samples[shard_start:shard_end])
            unknown = [s for s in sample_ids if s not in step_index]
            if unknown:
                raise StoreError(
                    f"样本 {unknown} 不属于 step={step}；本步样本为 {step_samples}"
                )
            not_owned = [s for s in sample_ids if s not in allowed]
            if not_owned:
                raise StoreError(
                    f"样本 {not_owned} 不分派给 rank={rank}（当前 {job['num_processes']} 进程），"
                    f"该进程应读 {sorted(allowed)}"
                )
            if len(set(sample_ids)) != len(sample_ids):
                raise StoreError("单次上报的 sample_ids 存在重复")

            now = utc_now_iso()
            newly = 0
            for sid in sample_ids:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO confirmations"
                    " (job_id, round_seq, step, sample_index, sample_id,"
                    "  process_rank, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (job_id, round_seq, step, step_index[sid], sid, rank, now),
                )
                newly += cur.rowcount
            self._conn.execute(
                "UPDATE jobs SET updated_at = ? WHERE job_id = ?", (now, job_id)
            )
            self._conn.commit()
            return self.step_status(job_id, step) | {"newly_recorded": newly}

    def step_status(self, job_id: str, step: int) -> dict:
        job = self.get_job(job_id)
        start, end = self.step_slice(job["sample_count"], job["batch_size"], step)
        step_samples = job["sample_ids"][start:end]
        rows = self._conn.execute(
            "SELECT sample_index FROM confirmations"
            " WHERE job_id = ? AND round_seq = ? AND step = ?",
            (job_id, job["round_seq"], step),
        ).fetchall()
        confirmed_indexes = {r["sample_index"] for r in rows}
        missing = [
            sid
            for i, sid in enumerate(step_samples)
            if (start + i) not in confirmed_indexes
        ]
        return {
            "step": step,
            "round_seq": job["round_seq"],
            "total": len(step_samples),
            "confirmed": len(confirmed_indexes),
            "missing_sample_ids": missing,
            "complete": not missing,
        }

    # ------------------------------------------------------------ checkpoint
    def register_checkpoint(
        self, job_id: str, round_seq: int, step: int, checkpoint_uri: str
    ) -> dict:
        with self._lock:
            job = self.get_job(job_id)
            if round_seq != job["round_seq"]:
                raise ConflictError(
                    f"检查点上报来自旧轮次 round={round_seq}，当前轮次为 {job['round_seq']}"
                )
            if step <= job["current_step"]:
                raise ConflictError(
                    f"step={step} 不大于当前检查点步 {job['current_step']}，"
                    "已登记的检查点不能重复/回退登记"
                )
            if step > job["total_steps"]:
                raise ConflictError(f"step={step} 超出总步数 {job['total_steps']}")

            # 关键规则：上次保存（current_step+1）到末步 step 的每一步，
            # 在当前轮次都必须收齐全部样本确认。
            incomplete = []
            for s in range(job["current_step"] + 1, step + 1):
                st = self.step_status(job_id, s)
                if not st["complete"]:
                    incomplete.append({"step": s, "missing": st["missing_sample_ids"]})
            if incomplete:
                raise ConflictError(
                    "存在未收齐确认的步，拒绝登记检查点：" + json.dumps(
                        incomplete, ensure_ascii=False
                    )
                )

            now = utc_now_iso()
            # 同一步的检查点只能登记一次：step <= current_step 已在上面拦截，
            # 回退又会删除更晚的旧检查点，因此这里直接 INSERT 即可
            self._conn.execute(
                "INSERT INTO checkpoints"
                " (job_id, step, checkpoint_uri, round_seq, registered_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (job_id, step, checkpoint_uri, round_seq, now),
            )
            self._conn.execute(
                "UPDATE jobs SET current_step = ?, updated_at = ? WHERE job_id = ?",
                (step, now, job_id),
            )
            self._conn.commit()
            return {
                "job_id": job_id,
                "step": step,
                "checkpoint_uri": checkpoint_uri,
                "round_seq": round_seq,
                "registered_at": now,
            }

    # -------------------------------------------------------------- rollback
    def rollback(
        self,
        job_id: str,
        round_seq: int,
        checkpoint_step: int,
        num_processes: Optional[int],
    ) -> dict:
        with self._lock:
            job = self.get_job(job_id)
            if round_seq != job["round_seq"]:
                raise ConflictError(
                    f"回退请求来自旧轮次 round={round_seq}，当前轮次为 {job['round_seq']}"
                )
            if checkpoint_step < 0 or checkpoint_step > job["current_step"]:
                raise ConflictError(
                    f"checkpoint_step={checkpoint_step} 必须是已登记的检查点步，"
                    f"可选范围 0..{job['current_step']}"
                )
            if checkpoint_step > 0:
                ck = self._conn.execute(
                    "SELECT 1 FROM checkpoints WHERE job_id = ? AND step = ?",
                    (job_id, checkpoint_step),
                ).fetchone()
                if ck is None:
                    raise ConflictError(
                        f"step={checkpoint_step} 没有已登记的检查点，不能回退到该位置"
                    )

            new_processes = (
                num_processes if num_processes is not None else job["num_processes"]
            )
            new_round = job["round_seq"] + 1
            now = utc_now_iso()
            # 清空当前轮次的所有确认——旧确认随旧轮次一并作废
            self._conn.execute(
                "DELETE FROM confirmations WHERE job_id = ? AND round_seq = ?",
                (job_id, job["round_seq"]),
            )
            # 作废晚于回退目标的检查点：它们位于被放弃的进度上，
            # 保留会污染回退后的可选检查点列表，并与新一轮登记冲突
            self._conn.execute(
                "DELETE FROM checkpoints WHERE job_id = ? AND step > ?",
                (job_id, checkpoint_step),
            )
            self._conn.execute(
                "UPDATE jobs SET current_step = ?, num_processes = ?,"
                " round_seq = ?, updated_at = ? WHERE job_id = ?",
                (checkpoint_step, new_processes, new_round, now, job_id),
            )
            self._conn.execute(
                "INSERT INTO rollback_events"
                " (job_id, from_step, checkpoint_step, old_processes,"
                "  new_processes, round_seq, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    job["current_step"],
                    checkpoint_step,
                    job["num_processes"],
                    new_processes,
                    new_round,
                    now,
                ),
            )
            self._conn.commit()
            return {
                "job_id": job_id,
                "checkpoint_step": checkpoint_step,
                "num_processes": new_processes,
                "round_seq": new_round,
                "next_step": checkpoint_step + 1,
                "rolled_back_at": now,
            }
