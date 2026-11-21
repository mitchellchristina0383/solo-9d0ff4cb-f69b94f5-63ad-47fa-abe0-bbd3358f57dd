"""端到端自测：覆盖预取、去重确认、检查点门控、回退换进程数、重启续跑。

运行：
    PYTHONPATH=. pytest -v
"""
from __future__ import annotations

import importlib
import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DISPATCHER_DB", db_path)
    import app.main as main
    importlib.reload(main)  # 保证每个用例拿到全新的 DB 文件
    with TestClient(main.app) as c:
        c.db_path = db_path
        c.main_module = main
        yield c


def _make_job(client, job_id="job-1", samples=12, batch=4, procs=2):
    sample_ids = [f"s{i:02d}" for i in range(samples)]
    r = client.post("/jobs", json={
        "job_id": job_id,
        "sample_ids": sample_ids,
        "batch_size": batch,
        "num_processes": procs,
    })
    assert r.status_code == 201, r.text
    return sample_ids, r.json()


def _prefetch(client, job_id, step, rank, round_seq):
    r = client.get(f"/jobs/{job_id}/prefetch",
                   params={"step": step, "process_rank": rank,
                           "round_seq": round_seq})
    assert r.status_code == 200, r.text
    return r.json()


def _ack(client, job_id, step, rank, sample_ids, round_seq):
    return client.post(f"/jobs/{job_id}/ack?round_seq={round_seq}",
                       json={"step": step, "process_rank": rank,
                             "sample_ids": sample_ids})


def _run_step(client, job_id, step, num_processes, round_seq):
    """模拟所有进程：预取分片 -> 确认，返回本步全量样本。"""
    all_samples = []
    for rank in range(num_processes):
        pf = _prefetch(client, job_id, step, rank, round_seq)
        assert pf["round_seq"] == round_seq
        all_samples.extend(pf["assigned_sample_ids"])
        r = _ack(client, job_id, step, rank, pf["assigned_sample_ids"], round_seq)
        assert r.status_code == 200, r.text
    return all_samples


# --------------------------------------------------------------------- cases
def test_health_and_create_job(client):
    samples, job = _make_job(client)
    assert job["total_steps"] == 3          # 12 个样本 / 批 4
    assert job["current_step"] == 0
    assert job["round_seq"] == 1
    assert client.get("/health").json() == {"status": "ok"}


def test_prefetch_partitions_cover_step_in_order(client):
    samples, job = _make_job(client)
    # 2 进程、批 4：rank0 -> 前两个，rank1 -> 后两个
    pf0 = _prefetch(client, "job-1", 1, 0, 1)
    pf1 = _prefetch(client, "job-1", 1, 1, 1)
    assert pf0["assigned_sample_ids"] == ["s00", "s01"]
    assert pf1["assigned_sample_ids"] == ["s02", "s03"]
    assert pf0["all_sample_ids"] == ["s00", "s01", "s02", "s03"]
    # 分片并集 == 本步全量、顺序保持
    assert pf0["assigned_sample_ids"] + pf1["assigned_sample_ids"] == pf0["all_sample_ids"]
    # 步 2 取下一批
    pf = _prefetch(client, "job-1", 2, 0, 1)
    assert pf["all_sample_ids"] == ["s04", "s05", "s06", "s07"]
    # 末步不满批：12 样本、批 4 时末步仍是 4；改用 10 样本验证短批
    samples10, _ = _make_job(client, "job-short", samples=10, batch=4, procs=3)
    pf = _prefetch(client, "job-short", 3, 0, 1)
    assert pf["all_sample_ids"] == ["s08", "s09"]
    # 3 进程分 2 个样本：rank0/rank1 各 1，rank2 为空分片
    pf2 = _prefetch(client, "job-short", 3, 2, 1)
    assert pf2["assigned_sample_ids"] == []


def test_duplicate_ack_counts_once(client):
    _make_job(client)
    pf = _prefetch(client, "job-1", 1, 0, 1)
    r1 = _ack(client, "job-1", 1, 0, pf["assigned_sample_ids"], 1)
    assert r1.json()["newly_recorded"] == 2
    # 完全重复上报：不再新增
    r2 = _ack(client, "job-1", 1, 0, pf["assigned_sample_ids"], 1)
    assert r2.status_code == 200
    assert r2.json()["newly_recorded"] == 0
    assert r2.json()["confirmed_sample_count"] == 2
    # 单样本重复也幂等
    r3 = _ack(client, "job-1", 1, 0, ["s00"], 1)
    assert r3.json()["newly_recorded"] == 0


def test_ack_rejects_wrong_step_and_wrong_rank(client):
    _make_job(client)
    # 样本不属于该步
    r = _ack(client, "job-1", 1, 0, ["s04"], 1)
    assert r.status_code == 400
    # 样本属于该步但不属于该 rank 的分片
    r = _ack(client, "job-1", 1, 0, ["s02"], 1)
    assert r.status_code == 400
    # rank 越界
    r = client.post("/jobs/job-1/ack?round_seq=1",
                    json={"step": 1, "process_rank": 5, "sample_ids": ["s00"]})
    assert r.status_code == 409


def test_checkpoint_requires_all_steps_confirmed(client):
    _make_job(client)
    # 一步都没确认 -> 拒绝
    r = client.post("/jobs/job-1/checkpoints?round_seq=1",
                    json={"step": 1, "checkpoint_uri": "ckpt://s1"})
    assert r.status_code == 409
    # 只跑完 step1 -> 登记 step1 成功
    _run_step(client, "job-1", 1, 2, 1)
    r = client.post("/jobs/job-1/checkpoints?round_seq=1",
                    json={"step": 1, "checkpoint_uri": "ckpt://s1"})
    assert r.status_code == 200, r.text
    # 跨到 step3 但 step2 未收齐 -> 拒绝并指出缺口
    _run_step(client, "job-1", 3, 2, 1)
    r = client.post("/jobs/job-1/checkpoints?round_seq=1",
                    json={"step": 3, "checkpoint_uri": "ckpt://s3"})
    assert r.status_code == 409
    body = r.json()["detail"]
    assert '"step": 2' in body
    # 补齐 step2 后登记成功
    _run_step(client, "job-1", 2, 2, 1)
    r = client.post("/jobs/job-1/checkpoints?round_seq=1",
                    json={"step": 3, "checkpoint_uri": "ckpt://s3"})
    assert r.status_code == 200
    job = client.get("/jobs/job-1").json()
    assert job["current_step"] == 3
    # 不能重复登记旧步
    r = client.post("/jobs/job-1/checkpoints?round_seq=1",
                    json={"step": 2, "checkpoint_uri": "ckpt://old"})
    assert r.status_code == 409


def test_prefetch_refuses_steps_at_or_before_checkpoint(client):
    _make_job(client)
    _run_step(client, "job-1", 1, 2, 1)
    client.post("/jobs/job-1/checkpoints?round_seq=1",
                json={"step": 1, "checkpoint_uri": "ckpt://s1"})
    r = client.get("/jobs/job-1/prefetch",
                   params={"step": 1, "process_rank": 0, "round_seq": 1})
    assert r.status_code == 409


def test_rollback_keeps_sample_order_and_re_shards_for_new_process_count(client):
    samples, _ = _make_job(client)  # 12 样本 / 批 4 / 2 进程 / 共 3 步
    # 跑完前两步，分别登记 step1、step2 检查点
    _run_step(client, "job-1", 1, 2, 1)
    assert client.post("/jobs/job-1/checkpoints?round_seq=1",
                       json={"step": 1, "checkpoint_uri": "ckpt://s1"}).status_code == 200
    _run_step(client, "job-1", 2, 2, 1)
    r = client.post("/jobs/job-1/checkpoints?round_seq=1",
                    json={"step": 2, "checkpoint_uri": "ckpt://s2"})
    assert r.status_code == 200

    # 回退到 step1，并把进程数从 2 改成 3
    r = client.post("/jobs/job-1/rollback?round_seq=1",
                    json={"checkpoint_step": 1, "num_processes": 3})
    assert r.status_code == 200, r.text
    rb = r.json()
    assert rb["round_seq"] == 2
    assert rb["num_processes"] == 3
    assert rb["next_step"] == 2

    job = client.get("/jobs/job-1").json()
    assert job["current_step"] == 1
    assert job["num_processes"] == 3
    assert job["round_seq"] == 2

    # 样本全局顺序、每步划分照旧：step2 仍是 s04..s07
    pf0 = _prefetch(client, "job-1", 2, 0, 2)
    pf1 = _prefetch(client, "job-1", 2, 1, 2)
    pf2 = _prefetch(client, "job-1", 2, 2, 2)
    assert pf0["all_sample_ids"] == ["s04", "s05", "s06", "s07"]
    # 4 个样本 / 3 进程 -> 2,1,1
    assert pf0["assigned_sample_ids"] == ["s04", "s05"]
    assert pf1["assigned_sample_ids"] == ["s06"]
    assert pf2["assigned_sample_ids"] == ["s07"]
    assert (pf0["assigned_sample_ids"] + pf1["assigned_sample_ids"]
            + pf2["assigned_sample_ids"]) == pf0["all_sample_ids"]


def test_stale_round_rejected_after_rollback(client):
    _make_job(client)
    _run_step(client, "job-1", 1, 2, 1)
    client.post("/jobs/job-1/checkpoints?round_seq=1",
                json={"step": 1, "checkpoint_uri": "ckpt://s1"})
    client.post("/jobs/job-1/rollback?round_seq=1",
                json={"checkpoint_step": 0, "num_processes": 2})

    # 旧轮次预取被拒
    r = client.get("/jobs/job-1/prefetch",
                   params={"step": 1, "process_rank": 0, "round_seq": 1})
    assert r.status_code == 409
    # 旧轮次确认被拒
    r = _ack(client, "job-1", 1, 0, ["s00", "s01"], 1)
    assert r.status_code == 409
    # 旧轮次检查点上报被拒
    r = client.post("/jobs/job-1/checkpoints?round_seq=1",
                    json={"step": 1, "checkpoint_uri": "ckpt://stale"})
    assert r.status_code == 409
    # 旧轮次回退也被拒
    r = client.post("/jobs/job-1/rollback?round_seq=1",
                    json={"checkpoint_step": 0})
    assert r.status_code == 409
    # 新一轮确认从零开始（旧确认已清空）
    st = client.get("/jobs/job-1/steps").json()
    assert all(s["confirmed"] == 0 for s in st)


def test_rollback_requires_existing_checkpoint(client):
    _make_job(client)
    _run_step(client, "job-1", 1, 2, 1)
    # step1 未登记检查点，不能回退到 1；只能回退到 0
    r = client.post("/jobs/job-1/rollback?round_seq=1",
                    json={"checkpoint_step": 1})
    assert r.status_code == 409
    r = client.post("/jobs/job-1/rollback?round_seq=1",
                    json={"checkpoint_step": 0})
    assert r.status_code == 200


def test_full_round_after_rollback_with_3_processes(client):
    """回退后换 3 进程，完整跑完后续步骤并登记检查点。"""
    _make_job(client)  # 12 / 4 / 2 -> 3 步
    _run_step(client, "job-1", 1, 2, 1)
    client.post("/jobs/job-1/checkpoints?round_seq=1",
                json={"step": 1, "checkpoint_uri": "ckpt://s1"})
    client.post("/jobs/job-1/rollback?round_seq=1",
                json={"checkpoint_step": 1, "num_processes": 3})

    # 新一轮 3 进程跑完 step2、step3
    for step in (2, 3):
        all_s = _run_step(client, "job-1", step, 3, 2)
        expected = [f"s{i:02d}" for i in range((step - 1) * 4, step * 4)]
        assert sorted(all_s) == sorted(expected)

    # step2 确认未齐时不能直接登记 step3（这里已齐，再验证门控逻辑）：
    r = client.post("/jobs/job-1/checkpoints?round_seq=2",
                    json={"step": 3, "checkpoint_uri": "ckpt://s3-r2"})
    assert r.status_code == 200, r.text
    job = client.get("/jobs/job-1").json()
    assert job["current_step"] == 3 and job["round_seq"] == 2

    # 检查点列表与回退历史都可查
    cks = client.get("/jobs/job-1/checkpoints").json()
    assert [c["step"] for c in cks] == [1, 3]
    history = client.get("/jobs/job-1/rollbacks").json()
    assert history[0]["new_processes"] == 3


def test_state_survives_restart(tmp_path, monkeypatch):
    """模拟服务重启：重新创建 Database/FastAPI 实例，从同一 SQLite 文件恢复。"""
    db_path = str(tmp_path / "restart.db")
    monkeypatch.setenv("DISPATCHER_DB", db_path)
    import app.main as main
    importlib.reload(main)

    with TestClient(main.app) as c:
        _make_job(c, "job-restart")
        _run_step(c, "job-restart", 1, 2, 1)
        c.post("/jobs/job-restart/checkpoints?round_seq=1",
               json={"step": 1, "checkpoint_uri": "ckpt://s1"})
        c.post("/jobs/job-restart/rollback?round_seq=1",
               json={"checkpoint_step": 1, "num_processes": 4})

    # —— 重启：重新加载模块、重新连接同一个 DB 文件 ——
    importlib.reload(main)
    with TestClient(main.app) as c2:
        job = c2.get("/jobs/job-restart").json()
        assert job["current_step"] == 1
        assert job["num_processes"] == 4
        assert job["round_seq"] == 2
        assert job["sample_count"] == 12
        # 仍可从当前检查点的下一步续跑，且按新进程数（4）切分
        pf = _prefetch(c2, "job-restart", 2, 3, 2)
        assert pf["assigned_sample_ids"] == ["s07"]  # 4 样本 / 4 进程，每进程 1 个
        _run_step(c2, "job-restart", 2, 4, 2)
        r = c2.post("/jobs/job-restart/checkpoints?round_seq=2",
                    json={"step": 2, "checkpoint_uri": "ckpt://s2-r2"})
        assert r.status_code == 200, r.text
        # 旧检查点记录仍在
        steps = [c["step"] for c in c2.get("/jobs/job-restart/checkpoints").json()]
        assert steps == [1, 2]


def test_rollback_prunes_later_checkpoints(client):
    """回退到较早检查点后，更晚的旧检查点必须作废；新一轮可重新登记同一步号。"""
    _make_job(client)  # 3 步，2 进程
    for step in (1, 2, 3):
        _run_step(client, "job-1", step, 2, 1)
    for step in (1, 2, 3):
        assert client.post("/jobs/job-1/checkpoints?round_seq=1",
                           json={"step": step,
                                 "checkpoint_uri": f"ckpt://old-s{step}"}).status_code == 200

    # 回退到 step1：step2/step3 的检查点应被删除，step1 保留
    r = client.post("/jobs/job-1/rollback?round_seq=1",
                    json={"checkpoint_step": 1, "num_processes": 2})
    assert r.status_code == 200
    steps = [c["step"] for c in client.get("/jobs/job-1/checkpoints").json()]
    assert steps == [1]

    # 新一轮重新跑到 step2 并登记，不应与旧记录冲突，URI 为新值
    _run_step(client, "job-1", 2, 2, 2)
    r = client.post("/jobs/job-1/checkpoints?round_seq=2",
                    json={"step": 2, "checkpoint_uri": "ckpt://new-s2"})
    assert r.status_code == 200
    cks = {c["step"]: c for c in client.get("/jobs/job-1/checkpoints").json()}
    assert sorted(cks) == [1, 2]
    assert cks[2]["checkpoint_uri"] == "ckpt://new-s2"
    assert cks[2]["round_seq"] == 2


def test_duplicate_job_and_not_found(client):
    _make_job(client)
    r = client.post("/jobs", json={
        "job_id": "job-1", "sample_ids": ["a"],
        "batch_size": 1, "num_processes": 1,
    })
    assert r.status_code == 409
    assert client.get("/jobs/nope").status_code == 404
