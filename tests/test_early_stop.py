"""早停功能的端到端自测。

覆盖：乱序上报触发早停（上报/自动回退/停训原子生效）、缺报等待、
同值重报幂等与改值拒绝、并列取最早、手动回退后保留可用检查点损失并
拒收作废检查点的迟到上报、停训后拒绝各类写操作、失败后可重报、重启后
停训状态与选中检查点仍可查。

运行：
    PYTHONPATH=. pytest tests/test_early_stop.py -v
"""
from __future__ import annotations

import importlib

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


def _make_job(client, job_id="job-es", samples=20, batch=4, procs=2, patience=2):
    sample_ids = [f"s{i:02d}" for i in range(samples)]
    body = {
        "job_id": job_id,
        "sample_ids": sample_ids,
        "batch_size": batch,
        "num_processes": procs,
    }
    if patience is not None:
        body["patience"] = patience
    r = client.post("/jobs", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _run_step(client, job_id, step, num_processes, round_seq):
    """模拟所有进程：预取分片 -> 确认。"""
    for rank in range(num_processes):
        pf = client.get(
            f"/jobs/{job_id}/prefetch",
            params={"step": step, "process_rank": rank, "round_seq": round_seq},
        )
        assert pf.status_code == 200, pf.text
        ids = pf.json()["assigned_sample_ids"]
        if not ids:
            continue
        r = client.post(
            f"/jobs/{job_id}/ack?round_seq={round_seq}",
            json={"step": step, "process_rank": rank, "sample_ids": ids},
        )
        assert r.status_code == 200, r.text


def _register(client, job_id, step, round_seq=1):
    r = client.post(
        f"/jobs/{job_id}/checkpoints?round_seq={round_seq}",
        json={"step": step, "checkpoint_uri": f"ckpt://{job_id}/step-{step}-r{round_seq}"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _report(client, job_id, step, checkpoint_round, loss):
    return client.post(
        f"/jobs/{job_id}/losses",
        json={"step": step, "checkpoint_round": checkpoint_round, "loss": loss},
    )


def _run_and_register(client, job_id, steps, procs=2, round_seq=1):
    for s in steps:
        _run_step(client, job_id, s, procs, round_seq)
        _register(client, job_id, s, round_seq)


# --------------------------------------------------------------------- cases
def test_loss_report_rejected_when_early_stop_not_enabled(client):
    _make_job(client, patience=None)
    _run_and_register(client, "job-es", [1])
    r = _report(client, "job-es", 1, 1, 0.5)
    assert r.status_code == 409
    assert "未启用早停" in r.json()["detail"]
    # 未启用时查询早停状态返回 enabled=false，不报错
    st = client.get("/jobs/job-es/early_stop").json()
    assert st["enabled"] is False and st["status"] == "running"


def test_loss_report_requires_existing_checkpoint(client):
    _make_job(client)
    # 检查点未登记就上报 -> 拒绝
    r = _report(client, "job-es", 1, 1, 0.5)
    assert r.status_code == 409
    # 登记轮次对不上（检查点是 round 1 登记的，报 round 2）-> 拒绝
    _run_and_register(client, "job-es", [1])
    r = _report(client, "job-es", 1, 2, 0.5)
    assert r.status_code == 409
    # 轮次正确 -> 受理
    r = _report(client, "job-es", 1, 1, 0.5)
    assert r.status_code == 200, r.text
    assert r.json()["newly_recorded"] is True


def test_same_value_repost_counts_once_and_different_value_rejected(client):
    _make_job(client)
    _run_and_register(client, "job-es", [1])
    r1 = _report(client, "job-es", 1, 1, 0.5)
    assert r1.json()["newly_recorded"] is True
    # 同值重报：只算一次
    r2 = _report(client, "job-es", 1, 1, 0.5)
    assert r2.status_code == 200
    assert r2.json()["newly_recorded"] is False
    assert len(client.get("/jobs/job-es/losses").json()) == 1
    # 改报不同值：拒绝
    r3 = _report(client, "job-es", 1, 1, 0.51)
    assert r3.status_code == 409
    # 非法损失值（NaN）：业务校验 400 拒绝（httpx 不愿编码 NaN，直接发原始体）
    r4 = client.post("/jobs/job-es/losses",
                     content='{"step": 1, "checkpoint_round": 1, "loss": NaN}',
                     headers={"Content-Type": "application/json"})
    assert r4.status_code == 400


def test_out_of_order_reports_trigger_early_stop(client):
    """乱序上报：step3/step4 的损失先于 step2 到达，判定在 step2 处等待；
    step2 补齐后连续判定到 step4，连续 2 次未刷新最低损失触发早停，
    上报 + 自动回退到最优检查点 step2 + 停训一并生效。"""
    _make_job(client, patience=2)  # 20 样本 / 批 4 -> 5 步
    _run_and_register(client, "job-es", [1, 2, 3, 4, 5])

    r = _report(client, "job-es", 1, 1, 1.0)
    j = r.json()["judgment"]
    assert (j["judged_through_step"], j["best_checkpoint_step"],
            j["no_improve_count"], j["triggered"]) == (1, 1, 0, False)

    # 乱序：step3、step4 先报，判定停在 step2 缺口处等待
    r = _report(client, "job-es", 3, 1, 0.9)
    j = r.json()["judgment"]
    assert j["waiting_for_step"] == 2 and j["judged_through_step"] == 1
    assert j["triggered"] is False
    r = _report(client, "job-es", 4, 1, 0.85)
    assert r.json()["judgment"]["waiting_for_step"] == 2

    # step2 补齐：连续判定 2(0.8 刷新)->3(未刷新 1)->4(未刷新 2) -> 触发
    r = _report(client, "job-es", 2, 1, 0.8)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["judgment"]["triggered"] is True
    assert body["judgment"]["best_checkpoint_step"] == 2
    es = body["early_stop"]
    assert es["status"] == "stopped"
    assert es["checkpoint_step"] == 2          # 回退到损失最低的检查点
    assert es["round_seq"] == 2                # 自动回退开启新一轮
    assert body["job_status"] == "stopped"

    # 原子生效后的终态
    job = client.get("/jobs/job-es").json()
    assert job["status"] == "stopped"
    assert job["current_step"] == 2
    assert job["round_seq"] == 2
    assert job["stop_checkpoint_step"] == 2
    assert job["stopped_at"] is not None
    # 晚于最优检查点的检查点与损失记录一并作废；可用的保留
    assert [c["step"] for c in client.get("/jobs/job-es/checkpoints").json()] == [1, 2]
    losses = client.get("/jobs/job-es/losses").json()
    assert [(l["step"], l["loss"]) for l in losses] == [(1, 1.0), (2, 0.8)]
    # 回退历史记录了这次自动回退
    events = client.get("/jobs/job-es/rollbacks").json()
    assert len(events) == 1
    assert events[0]["kind"] == "early_stop"
    assert events[0]["from_step"] == 5 and events[0]["checkpoint_step"] == 2
    # 早停状态接口同样可查
    st = client.get("/jobs/job-es/early_stop").json()
    assert st["status"] == "stopped" and st["stop_checkpoint_step"] == 2


def test_tie_breaks_to_earliest_checkpoint(client):
    """最低损失并列时回退到最早的检查点。"""
    _make_job(client, patience=2)
    _run_and_register(client, "job-es", [1, 2, 3])
    _report(client, "job-es", 1, 1, 0.5)
    _report(client, "job-es", 2, 1, 0.5)   # 并列最低，未严格刷新 -> 计数 1
    r = _report(client, "job-es", 3, 1, 0.5)  # 计数 2 -> 触发
    body = r.json()
    assert body["judgment"]["triggered"] is True
    assert body["early_stop"]["checkpoint_step"] == 1  # 并列取最早
    assert client.get("/jobs/job-es").json()["stop_checkpoint_step"] == 1


def test_manual_rollback_keeps_usable_losses_and_rejects_late_reports(client):
    """手动回退后：仍可用检查点的损失保留并继续参与判定；
    已作废检查点的迟到结果拒收——即使同步号后来在新轮次重新登记。"""
    _make_job(client, patience=3)
    _run_and_register(client, "job-es", [1, 2, 3])
    _report(client, "job-es", 1, 1, 1.0)
    _report(client, "job-es", 2, 1, 0.9)
    _report(client, "job-es", 3, 1, 0.95)

    # 手动回退到 step1：step2/step3 检查点作废，step1 的损失保留
    r = client.post("/jobs/job-es/rollback?round_seq=1",
                    json={"checkpoint_step": 1})
    assert r.status_code == 200, r.text
    assert [c["step"] for c in client.get("/jobs/job-es/checkpoints").json()] == [1]
    losses = client.get("/jobs/job-es/losses").json()
    assert [(l["step"], l["round_seq"], l["loss"]) for l in losses] == [(1, 1, 1.0)]

    # 新一轮重新登记 step2（round 2）
    _run_step(client, "job-es", 2, 2, 2)
    _register(client, "job-es", 2, round_seq=2)

    # 旧检查点（step2, round1）的迟到结果：拒收，即使同一步已重新登记
    r = _report(client, "job-es", 2, 1, 0.9)
    assert r.status_code == 409
    # 新检查点（step2, round2）的上报：受理，且判定从最早的 step1 接上
    r = _report(client, "job-es", 2, 2, 0.85)
    assert r.status_code == 200, r.text
    j = r.json()["judgment"]
    assert j["judged_through_step"] == 2
    assert j["best_checkpoint_step"] == 2 and j["best_loss"] == 0.85
    assert j["no_improve_count"] == 0 and j["triggered"] is False


def test_kept_checkpoint_still_accepts_report_after_manual_rollback(client):
    """手动回退保留的检查点，仍可按其原登记轮次补报损失并参与判定。"""
    _make_job(client, patience=2)
    _run_and_register(client, "job-es", [1, 2])
    _report(client, "job-es", 2, 1, 0.9)   # step1 缺报，判定等待
    # 手动回退到 step1：step2 检查点及其损失作废，step1 保留
    r = client.post("/jobs/job-es/rollback?round_seq=1",
                    json={"checkpoint_step": 1})
    assert r.status_code == 200, r.text
    assert client.get("/jobs/job-es/losses").json() == []
    # 保留的 step1（round 1 登记）首次上报：受理
    r = _report(client, "job-es", 1, 1, 1.0)
    assert r.status_code == 200, r.text
    j = r.json()["judgment"]
    assert j["judged_through_step"] == 1 and j["best_checkpoint_step"] == 1


def test_stopped_job_rejects_writes(client):
    """停训后：预取、样本确认、检查点登记、手动回退一律 409；
    新损失上报拒收，已保留记录的同值重报仍幂等。"""
    _make_job(client, patience=1)
    _run_and_register(client, "job-es", [1, 2])
    _report(client, "job-es", 1, 1, 1.0)
    r = _report(client, "job-es", 2, 1, 1.1)  # patience=1，未刷新即触发
    assert r.json()["early_stop"]["checkpoint_step"] == 1
    assert client.get("/jobs/job-es").json()["status"] == "stopped"

    # 预取拒绝
    r = client.get("/jobs/job-es/prefetch",
                   params={"step": 2, "process_rank": 0, "round_seq": 2})
    assert r.status_code == 409
    # 样本确认拒绝
    r = client.post("/jobs/job-es/ack?round_seq=2",
                    json={"step": 2, "process_rank": 0, "sample_ids": ["s04"]})
    assert r.status_code == 409
    # 检查点登记拒绝
    r = client.post("/jobs/job-es/checkpoints?round_seq=2",
                    json={"step": 2, "checkpoint_uri": "ckpt://late"})
    assert r.status_code == 409
    # 手动回退拒绝
    r = client.post("/jobs/job-es/rollback?round_seq=2",
                    json={"checkpoint_step": 1})
    assert r.status_code == 409
    # 已作废检查点（step2, round1）的迟到上报：拒收
    r = _report(client, "job-es", 2, 1, 1.1)
    assert r.status_code == 409
    # 保留检查点的同值重报：仍幂等受理，只算一次
    r = _report(client, "job-es", 1, 1, 1.0)
    assert r.status_code == 200 and r.json()["newly_recorded"] is False
    assert r.json()["job_status"] == "stopped"
    # 保留检查点改报不同值：仍拒绝
    r = _report(client, "job-es", 1, 1, 0.9)
    assert r.status_code == 409


def test_failed_trigger_report_can_be_retried(client, monkeypatch):
    """触发早停的事务中途失败：上报/回退/停训都不留痕迹，可原样重报。"""
    _make_job(client, patience=1)
    _run_and_register(client, "job-es", [1, 2])
    _report(client, "job-es", 1, 1, 1.0)

    db = client.main_module.app.state.db
    original = db._apply_rollback

    def boom(*args, **kwargs):
        raise RuntimeError("模拟存储故障")

    monkeypatch.setattr(db, "_apply_rollback", boom)
    with pytest.raises(RuntimeError):
        _report(client, "job-es", 2, 1, 1.1)

    # 失败不留痕迹：触发上报未落库、未回退、未停训
    job = client.get("/jobs/job-es").json()
    assert job["status"] == "running"
    assert job["current_step"] == 2 and job["round_seq"] == 1
    assert [(l["step"], l["loss"]) for l in
            client.get("/jobs/job-es/losses").json()] == [(1, 1.0)]
    assert client.get("/jobs/job-es/rollbacks").json() == []

    # 故障恢复后原样重报：正常触发并一起生效
    monkeypatch.setattr(db, "_apply_rollback", original)
    r = _report(client, "job-es", 2, 1, 1.1)
    assert r.status_code == 200, r.text
    assert r.json()["early_stop"]["checkpoint_step"] == 1
    job = client.get("/jobs/job-es").json()
    assert job["status"] == "stopped" and job["stop_checkpoint_step"] == 1


def test_stop_state_survives_restart(tmp_path, monkeypatch):
    """重启后仍能查到停训状态和选中的检查点，写操作仍被拒绝。"""
    db_path = str(tmp_path / "restart.db")
    monkeypatch.setenv("DISPATCHER_DB", db_path)
    import app.main as main
    importlib.reload(main)

    with TestClient(main.app) as c:
        _make_job(c, "job-stop", patience=1)
        _run_and_register(c, "job-stop", [1, 2])
        _report(c, "job-stop", 1, 1, 0.7)
        r = _report(c, "job-stop", 2, 1, 0.8)
        assert r.json()["early_stop"]["checkpoint_step"] == 1

    # —— 重启：重新加载模块、重新连接同一个 DB 文件 ——
    importlib.reload(main)
    with TestClient(main.app) as c2:
        job = c2.get("/jobs/job-stop").json()
        assert job["status"] == "stopped"
        assert job["stop_checkpoint_step"] == 1
        assert job["stopped_at"] is not None
        st = c2.get("/jobs/job-stop/early_stop").json()
        assert st["status"] == "stopped" and st["stop_checkpoint_step"] == 1
        assert [(l["step"], l["loss"]) for l in
                c2.get("/jobs/job-stop/losses").json()] == [(1, 0.7)]
        r = c2.get("/jobs/job-stop/prefetch",
                   params={"step": 2, "process_rank": 0, "round_seq": 2})
        assert r.status_code == 409


def test_patience_must_be_positive(client):
    r = client.post("/jobs", json={
        "job_id": "job-bad", "sample_ids": ["a", "b"],
        "batch_size": 1, "num_processes": 1, "patience": 0,
    })
    assert r.status_code == 422
