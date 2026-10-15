"""Background jobs that keep running when the user switches pages.

Streamlit stops a page's script as soon as the user navigates away, which used to cancel
chats, agent runs, comparisons and downloads half-way. Long work now runs in a worker
thread; pages only read the job's state and redraw it with an auto-refreshing fragment.

Worker functions must not call Streamlit: they receive the Job and write to `job.state`,
`job.detail` and `job.progress`. Anything that needs session state (e.g. the Ollama URL)
is read on the page and passed in.
"""

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import streamlit as st
from i18n import tr

KEEP_FINISHED = 40
ICONS = {
    "running": ":material/progress_activity:",
    "paused": ":orange[:material/pause_circle:]",
    "done": ":green[:material/check_circle:]",
    "error": ":red[:material/error:]",
    "cancelled": ":material/stop_circle:",
}


@dataclass
class Job:
    kind: str  # "pull", "chat", "agent", "compare" or "departments"
    title: str
    page: str  # page file, for the sidebar link back to it
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "running"  # running, paused (waiting for a human), done, error, cancelled
    detail: str = ""
    progress: float | None = None
    state: dict = field(default_factory=dict)
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def running(self) -> bool:
        return self.status == "running"

    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started


@st.cache_resource
def _registry() -> dict[str, Job]:
    """One registry per server process, shared by every page and browser tab."""
    return {}


_lock = threading.Lock()


def start(job: Job, work: Callable[[Job], None]) -> Job:
    with _lock:
        registry = _registry()
        registry[job.id] = job
        finished = [j for j in registry.values() if j.finished]
        for old in sorted(finished, key=lambda j: j.finished)[:-KEEP_FINISHED]:
            registry.pop(old.id, None)
    _spawn(job, work)
    return job


def resume(job: Job, work: Callable[[Job], None]) -> None:
    """Continue a paused job (e.g. after a human approved an agent's external call)."""
    job.status, job.finished = "running", None
    _spawn(job, work)


def _spawn(job: Job, work: Callable[[Job], None]) -> None:
    def target():
        try:
            work(job)
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.state["traceback"] = traceback.format_exc()
            job.status = "error"
        else:
            if job.status == "running":  # the work may have paused the job itself
                job.status = "cancelled" if job.cancel.is_set() else "done"
        if job.status != "paused":
            job.finished = time.time()

    threading.Thread(target=target, name=f"job-{job.kind}-{job.id}", daemon=True).start()


def get(job_id: str | None) -> Job | None:
    return _registry().get(job_id) if job_id else None


def find(kind: str, **match) -> Job | None:
    """Most recent job of this kind whose state matches, e.g. find("pull", model=tag)."""
    candidates = [
        j
        for j in _registry().values()
        if j.kind == kind and all(j.state.get(k) == v for k, v in match.items())
    ]
    return max(candidates, key=lambda j: j.started, default=None)


def active() -> list[Job]:
    return [j for j in _registry().values() if j.status in ("running", "paused")]


@st.fragment(run_every=2)
def sidebar_panel() -> None:
    """Running jobs on every page, so work started elsewhere stays visible."""
    recent = [
        j
        for j in _registry().values()
        if j.status in ("running", "paused") or (j.finished and time.time() - j.finished < 20)
    ]
    if not recent:
        return
    st.markdown(
        tr(
            "**背景工作**（切換頁面不會中斷）",
            "**Background jobs** (they keep running if you switch pages)",
        )
    )
    for job in sorted(recent, key=lambda j: j.started):
        detail = f" · {job.detail}" if job.detail and job.status != "done" else ""
        st.page_link(
            job.page,
            label=f"{ICONS[job.status]} {job.title}{detail} · {job.elapsed():.0f}s",
        )
        if job.running and job.progress is not None:
            st.progress(min(max(job.progress, 0.0), 1.0))


def _recent(seconds: float = 60) -> list[Job]:
    now = time.time()
    return sorted(
        (
            j
            for j in _registry().values()
            if j.status in ("running", "paused") or (j.finished and now - j.finished < seconds)
        ),
        key=lambda j: j.started,
    )


@st.fragment(run_every=2)
def _job_list() -> None:
    recent = _recent()
    if not recent:
        st.caption(tr("目前沒有背景工作。", "No background tasks."))
        return
    for job in recent:
        detail = f" · {job.detail}" if job.detail and job.status != "done" else ""
        st.page_link(
            job.page, label=f"{ICONS[job.status]} {job.title}{detail} · {job.elapsed():.0f}s"
        )
        if job.running and job.progress is not None:
            st.progress(min(max(job.progress, 0.0), 1.0))


def header_popover() -> None:
    """Top-right button listing background tasks (they keep running across pages)."""
    active_jobs = active()
    label = (
        tr(f"{len(active_jobs)} 個工作執行中", f"{len(active_jobs)} running")
        if active_jobs
        else tr("背景工作", "Tasks")
    )
    icon = ":material/progress_activity:" if active_jobs else ":material/task_alt:"
    with st.popover(label, icon=icon, use_container_width=True):
        st.markdown(
            tr(
                "**背景工作**（切換頁面不會中斷）",
                "**Background tasks** (they keep running when you switch pages)",
            )
        )
        _job_list()
