from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RunLifecycle:
    running: bool = False
    finished_emitted: bool = False
    run_id: int = 0
    active_run_id: int | None = None
    state: str = "IDLE"
    terminal_count: int = 0
    idle_transition_count: int = 0
    log_copy_count: int = 0
    run_terminal_count: int = 0
    run_idle_transition_count: int = 0
    run_log_copy_count: int = 0
    _completed_run_ids: set[int] = field(default_factory=set)
    _log_copied_run_ids: set[int] = field(default_factory=set)

    def start(self) -> int:
        if self.running:
            raise RuntimeError("gui_run_already_active")
        self.run_id += 1
        self.active_run_id = self.run_id
        self.running = True
        self.finished_emitted = False
        self.state = "RUNNING"
        self.run_terminal_count = 0
        self.run_idle_transition_count = 0
        self.run_log_copy_count = 0
        return self.run_id

    def finish_once(self, run_id: int | None = None, *, terminal_state: str = "FINISHED") -> bool:
        effective_run_id = self.active_run_id if run_id is None else int(run_id)
        if effective_run_id is None:
            return False
        if effective_run_id != self.active_run_id:
            return False
        if effective_run_id in self._completed_run_ids:
            return False
        if self.finished_emitted:
            return False
        self._completed_run_ids.add(effective_run_id)
        self.running = False
        self.finished_emitted = True
        self.terminal_count += 1
        self.run_terminal_count += 1
        self.state = str(terminal_state or "FINISHED")
        return True

    def mark_log_copy_once(self, run_id: int | None = None) -> bool:
        effective_run_id = self.active_run_id if run_id is None else int(run_id)
        if effective_run_id is None:
            return False
        if effective_run_id in self._log_copied_run_ids:
            return False
        self._log_copied_run_ids.add(effective_run_id)
        self.log_copy_count += 1
        self.run_log_copy_count += 1
        return True

    def transition_idle_once(self, run_id: int | None = None) -> bool:
        effective_run_id = self.active_run_id if run_id is None else int(run_id)
        if effective_run_id is None:
            return False
        if effective_run_id not in self._completed_run_ids:
            return False
        if self.state == "IDLE":
            return False
        self.state = "IDLE"
        self.active_run_id = None
        self.idle_transition_count += 1
        self.run_idle_transition_count += 1
        return True

    def reset(self) -> None:
        self.running = False
        self.finished_emitted = False
        self.active_run_id = None
        self.state = "IDLE"
        self.run_terminal_count = 0
        self.run_idle_transition_count = 0
        self.run_log_copy_count = 0
