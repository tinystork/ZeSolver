from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RunLifecycle:
    running: bool = False
    finished_emitted: bool = False

    def start(self) -> None:
        if self.running:
            raise RuntimeError("gui_run_already_active")
        self.running = True
        self.finished_emitted = False

    def finish_once(self) -> bool:
        if self.finished_emitted:
            return False
        self.running = False
        self.finished_emitted = True
        return True

    def reset(self) -> None:
        self.running = False
        self.finished_emitted = False
