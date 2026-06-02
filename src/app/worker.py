from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from src.pipeline.pipeline_factory import get_pipeline


class PipelineWorker(QThread):
    progress_changed = pyqtSignal(int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, pipeline_name: str, config: dict, video_path: str, parent=None) -> None:
        super().__init__(parent)
        self.pipeline_name = pipeline_name
        self.config = config
        self.video_path = video_path

    def run(self) -> None:
        pipeline = None
        try:
            pipeline = get_pipeline(self.pipeline_name)
            if hasattr(pipeline, "set_progress_callback"):
                pipeline.set_progress_callback(self._on_progress)
            pipeline.load(self.config)
            result = pipeline.run(self.video_path)
            self.succeeded.emit(result)
        except Exception as exc:
            details = "".join(traceback.format_exception(exc))
            self.failed.emit(details)
        finally:
            if pipeline is not None:
                try:
                    pipeline.cleanup()
                except Exception:
                    pass

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_changed.emit(percent, message)
