"""
Base pipeline classes and utilities.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from ..config import get_config


class PipelineStep(ABC):
    """
    Base class for a pipeline step.

    Each step has:
    - A name
    - Input requirements
    - Execution logic
    - Output artifacts
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"pipeline.{name}")
        self.start_time = None
        self.end_time = None
        self.status = "pending"
        self.error = None

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline step.

        Parameters
        ----------
        context : dict
            Shared context containing outputs from previous steps

        Returns
        -------
        dict
            Updated context with this step's outputs
        """
        pass

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the step with timing and error handling.

        Parameters
        ----------
        context : dict
            Shared context

        Returns
        -------
        dict
            Updated context
        """
        self.logger.info(f"Starting step: {self.name}")
        self.start_time = datetime.now()
        self.status = "running"

        try:
            context = self.run(context)
            self.status = "completed"
            self.logger.info(f"✓ Completed step: {self.name}")

        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            self.logger.error(f"✗ Failed step: {self.name} - {e}")
            raise

        finally:
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            self.logger.info(f"  Duration: {duration:.1f}s")

        return context

    @property
    def duration(self) -> Optional[float]:
        """Get step duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class Pipeline:
    """
    Base pipeline that orchestrates multiple steps.

    Features:
    - Sequential step execution
    - Shared context between steps
    - Error handling and logging
    - Summary reporting

    Parameters
    ----------
    name : str
        Pipeline name
    steps : list of PipelineStep
        Ordered list of pipeline steps
    """

    def __init__(self, name: str, steps: List[PipelineStep]):
        self.name = name
        self.steps = steps
        self.logger = logging.getLogger(f"pipeline.{name}")
        self.config = get_config()
        self.start_time = None
        self.end_time = None
        self.context = {}

    def run(self, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Parameters
        ----------
        initial_context : dict, optional
            Initial context values

        Returns
        -------
        dict
            Final context after all steps
        """
        self.logger.info("=" * 70)
        self.logger.info(f"PIPELINE: {self.name}")
        self.logger.info("=" * 70)
        self.logger.info(f"Steps: {len(self.steps)}")
        self.logger.info(f"Project root: {self.config.project_root}")
        self.logger.info("")

        self.start_time = datetime.now()

        # Initialize context
        self.context = initial_context or {}
        self.context['config'] = self.config
        self.context['pipeline_name'] = self.name
        self.context['start_time'] = self.start_time

        # Execute steps sequentially
        for i, step in enumerate(self.steps, 1):
            self.logger.info(f"[{i}/{len(self.steps)}] {step.name}")
            self.context = step.execute(self.context)
            self.logger.info("")

        self.end_time = datetime.now()

        # Print summary
        self._print_summary()

        return self.context

    def _print_summary(self):
        """Print pipeline execution summary."""
        total_duration = (self.end_time - self.start_time).total_seconds()

        self.logger.info("=" * 70)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 70)

        # Step summaries
        for i, step in enumerate(self.steps, 1):
            status_icon = "✓" if step.status == "completed" else "✗"
            duration = f"{step.duration:.1f}s" if step.duration else "N/A"
            self.logger.info(f"  [{i}] {status_icon} {step.name}: {step.status} ({duration})")
            if step.error:
                self.logger.info(f"      Error: {step.error}")

        self.logger.info("")
        self.logger.info(f"Total duration: {total_duration:.1f}s")

        # Check if all succeeded
        all_completed = all(step.status == "completed" for step in self.steps)
        if all_completed:
            self.logger.info("Status: ✓ All steps completed successfully")
        else:
            failed_steps = [s.name for s in self.steps if s.status == "failed"]
            self.logger.error(f"Status: ✗ {len(failed_steps)} step(s) failed: {', '.join(failed_steps)}")

        self.logger.info("=" * 70)

    @property
    def duration(self) -> Optional[float]:
        """Get total pipeline duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def get_status(self) -> Dict[str, Any]:
        """
        Get pipeline status.

        Returns
        -------
        dict
            Status information
        """
        return {
            'name': self.name,
            'total_steps': len(self.steps),
            'completed_steps': sum(1 for s in self.steps if s.status == "completed"),
            'failed_steps': sum(1 for s in self.steps if s.status == "failed"),
            'duration': self.duration,
            'steps': [
                {
                    'name': s.name,
                    'status': s.status,
                    'duration': s.duration,
                    'error': s.error
                }
                for s in self.steps
            ]
        }
