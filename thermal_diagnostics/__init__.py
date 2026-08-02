"""Core diagnostics package for electrical contact thermography."""

from .models import AssessmentStatus, DiagnosticInputs, DiagnosticResult
from .physics import evaluate_contact

__all__ = ["AssessmentStatus", "DiagnosticInputs", "DiagnosticResult", "evaluate_contact"]
