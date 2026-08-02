from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class AssessmentStatus(StrEnum):
    NOT_ASSESSABLE = "Недостаточно данных"
    INCONCLUSIVE = "Требует повторной проверки"
    NORMAL = "Норма"
    MAINTENANCE = "Требует ТО"
    PRE_EMERGENCY = "Предаварийное"
    EMERGENCY = "Аварийное"


class DiagnosticInputs(BaseModel):
    t_contact: float = Field(description="Температура контакта, °C")
    t_reference: float = Field(description="Температура исправного эталонного участка, °C")
    t_ambient: float = Field(description="Температура воздуха, °C")
    i_actual: float = Field(ge=0, description="Фактический ток, A")
    i_nominal: float = Field(gt=0, description="Номинальный ток, A")
    current_exponent: float = Field(default=2.0, ge=1.0, le=2.5)
    temperature_uncertainty: float = Field(default=2.0, gt=0, le=20)
    max_permissible_rise: float | None = Field(default=None, gt=0)
    switching_cycles: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_temperature_range(self) -> "DiagnosticInputs":
        for value in (self.t_contact, self.t_reference, self.t_ambient):
            if not -80 <= value <= 300:
                raise ValueError("Темпера выходит за допустимый диапазон -80…300 °C")
        return self


class DiagnosticResult(BaseModel):
    status: AssessmentStatus
    assessable: bool
    requires_human_approval: bool
    k_defect: float | None = None
    k_defect_ci_low: float | None = None
    k_defect_ci_high: float | None = None
    contact_rise: float | None = None
    reference_rise: float | None = None
    normalized_contact_rise: float | None = None
    defect_type: str
    probable_causes: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    technical_rationale: str
    recommendation: str
