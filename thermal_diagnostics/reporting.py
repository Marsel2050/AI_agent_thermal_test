from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict

from .models import DiagnosticInputs, DiagnosticResult


class GeneratedText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technical_rationale: str
    recommendation: str


def enhance_report(
    result: DiagnosticResult,
    inputs: DiagnosticInputs,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[DiagnosticResult, str | None]:
    """Optionally rewrite two narrative fields without changing the diagnosis.

    All safety-relevant fields come from the deterministic engine. If Groq is
    unavailable or rejects the request, the local report remains intact.
    """

    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        return result, None

    selected_model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    schema = GeneratedText.model_json_schema()
    payload = {
        "model": selected_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты редактор технических отчётов. Не изменяй статус, числа или "
                    "срочность. Не выдумывай причину дефекта. Верни только краткое "
                    "проверяемое обоснование и рекомендацию."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Статус: {result.status.value}\n"
                    f"Исходное обоснование: {result.technical_rationale}\n"
                    f"Исходная рекомендация: {result.recommendation}\n"
                    f"Температуры: контакт {inputs.t_contact} °C, "
                    f"эталон {inputs.t_reference} °C, воздух {inputs.t_ambient} °C."
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "diagnostic_narrative",
                "strict": True,
                "schema": schema,
            },
        },
        "temperature": 0,
    }
    proxy = os.getenv("GROQ_PROXY_URL") or None
    try:
        import httpx

        with httpx.Client(proxy=proxy, timeout=30.0) as client:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]["content"]
        generated = GeneratedText.model_validate_json(message)
        return result.model_copy(
            update={
                "technical_rationale": generated.technical_rationale,
                "recommendation": generated.recommendation,
            }
        ), None
    except Exception as exc:  # local deterministic report is the safe fallback
        return result, f"LLM недоступна: {exc}"
