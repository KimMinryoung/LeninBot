"""Shared translation execution; language/format policy belongs to adapters."""
from __future__ import annotations

class TranslationCallError(RuntimeError):
    """제공자가 실제로 무엇을 돌려줬는지 함께 들고 다니는 오류.

    예전에는 빈 응답이 json.loads에서 "Expecting value: line 1 column 1"으로
    터졌다. 그 문장만 보고는 모델이 거부한 것인지, 응답이 잘린 것인지, 아예
    비어서 온 것인지 구분할 수 없어서 로그를 봐도 손을 못 댔다.
    """


class TranslationProviderError(TranslationCallError):
    def __init__(self, result):
        self.result = result
        super().__init__(f"{result.error_kind or 'truncated'}: {result.error or 'incomplete output'}")


def generate_translation(feature: str, prompt: str, *, system: str, on_result=None,
                         cancelled=None, label: str | None = None) -> str:
    """Retry transient transport failures only; validation retries belong to callers."""
    import random
    import time
    from llm.call_registry import generate_detailed

    for attempt in range(3):
        if cancelled and cancelled():
            raise TranslationCallError("translation run stopped after a permanent provider error")
        result = generate_detailed(feature, prompt, system=system, label=label)
        if on_result:
            on_result(result)
        if result.text and not result.truncated and not result.error_kind:
            return result.text
        if not result.retryable or attempt == 2:
            raise TranslationProviderError(result)
        time.sleep(min(60, max(0, result.retry_after or (2 ** attempt + random.random()))))
    raise AssertionError("unreachable")


def validate_cached(candidate, validator):
    if candidate is None:
        return None, []
    problems = validator(candidate)
    return (None, problems) if problems else (candidate, [])


def default_correction(problems: list[str]) -> str:
    return ("\n\nPrevious translation failed validation:\n"
            + "\n".join("- " + p for p in problems)
            + "\nTranslate the same source again, fixing these issues. "
              "Preserve every protected placeholder exactly.")


def translate_validated(*, generate, parse, validate, attempts=2, cached=None,
                        store=None, on_invalid=None, correction=default_correction):
    """Shared cache → generation → parsing → validation → correction loop.

    Only parse/validation failures trigger a corrected translation. Provider
    failures propagate, with transport retries handled by generate_translation.
    Callbacks keep archival markers and site markup policy outside the engine;
    `correction` renders the retry instruction in the adapter's own wording.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    accepted, problems = validate_cached(cached, validate)
    if accepted is not None:
        return accepted
    render = correction
    correction = ""
    for attempt in range(1, attempts + 1):
        raw = generate(correction)
        try:
            value = parse(raw)
            problems = validate(value)
        except (ValueError, TypeError, KeyError) as exc:
            problems = [str(exc)]
        if not problems:
            if store:
                store(value)
            return value
        if on_invalid:
            on_invalid(attempt, problems)
        correction = render(problems)
    raise TranslationCallError("validation failed: " + "; ".join(problems))
