"""Forte Process Intelligence - Streamlit UI for the existing FastAPI job API.

Run: streamlit run app.py
Environment: FORTE_API_URL (optional), FORTE_API_KEY (optional).
"""

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
API_URL = os.getenv("FORTE_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ["FORTE_API_KEY"]
HEADERS = {"X-API-Key": API_KEY}
POLL_INTERVAL_SECONDS = 2
MAX_WAIT_SECONDS = 295

EVENT_NAMES = {
    "CASHIER_PRESENT": "Кассир в рабочей зоне",
    "CUSTOMER_PRESENT": "Клиент в кадре",
    "HANDOVER_EVENT": "Передача объекта",
    "CASH_ON_DESK_PROPOSAL": "Возможные наличные на столе",
    "CASHIER_AWAY_FROM_WORKSTATION": "Кассир отошёл от рабочего места",
    "CASHIER_NOT_VISIBLE": "Кассир вне кадра",
    "CASH_PERSISTENCE_CANDIDATE": "Наличные, вероятно, остались на столе",
    "MULTIPLE_CUSTOMERS_PRESENT": "Несколько клиентов в кадре",
    "CASHIER_AWAY_WHILE_CUSTOMER_PRESENT": "Кассир отошёл при присутствии клиента",
    "BAG_STATIONARY_PROPOSAL": "Сумка на рабочем месте (предположение модели)",
    "FOOD_DRINK_OBJECT_PROPOSAL": "Напиток/еда на рабочем месте (предположение модели)",
    "PERSONAL_DEVICE_PROPOSAL": "Телефон в зоне сотрудника (предположение модели)",
    "WORKSTATION_UNATTENDED_PROPOSAL": "Рабочее место без сотрудника (экран не проверен)",
}

PRIMARY_EVENTS = {
    "CASH_ON_DESK_PROPOSAL",
    "CASHIER_AWAY_FROM_WORKSTATION",
    "CASH_PERSISTENCE_CANDIDATE",
    "HANDOVER_EVENT",
    "CASHIER_NOT_VISIBLE",
    "BAG_STATIONARY_PROPOSAL",
    "FOOD_DRINK_OBJECT_PROPOSAL",
    "PERSONAL_DEVICE_PROPOSAL",
    "WORKSTATION_UNATTENDED_PROPOSAL",
}


# ------------------------------------------------------------
# DATA HELPERS
# ------------------------------------------------------------

def seconds(value):
    try:
        return f"{float(value):.2f} сек"
    except (TypeError, ValueError):
        return "—"


def url_path(path):
    """Extract a relative media path without permitting an external host."""
    if not isinstance(path, str) or not path.strip():
        return None

    path = path.strip()
    if path.startswith(("http://", "https://")):
        parsed = urlparse(path)
        expected = urlparse(API_URL)
        if (parsed.hostname, parsed.port) != (expected.hostname, expected.port):
            return None
        path = parsed.path

    # API already %-encodes some names (e.g. Cyrillic Х -> %D0%A5).
    # Decode ONCE before safe path validation, local lookup, and URL quoting.
    path = unquote(path)
    path = path.replace("\\", "/").lstrip("/")
    if path.startswith("evidence/"):
        path = path[len("evidence/"):]

    if path.startswith("data/"):
        path = path[len("data/"):]

    parts = Path(path).parts
    if not parts or any(part == ".." for part in parts):
        return None
    if not parts[0] in ("cash_auto_results", "handover_auto_results", "presence_results"):
        return None

    return "/".join(parts)


def load_evidence(path):
    """Prefer the local data directory; otherwise use the API static mount.

    API and Streamlit run on the same machine in the hackathon setup.
    Static serving can differ by deployment, so the URL is fallback only.
    """
    relative = url_path(path)
    if relative is None:
        return None

    data_root = (PROJECT_ROOT / "data").resolve()
    local_file = (data_root / relative).resolve()

    try:
        local_file.relative_to(data_root)  # prevent escaping local data
        if local_file.is_file():
            return local_file.read_bytes()
    except (ValueError, OSError):
        pass

    safe_path = quote(relative, safe="/")
    for endpoint in (
        f"{API_URL}/evidence/{safe_path}",
        f"{API_URL}/evidence/data/{safe_path}",
    ):
        try:
            response = requests.get(endpoint, timeout=5)
            if (response.ok and
                    response.headers.get("Content-Type", "").lower().startswith("image/")):
                return response.content
        except requests.RequestException:
            continue
    return None


def show_evidence(paths, maximum=3):
    if not paths:
        st.info("Кадры-доказательства не приложены.")
        return

    unique = list(dict.fromkeys(p for p in paths if p))[:maximum]
    if not unique:
        return

    cols = st.columns(len(unique))
    for col, path in zip(cols, unique):
        with col:
            image_bytes = load_evidence(path)
            if image_bytes is not None:
                st.image(image_bytes, caption=Path(str(path)).name,
                         use_container_width=True)
            else:
                st.warning("Не удалось открыть кадр")
                st.caption(str(path))


def show_norms(norms):
    if not norms:
        st.caption("Проверенная нормативная ссылка не указана.")
        return

    st.markdown("**Нормативные основания из каталога правил**")
    for entry in norms:
        short = entry.get("short") or entry.get("title") or "Нормативный документ"
        clause = entry.get("clause")
        title = f"{short} — {clause}" if clause else short
        url = str(entry.get("url") or "")
        if url.startswith(("https://", "http://")):
            st.markdown(f"- [{title}]({url})")
        else:
            st.markdown(f"- {title}")
    st.caption("Ссылки отображаются по правилам проекта; применимость пунктов требует юридической проверки.")


def event_row(event):
    kind = event.get("type", "EVENT")
    return {
        "Время": seconds(event.get("timestamp_sec", event.get("start_sec"))),
        "Событие": EVENT_NAMES.get(kind, kind),
        "Источник": event.get("source", "—"),
    }


def show_report(result):
    st.divider()
    st.header("Результат анализа")

    verdict = result.get("verdict", "unknown")
    violations = result.get("violations") or []
    signals = result.get("signals") or []
    rule_candidates = result.get("rule_candidates") or []
    candidate_rule_ids = sorted({str(c.get("rule_id")) for c in rule_candidates
                                 if c.get("rule_id")})
    object_candidates = [c for c in rule_candidates if c.get("rule_id") in ("COL-03", "WR-02", "WR-03", "WS-01")
                         and c.get("candidate_only") is True]
    operation = result.get("operation_context") or {}
    applicability = result.get("rule_applicability") or {}
    analysis_status = result.get("analysis_status", "unknown")
    requires_review = bool(result.get("requires_review")) or any(
        bool(v.get("requires_review")) for v in violations
    )

    if verdict == "violation":
        st.error("⚠️ AI обнаружил признаки нарушения кассовой дисциплины")
    elif verdict == "no_violation" and rule_candidates:
        st.warning("⚠️ Автоматических нарушений не подтверждено, но есть признаки для ручной проверки")
        st.caption("Бинарный API-вердикт остаётся no_violation. Кандидаты ниже не являются подтверждёнными нарушениями.")
    elif verdict == "no_violation":
        st.success("Нарушений по реализованным автоматическим правилам не обнаружено")
    else:
        st.warning(f"Неопределённый вердикт: {verdict}")

    columns = st.columns(5)
    # UI presentation only: the API binary verdict remains unchanged.
    if verdict == "violation":
        display_status = "AI ALERT"
    elif verdict == "no_violation" and candidate_rule_ids:
        display_status = "REVIEW"
    elif verdict == "no_violation" and analysis_status == "partial":
        display_status = "PARTIAL"
    elif verdict == "no_violation":
        display_status = "NO ALERT"
    else:
        display_status = "UNKNOWN"
    columns[0].metric("Результат проверки", display_status)
    st.caption(f"Бинарный ответ API (без изменений): `{verdict}`. "
               "REVIEW означает только кандидатов на ручную проверку.")
    columns[1].metric("Время анализа", seconds(result.get("processing_time_sec")))
    columns[2].metric("Событий", len(signals))
    columns[3].metric("AI-оповещений", len(violations))
    columns[4].metric("Типов для review", len(candidate_rule_ids))
    if candidate_rule_ids:
        st.caption("Предложенные правила (НЕ вердикты): " + ", ".join(candidate_rule_ids))

    if analysis_status == "partial":
        st.warning("Анализ частичный: некоторые детекторы пропущены или завершились с ограничениями.")
    if requires_review:
        st.info("**Требует проверки специалистом.** AI-оповещение не является юридически установленным нарушением.")
    for warning in result.get("analysis_warnings") or []:
        st.caption(f"⚠️ {warning}")

    st.subheader("Контекст операции и применимость правил")
    observed = operation.get("observed_activity") or "не определён"
    op_type = operation.get("operation_type") or "unknown"
    st.write(f"**Визуальный контекст:** `{observed}` · **Тип операции:** `{op_type}`")
    if op_type == "unknown":
        st.caption("Направление передачи и точный тип операции не подтверждены визуально. Система не подменяет неизвестное предположением.")
    screened = applicability.get("actually_screened_rule_ids") or []
    st.caption(f"Автоматически проверены: {', '.join(map(str, screened)) or 'нет'} · "
               f"Каталог правил: {applicability.get('catalog_rule_count', '—')}. "
               "Наличие правила в каталоге не означает, что оно проверено.")

    # Core innovation: show the reconstructed chain before raw telemetry.
    st.subheader("Evidence Timeline — почему AI принял решение")
    selected = [event for event in signals if event.get("type") in PRIMARY_EVENTS]
    selected.sort(key=lambda e: float(e.get("timestamp_sec") or e.get("start_sec") or 0))
    if selected:
        for event in selected:
            name = EVENT_NAMES.get(event.get("type"), event.get("type", "Событие"))
            st.markdown(f"**{seconds(event.get('timestamp_sec', event.get('start_sec')))}** · {name}")
            if event.get("type") == "CASH_ON_DESK_PROPOSAL":
                conf = event.get("cash_confidence")
                if isinstance(conf, (int, float)):
                    st.caption(f"Предположение YOLO · оценка модели {conf:.3f}, не достоверность нарушения")
            elif event.get("type") == "CASH_PERSISTENCE_CANDIDATE":
                st.caption(f"Временные совпадения объекта: {event.get('matches', '—')}. Визуальное сходство не равно доказательству.")
            elif event.get("type") == "HANDOVER_EVENT":
                st.caption("Передача объекта сама по себе не считается нарушением.")
            elif event.get("type") == "BAG_STATIONARY_PROPOSAL":
                st.caption("Объект категории сумка найден повторно. Тип сумки и процедура не установлены.")
            elif event.get("type") == "FOOD_DRINK_OBJECT_PROPOSAL":
                st.caption("Обнаружен возможный напиток или еда. Нарушение пока не подтверждено.")
            elif event.get("type") == "PERSONAL_DEVICE_PROPOSAL":
                st.caption("Возможное мобильное устройство; зона кассы и статус устройства требуют проверки.")
            elif event.get("type") == "WORKSTATION_UNATTENDED_PROPOSAL":
                st.caption("Оборудование видно во время отсутствия сотрудника; активность экрана не определена.")
    else:
        st.info("Для данного видео цепочка значимых событий не сформирована.")

    st.subheader("Найденные нарушения и доказательства")
    if violations:
        for index, violation in enumerate(violations, 1):
            with st.container(border=True):
                st.markdown(f"### {index}. {violation.get('type', 'Нарушение')}")
                st.write(
                    f"**Правило:** {violation.get('rule_id', '—')}  |  "
                    f"**Время:** {seconds(violation.get('timestamp_sec'))}  |  "
                    f"**Класс:** {violation.get('subtype', '—')}"
                )
                if violation.get("description"):
                    st.write(violation["description"])
                if violation.get("requires_review"):
                    st.caption("Статус: AI ALERT / REQUIRES REVIEW")
                st.markdown("**Кадры-доказательства**")
                show_evidence(violation.get("evidence_frames") or [], maximum=3)
                show_norms(violation.get("normative_sources") or [])
    else:
        st.info("Подтверждённых автоматическими правилами нарушений нет. Это не гарантия соблюдения всех процедур.")

    st.subheader("Очередь проверки — кандидаты, не вердикты")
    if object_candidates:
        st.warning("Эти объекты автоматически выявлены по видео, но **не доказывают нарушение**. "
                   "Нужна проверка типа объекта, операции, применимости внутренних правил и контекста.")
        readable_missing = {
            "OFFICIAL_COLLECTION_BAG_IDENTIFICATION": "Подтвердить, что это официальная инкассаторская сумка",
            "COLLECTION_OPERATION_CONFIRMED": "Подтвердить, что происходит инкассация",
            "BAG_CUSTODY_VERIFICATION": "Проверить, кто отвечает за сумку и оставалась ли она без надзора",
            "CASH_WORKSPACE_PROXIMITY": "Подтвердить расположение в зоне работы с наличными",
            "INTERNAL_BANK_POLICY_VALIDATION": "Проверить внутренний регламент банка",
            "DEVICE_IDENTITY_AND_POLICY": "Подтвердить, что это именно личный телефон и что его использование запрещено",
            "WORKSTATION_SCREEN_ACTIVE_UNLOCKED": "Подтвердить, что экран был активен и сессия не заблокирована",
            "CASHIER_IDENTITY_AND_ROLE": "Подтвердить, что сотрудник действительно оставил рабочую станцию",
        }
        for i, candidate in enumerate(object_candidates, 1):
            with st.container(border=True):
                rule_id = candidate.get("rule_id")
                review_titles = {
                    "COL-03": "Возможная сумка без присмотра — тип не подтверждён",
                    "WR-02": "Возможный напиток на рабочем месте — контекст не подтверждён",
                    "WR-03": "Возможный телефон у кассы — требует проверки",
                    "WS-01": "Рабочая станция без сотрудника — статус экрана неизвестен",
                }
                title = review_titles.get(rule_id, candidate.get("type") or candidate.get("subtype"))
                st.markdown(f"**{i}. {rule_id} · {title}**")
                st.caption(f"Время: {seconds(candidate.get('timestamp_sec'))} · "
                           "Статус: REVIEW ONLY · модель объектов COCO")
                st.write(candidate.get("description") or "Предположение модели о наличии объекта.")
                st.markdown("**Что ещё требуется проверить:**")
                missing = candidate.get("missing_context") or []
                for missing_item in missing:
                    st.write("• " + readable_missing.get(missing_item, str(missing_item)))
                show_evidence(candidate.get("evidence_frames") or [], maximum=3)
                st.caption("Нормативные ссылки приведены для проверки применимости, а не как установленные основания нарушения.")
                show_norms(candidate.get("normative_sources") or [])
    else:
        st.caption("Кандидатов COL-03 / WR-02 / WR-03 / WS-01 для ручной проверки не выявлено.")

    st.subheader("Роли участников")
    roles = result.get("roles") or {}
    visible_roles = [(pid, role) for pid, role in roles.items() if role != "IGNORE"]
    if visible_roles:
        st.write(" · ".join(f"ID {pid}: **{role}**" for pid, role in visible_roles))
        st.caption("ID относятся к трекам людей, а не к установленным личностям; один человек может иметь несколько ID.")
    else:
        st.caption("Роли не определены.")

    with st.expander("Все события и диагностические кадры", expanded=False):
        if signals:
            st.dataframe([event_row(event) for event in signals],
                         use_container_width=True, hide_index=True)
            for index, event in enumerate(signals, 1):
                paths = event.get("evidence_frames") or []
                if paths:
                    with st.expander(f"{index}. {EVENT_NAMES.get(event.get('type'), event.get('type'))} — кадры"):
                        show_evidence(paths, maximum=3)
        else:
            st.caption("События не обнаружены.")

    with st.expander("Технический JSON", expanded=False):
        st.json(result)
    st.download_button(
        "Скачать JSON-отчёт",
        data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{Path(result.get('video') or 'video').stem}_analysis.json",
        mime="application/json",
        use_container_width=False,
    )


# ------------------------------------------------------------
# APP
# ------------------------------------------------------------

st.set_page_config(page_title="Forte Process Intelligence", page_icon="🏦", layout="wide")
st.title("Forte Process Intelligence")
st.caption("Explainable AI Compliance Auditor · анализ кассовых операций по видео")

uploaded_file = st.file_uploader("Загрузить видео в формате .mov", type=["mov"])

if uploaded_file is None:
    st.info("Загрузите видео, затем нажмите «Анализировать».")
    st.stop()

video_bytes = uploaded_file.getvalue()
file_fingerprint = hashlib.sha256(video_bytes).hexdigest()

if st.session_state.get("video_fingerprint") != file_fingerprint:
    st.session_state.video_fingerprint = file_fingerprint
    st.session_state.analysis_result = None
    st.session_state.active_job_id = None

st.video(video_bytes)

if st.button("Анализировать видео", type="primary", use_container_width=True):
    st.session_state.analysis_result = None
    st.session_state.active_job_id = None

    try:
        with st.spinner("Загрузка видео в API..."):
            response = requests.post(
                f"{API_URL}/api/v1/analyze",
                headers=HEADERS,
                files={"file": (uploaded_file.name, video_bytes, "video/quicktime")},
                timeout=90,
            )
            response.raise_for_status()
            job_id = response.json().get("job_id")
            if not job_id:
                raise ValueError("API не вернул job_id")
            st.session_state.active_job_id = job_id
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Не удалось начать анализ: {exc}")


if st.session_state.get("active_job_id") and st.session_state.get("analysis_result") is None:
    job_id = st.session_state.active_job_id
    progress = st.progress(0)
    status = st.empty()
    started = time.monotonic()

    while True:
        elapsed = time.monotonic() - started
        if elapsed >= MAX_WAIT_SECONDS:
            progress.empty()
            status.warning("Ожидание превысило 295 секунд. Задача остаётся в API; можно проверить её по Job ID.")
            st.code(job_id)
            break

        try:
            response = requests.get(
                f"{API_URL}/api/v1/jobs/{job_id}",
                headers=HEADERS,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            status.error(f"Ошибка при проверке задачи {job_id}: {exc}")
            break

        state = payload.get("status", "unknown")
        if state == "done":
            result = payload.get("result")
            if not isinstance(result, dict):
                status.error("API завершил задачу, но не вернул корректный JSON.")
                break
            st.session_state.analysis_result = result
            progress.progress(100)
            status.success("Анализ завершён")
            break
        if state == "failed":
            progress.empty()
            status.error(payload.get("error") or "Ошибка анализа на сервере.")
            break
        if state not in ("processing", "queued", "pending"):
            status.error(f"Неизвестный статус задачи: {state}")
            break

        progress.progress(min(90, int(elapsed / MAX_WAIT_SECONDS * 90)))
        status.info(f"AI анализирует видео: {int(elapsed)} сек. · Job ID: {job_id}")
        time.sleep(POLL_INTERVAL_SECONDS)


result = st.session_state.get("analysis_result")
if isinstance(result, dict):
    show_report(result)
