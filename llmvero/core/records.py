"""Turn raw event rows (e.g. logins: user_id + date_time) into exact fact sheets.

Language models can't count thousands of rows reliably, so the numbers are computed here in Python.
The model then only writes questions and answers from these facts, so the training data is correct.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

PARTS_OF_DAY = [("night (00:00-05:59)", 0, 6), ("morning (06:00-11:59)", 6, 12),
                ("afternoon (12:00-17:59)", 12, 18), ("evening (18:00-23:59)", 18, 24)]  # fmt: skip


class RecordError(ValueError):
    pass


def parse_time(value: Any) -> datetime:
    """Accept '2026-11-07 09:15:00', ISO 8601 (with T, Z or an offset) or a Unix timestamp."""
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e11 else value
        return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty time")
    text = value.strip().replace("Z", "+00:00")
    # Keep the clock time as written: 09:15 in the file is 09:15 in the facts.
    return datetime.fromisoformat(text).replace(tzinfo=None)


def _hour(h: int) -> str:
    return f"{h:02d}:00-{h:02d}:59"


def _t(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def _pct(n: int, total: int) -> str:
    return f"{n / total * 100:.0f}%"


def _users(ids: list[str], limit: int = 30) -> str:
    shown = ", ".join(ids[:limit])
    return shown + (f" and {len(ids) - limit} more" if len(ids) > limit else "")


# Common names for "who" and "when", compared without case, underscores or dashes.
USER_KEYS = ["userid", "user", "uid", "username", "useremail", "email", "accountid", "account", "memberid",
             "customerid", "employeeid", "personid", "login", "useridentifier", "user.id", "user.username",
             "user.email", "user.name"]  # fmt: skip
TIME_KEYS = ["datetime", "timestamp", "time", "createdat", "loggedinat", "loginat", "logintime", "lastlogin",
             "eventtime", "occurredat", "date", "ts", "at", "created", "when"]  # fmt: skip


def _norm(key: str) -> str:
    return re.sub(r"[\s_\-]", "", key.lower())


def _flatten(row: dict, prefix: str = "") -> dict:
    """{"user": {"id": 1}} -> {"user.id": 1}, so nested fields can be found and named."""
    out: dict = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def load_rows(raw: bytes | str) -> tuple[list[dict], str | None]:
    """Read rows from a JSON array, an object holding one (e.g. {"logins": [...]}), JSON Lines or CSV.

    Returns the rows and, when the data names them, what one row is (e.g. "login").
    """
    text = (raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw).strip()
    if not text:
        raise RecordError("the request body is empty")
    hint = None
    if text[0] in "[{":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:  # JSON Lines: one object per line
                data = [json.loads(line) for line in text.splitlines() if line.strip()]
            except json.JSONDecodeError as exc:
                raise RecordError(f"not valid JSON or JSON Lines: {exc}") from exc
        if isinstance(data, dict):
            lists = [(k, v) for k, v in data.items() if isinstance(v, list) and v and isinstance(v[0], dict)]
            if lists:
                key, data = max(lists, key=lambda kv: len(kv[1]))
                hint = re.sub(r"s$", "", key.replace("_", " ").strip().lower()) or None
            else:
                data = [data]
        rows = data
    else:
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    if not isinstance(rows, list) or not rows:
        raise RecordError("found no rows")
    if not all(isinstance(r, dict) for r in rows):
        raise RecordError('every row must be an object like {"user_id": 1, "date_time": "..."}')
    return rows, hint


def _looks_like_time(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 1e8 < value < 1e14  # Unix seconds or milliseconds
    if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}", value.strip()):
        try:
            parse_time(value)
            return True
        except ValueError:
            return False
    return False


def detect_fields(rows: list[dict]) -> tuple[str, str]:
    """Guess which field is the user and which is the time, from names first, then from values."""
    sample = rows[:50]
    keys: list[str] = []
    for row in sample:
        keys += [k for k in row if k not in keys]
    by_norm = {_norm(k): k for k in keys}

    time_field = next((by_norm[n] for n in TIME_KEYS if n in by_norm), None)
    if time_field is None:
        time_field = next(
            (k for k in keys if all(_looks_like_time(r.get(k)) for r in sample if k in r)), None
        )
    user_field = next((by_norm[n] for n in USER_KEYS if n in by_norm and by_norm[n] != time_field), None)
    if user_field is None:
        user_field = next((k for k in keys if "user" in k.lower() and k != time_field), None)
    if not time_field or not user_field:
        raise RecordError(
            f"couldn't tell which field is the {'user' if not user_field else 'time'}. Fields found: "
            f"{', '.join(keys)}. Name them with ?user_field=...&time_field=..."
        )
    return user_field, time_field


def _event_from(time_field: str) -> str | None:
    return "login" if re.search(r"log(ged)?_?in", time_field.lower()) else None


@dataclass
class Analysis:
    """Fact sheets as text (sent to the model) and the same numbers as data (the tool results)."""

    event: str
    date_from: str
    date_to: str
    sections: list[tuple[str, str]]
    summary: dict
    hourly: dict
    top_users: dict
    daily: dict
    users: dict[str, dict] = field(default_factory=dict)

    def tool_call_for(self, section: str, question: str) -> tuple[str, dict, dict] | None:
        """The tool call that answers a question generated from `section`: (name, arguments, result)."""
        period = {"date_from": self.date_from, "date_to": self.date_to}
        simple = {
            "overview": ("get_activity_summary", self.summary),
            "hourly": ("get_hourly_counts", self.hourly),
            "users": ("get_top_users", self.top_users),
            "daily": ("get_daily_counts", self.daily),
        }
        if section in simple:
            name, result = simple[section]
            return name, period, result
        if section.startswith("user_profiles"):
            named = {m.lower() for m in _USER_MENTION.findall(question)}
            if len(named) != 1:
                return None
            wanted = named.pop()
            user = next((u for u in self.users if u.lower() == wanted), None)
            if user is None:
                return None
            return "get_user_activity", {"user_id": user, **period}, self.users[user]
        return None


def analyze_records(
    records: list[dict],
    *,
    user_field: str | None = None,
    time_field: str | None = None,
    event: str | None = None,
    max_user_profiles: int = 50,
) -> Analysis:
    """Compute the facts from the rows, as text sheets and as data.

    Field names are detected when not given; `event` (e.g. "login") only changes the wording.
    """
    if not records:
        raise RecordError("no rows")
    if not all(isinstance(r, dict) for r in records):
        raise RecordError("every row must be an object")
    records = [_flatten(r) for r in records]
    if not user_field or not time_field:
        found_user, found_time = detect_fields(records)
        user_field, time_field = user_field or found_user, time_field or found_time
    event = (event or _event_from(time_field) or "event").strip().lower()
    events: list[tuple[str, datetime]] = []
    for i, row in enumerate(records):
        if row.get(user_field) in (None, "") or row.get(time_field) in (None, ""):
            raise RecordError(f"row {i} needs '{user_field}' and '{time_field}'")
        try:
            events.append((str(row[user_field]), parse_time(row[time_field])))
        except (ValueError, OSError, OverflowError) as exc:
            raise RecordError(f"row {i}: can't read {time_field} {row[time_field]!r} ({exc})") from exc
    events.sort(key=lambda e: e[1])

    total = len(events)
    plural = f"{event}s"
    by_user: dict[str, list[datetime]] = defaultdict(list)
    for user, dt in events:
        by_user[user].append(dt)
    users_ranked = sorted(by_user, key=lambda u: (-len(by_user[u]), by_user[u][0]))
    per_hour = Counter(dt.hour for _, dt in events)
    users_per_hour: dict[int, set] = defaultdict(set)
    for user, dt in events:
        users_per_hour[dt.hour].add(user)
    days = sorted({dt.date() for _, dt in events})
    first_day, last_day = days[0], days[-1]
    period = (
        f"on {first_day:%A, %Y-%m-%d}"
        if first_day == last_day
        else f"from {first_day:%A, %Y-%m-%d} to {last_day:%A, %Y-%m-%d} ({(last_day - first_day).days + 1} days)"
    )
    title = f"{event.capitalize()} activity {period}"

    # Busiest 15-minute window, sliding over the sorted times.
    times = [dt for _, dt in events]
    best_n, best_start, end = 0, times[0], 0
    for i, start in enumerate(times):
        end = max(end, i)
        while end < total and (times[end] - start).total_seconds() < 900:
            end += 1
        if end - i > best_n:
            best_n, best_start = end - i, start

    hours_ranked = sorted(per_hour, key=lambda h: (-per_hour[h], h))
    peak = hours_ranked[0]
    fewest = min(per_hour.values())
    quiet_hours = sorted(h for h in per_hour if per_hour[h] == fewest)
    zero_hours = [h for h in range(24) if h not in per_hour]
    first_user, first_dt = events[0]
    last_user, last_dt = events[-1]

    overview = [
        f"# {title}",
        "",
        f"- Total {plural}: {total}",
        f"- Unique users: {len(by_user)}",
        f"- Average {plural} per user: {total / len(by_user):.1f}",
        f"- Peak hour: {_hour(peak)} with {per_hour[peak]} {plural} ({_pct(per_hour[peak], total)} of the total) "
        f"by {len(users_per_hour[peak])} different users",
    ]
    if len(hours_ranked) > 1:
        rest = ", ".join(f"{_hour(h)} ({per_hour[h]})" for h in hours_ranked[1:3])
        overview.append(f"- Next busiest hours: {rest}")
    overview += [
        f"- Busiest 15 minutes: {_t(best_start)}-{_t_plus(best_start, 15)} with {best_n} {plural}",
        f"- Quietest {'hour' if len(quiet_hours) == 1 else 'hours'} with any activity: "
        f"{' and '.join(_hour(h) for h in quiet_hours)} ({fewest} {event if fewest == 1 else plural} each)",
        f"- Hours with no {plural}: {', '.join(_hour(h) for h in zero_hours) if zero_hours else 'none'}",
        f"- First {event}: {first_dt:%Y-%m-%d %H:%M} by user {first_user}",
        f"- Last {event}: {last_dt:%Y-%m-%d %H:%M} by user {last_user}",
    ]
    for name, lo, hi in PARTS_OF_DAY:
        n = sum(per_hour[h] for h in range(lo, hi))
        overview.append(f"- {plural.capitalize()} in the {name}: {n} ({_pct(n, total)})")

    span = {"event": event, "date_from": str(first_day), "date_to": str(last_day)}
    summary = {
        **span,
        "total": total,
        "unique_users": len(by_user),
        "average_per_user": round(total / len(by_user), 1),
        "peak_hour": {
            "hour": _hour(peak),
            "count": per_hour[peak],
            "users": len(users_per_hour[peak]),
            "share_percent": _share(per_hour[peak], total),
        },
        "next_busiest_hours": [{"hour": _hour(h), "count": per_hour[h]} for h in hours_ranked[1:3]],
        "busiest_15_minutes": {"from": _t(best_start), "to": _t_plus(best_start, 15), "count": best_n},
        "quietest_active_hours": {"hours": [_hour(h) for h in quiet_hours], "count_each": fewest},
        "hours_with_none": [_hour(h) for h in zero_hours],
        "first": {"time": f"{first_dt:%Y-%m-%d %H:%M}", "user_id": first_user},
        "last": {"time": f"{last_dt:%Y-%m-%d %H:%M}", "user_id": last_user},
        "parts_of_day": {
            name.split(" ")[0]: {
                "hours": name.split(" ")[1].strip("()"),
                "count": (n := sum(per_hour[h] for h in range(lo, hi))),
                "share_percent": _share(n, total),
            }
            for name, lo, hi in PARTS_OF_DAY
        },
    }
    hourly_data = {
        **span,
        "total": total,
        "hours": [
            {"hour": _hour(h), "count": per_hour[h], "users": len(users_per_hour[h])} for h in range(24)
        ],
    }

    hourly = [f"# {plural.capitalize()} per hour {period}", "", f"Total: {total} {plural}", ""]
    for h in range(24):
        if per_hour[h]:
            hourly.append(f"- {_hour(h)}: {per_hour[h]} {plural} by {len(users_per_hour[h])} users")
        else:
            hourly.append(f"- {_hour(h)}: no {plural}")

    once = [u for u in users_ranked if len(by_user[u]) == 1]
    top = users_ranked[:10]
    ranking = [f"# Most and least active users {period}", ""]
    for rank, u in enumerate(top, 1):
        ranking.append(
            f"{rank}. User {u}: {len(by_user[u])} {plural} ({_pct(len(by_user[u]), total)} of all)"
        )
    top_share = sum(len(by_user[u]) for u in top)
    ranking += [
        "",
        f"- The most active user is user {top[0]} with {len(by_user[top[0]])} {plural}.",
        f"- The top {len(top)} users made {top_share} of the {total} {plural} ({_pct(top_share, total)}).",
        f"- Users with only one {event} ({len(once)}): {_users(once) if once else 'none'}",
    ]

    top_data = {
        **span,
        "total": total,
        "unique_users": len(by_user),
        "top_users": [
            {
                "rank": rank,
                "user_id": u,
                "count": len(by_user[u]),
                "share_percent": _share(len(by_user[u]), total),
            }
            for rank, u in enumerate(top, 1)
        ],
        "top_10_total": top_share,
        "top_10_share_percent": _share(top_share, total),
        "users_with_one": once,
    }
    per_day = Counter(dt.date() for _, dt in events)
    busiest_day = max(days, key=lambda d: per_day[d])
    quietest_day = min(days, key=lambda d: per_day[d])
    daily_data = {
        **span,
        "days": [{"date": str(d), "weekday": f"{d:%A}", "count": per_day[d]} for d in days],
        "busiest_day": {"date": str(busiest_day), "count": per_day[busiest_day]},
        "quietest_day": {"date": str(quietest_day), "count": per_day[quietest_day]},
        "average_per_day": round(total / len(days), 1),
    }

    sections = [
        ("overview", "\n".join(overview)),
        ("hourly", "\n".join(hourly)),
        ("users", "\n".join(ranking)),
    ]

    if len(days) > 1:
        daily = [f"# {plural.capitalize()} per day {period}", ""]
        daily += [f"- {d:%A %Y-%m-%d}: {per_day[d]} {plural}" for d in days]
        daily += [
            "",
            f"- Busiest day: {busiest_day:%A %Y-%m-%d} ({per_day[busiest_day]} {plural})",
            f"- Quietest day: {quietest_day:%A %Y-%m-%d} ({per_day[quietest_day]} {plural})",
            f"- Average per day: {total / len(days):.1f}",
        ]
        sections.append(("daily", "\n".join(daily)))

    user_data: dict[str, dict] = {}
    for u in users_ranked:
        ts = by_user[u]
        hours = Counter(t.hour for t in ts)
        ranked = hours.most_common()
        clear = ranked[0][1] >= 2 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
        user_data[u] = {
            **span,
            "user_id": u,
            "count": len(ts),
            "first": f"{ts[0]:%Y-%m-%d %H:%M}",
            "last": f"{ts[-1]:%Y-%m-%d %H:%M}",
            "usual_hour": {"hour": _hour(ranked[0][0]), "count": ranked[0][1]} if clear else None,
            "distinct_hours": len(hours),
            "parts_of_day": [
                name.split(" ")[0] for name, lo, hi in PARTS_OF_DAY if any(lo <= t.hour < hi for t in ts)
            ],
            "days_active": len({t.date() for t in ts}),
        }

    profiled = users_ranked[:max_user_profiles]
    for start in range(0, len(profiled), 10):
        batch = profiled[start : start + 10]
        lines = [
            f"# User {event} habits {period}",
            "",
            f"This sheet covers only {len(batch)} of the {len(by_user)} users. Use it only for questions about one "
            f'named user (e.g. "When did user {batch[0]} first log in?"). Never use it to say who was first, '
            f"last, most active or active at a time of day overall: those facts are in the summary sheets.",
            "",
        ]
        for u in batch:
            ts = by_user[u]
            hours = Counter(t.hour for t in ts)
            ranked = hours.most_common()
            if ranked[0][1] >= 2 and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
                habit = f"usual hour {_hour(ranked[0][0])} ({ranked[0][1]} of {len(ts)})"
            elif len(ts) == 1:
                habit = f"only one {event}, at {ts[0]:%H:%M}"
            else:
                habit = f"no usual hour ({event}s spread across {len(hours)} different hours)"
            parts = [
                name.split(" ")[0] for name, lo, hi in PARTS_OF_DAY if any(lo <= t.hour < hi for t in ts)
            ]
            line = (
                f"- User {u}: {len(ts)} {event if len(ts) == 1 else plural}; first at {ts[0]:%Y-%m-%d %H:%M}, last at {ts[-1]:%Y-%m-%d %H:%M}; "
                f"{habit}; active in the {', '.join(parts)}"
            )
            if len(days) > 1:
                line += f"; active on {len({t.date() for t in ts})} of {len(days)} days"
            lines.append(line + ".")
        sections.append((f"user_profiles_{start // 10 + 1}", "\n".join(lines)))
    if len(users_ranked) > max_user_profiles:
        sections[-1] = (
            sections[-1][0],
            sections[-1][1]
            + f"\n\n{len(users_ranked) - max_user_profiles} less active users are not profiled.",
        )
    return Analysis(
        event=event,
        date_from=str(first_day),
        date_to=str(last_day),
        sections=sections,
        summary=summary,
        hourly=hourly_data,
        top_users=top_data,
        daily=daily_data,
        users=user_data,
    )


def summarize_records(records: list[dict], **kwargs: Any) -> list[tuple[str, str]]:
    """Just the text fact sheets: [(section, text), ...]."""
    return analyze_records(records, **kwargs).sections


def _share(n: int, total: int) -> int:
    return int(f"{n / total * 100:.0f}")


def tool_definitions(event: str = "event") -> list[dict]:
    """The functions a model fine-tuned with style=tool_calls learns to call. Implement them in your app."""
    plural = f"{event}s"
    dates = {
        "date_from": {"type": "string", "description": "First day, YYYY-MM-DD"},
        "date_to": {"type": "string", "description": "Last day, YYYY-MM-DD (same as date_from for one day)"},
    }
    specs = [
        (
            "get_activity_summary",
            f"Totals and peak times for {plural} in a date range: total, unique users, peak hour, busiest "
            f"15 minutes, quietest hours, first and last {event}, split by part of the day.",
            {},
        ),
        (
            "get_hourly_counts",
            f"Number of {plural} and distinct users for each hour of the day in a date range.",
            {},
        ),
        (
            "get_top_users",
            f"Most active users by number of {plural} in a date range, and users with only one {event}.",
            {},
        ),
        (
            "get_daily_counts",
            f"Number of {plural} per day in a date range, with the busiest and quietest day.",
            {},
        ),
        (
            "get_user_activity",
            f"One user's {plural} in a date range: count, first and last time, usual hour, parts of the day.",
            {"user_id": {"type": "string", "description": "The user's id"}},
        ),
    ]
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {**extra, **dates},
                    "required": [*extra, "date_from", "date_to"],
                    "additionalProperties": False,
                },
            },
        }
        for name, description, extra in specs
    ]


def _t_plus(dt: datetime, minutes: int) -> str:
    total = dt.hour * 60 + dt.minute + minutes
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"


DEFAULT_RECORD_INSTRUCTIONS = (
    "The document is a fact sheet computed from raw {event} records. Write questions an admin or analyst would "
    "ask about this activity: peak hours, how many {event}s, busiest times, most active users, a specific "
    "user's habits. Always name the date or period in the question or the answer. Answers must use the exact "
    "numbers from the fact sheet and never estimate or add numbers up. Don't ask about one specific user by id "
    '("how many times did user 7 log in?"): those questions are written from the user sheets.'
)

PROFILE_INSTRUCTIONS = (
    "This sheet describes a few individual users, not the whole {period}. Write exactly one question per user on "
    'the sheet. Each question names exactly one user by id (e.g. "When did user 7 first log in on 2026-11-06?") '
    "and asks only about that user's own {event}s. Never ask about several users together, totals, peak hours, "
    "or who was first, last or most active: this sheet can't answer those. Answers use the exact numbers from "
    "the sheet."
)

_PROFILE_LINE = re.compile(r"^- User (.+?): ", re.MULTILINE)
# "user 23", "user u-17", "user maria.santos@example.com": dots are allowed inside an id, not at its end.
_USER_MENTION = re.compile(r"\buser\s+([\w@-]+(?:\.[\w@-]+)*)", re.IGNORECASE)


def profile_users(sheet: str) -> list[str]:
    """User ids listed on a user-habits sheet."""
    return _PROFILE_LINE.findall(sheet)


def names_a_known_user(question: str, users: Iterable[str]) -> bool:
    """True when the question mentions any of these users by id (e.g. "user 23")."""
    known = {u.lower() for u in users}
    return any(m.lower() in known for m in _USER_MENTION.findall(question))


def is_single_user_question(question: str, users: list[str]) -> bool:
    """True when the question names exactly one user, and that user is on the sheet."""
    named = {m.lower() for m in _USER_MENTION.findall(question)}
    return len(named) == 1 and named <= {u.lower() for u in users}


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATES = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})(?:,?\s+\d{{4}})?",
    re.IGNORECASE,
)
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _tokens(text: str) -> set[str]:
    times = {t.zfill(5) for t in _TIME.findall(text)}
    numbers = set(_NUMBER.findall(_TIME.sub(" ", text)))
    return times | numbers


def ungrounded_numbers(answer: str, facts: Any) -> list[str]:
    """Numbers and times in the answer that don't appear in the facts (dates are ignored)."""
    known = _tokens(json.dumps(facts) if not isinstance(facts, str) else facts)
    known |= {str(int(float(n))) for n in known if n.replace(".", "", 1).isdigit() and "." in n}
    wanted = _tokens(_DATES.sub(" ", answer))
    return sorted(t for t in wanted if t not in known)
