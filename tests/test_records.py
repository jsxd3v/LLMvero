import pytest

from llmvero.core.records import RecordError, parse_time, summarize_records

ROWS = [
    {"user_id": 1, "date_time": "2026-11-06 09:05:00"},
    {"user_id": 1, "date_time": "2026-11-06 09:20:00"},
    {"user_id": 2, "date_time": "2026-11-06T09:40:00Z"},
    {"user_id": 1, "date_time": "2026-11-06 13:00:00"},
    {"user_id": 3, "date_time": "2026-11-06 22:10:00"},
]


def test_parse_time_formats():
    assert parse_time("2026-11-06 09:05:00").hour == 9
    assert parse_time("2026-11-06T09:05:00+07:00").hour == 9  # clock time kept as written
    assert parse_time("2026-11-06T09:05:00Z").minute == 5
    assert parse_time(1_700_000_000).year == 2023
    assert parse_time(1_700_000_000_000).year == 2023  # milliseconds


def test_login_facts_are_exact():
    facts = dict(summarize_records(ROWS, event="login"))
    overview = facts["overview"]
    assert "Login activity on Friday, 2026-11-06" in overview
    assert "Total logins: 5" in overview and "Unique users: 3" in overview
    assert "Peak hour: 09:00-09:59 with 3 logins (60% of the total) by 2 different users" in overview
    assert "First login: 2026-11-06 09:05 by user 1" in overview
    assert "Last login: 2026-11-06 22:10 by user 3" in overview
    assert "1. User 1: 3 logins" in facts["users"]
    assert "Users with only one login (2): 2, 3" in facts["users"]
    assert "13:00-13:59: 1 logins by 1 users" in facts["hourly"]
    assert "User 1: 3 logins; first at 2026-11-06 09:05, last at 2026-11-06 13:00" in facts["user_profiles_1"]
    assert "daily" not in facts


def test_multi_day_adds_daily_section():
    rows = ROWS + [{"user_id": 2, "date_time": "2026-11-07 10:00:00"}]
    facts = dict(summarize_records(rows, event="login"))
    assert "Busiest day: Friday 2026-11-06 (5 logins)" in facts["daily"]
    assert "active on 2 of 2 days" in facts["user_profiles_1"]


def test_custom_fields_and_bad_rows():
    facts = dict(
        summarize_records([{"uid": "ana", "ts": "2026-11-06 08:00"}], user_field="uid", time_field="ts")
    )
    assert "User ana" in facts["users"]
    with pytest.raises(RecordError, match="couldn't tell which field is the time"):
        summarize_records([{"user_id": 1}])
    with pytest.raises(RecordError, match="can't read"):
        summarize_records([{"user_id": 1, "date_time": "yesterday"}])


def test_detects_common_field_names_and_nesting():
    from llmvero.core.records import detect_fields, load_rows

    assert detect_fields([{"id": 9, "userId": 1, "createdAt": "2026-11-06T09:00:00"}]) == (
        "userId",
        "createdAt",
    )
    rows, hint = load_rows('{"logins": [{"user": {"id": 1}, "when": "2026-11-06 09:00"}]}')
    assert hint == "login"
    facts = dict(summarize_records(rows, event="login"))
    assert "User 1" in facts["users"]
    # No known names: falls back to a field whose values look like dates, and a field with "user" in it.
    assert detect_fields([{"who_user": "a", "stamp": "2026-11-06 09:00"}]) == ("who_user", "stamp")


def test_user_sheets_only_keep_questions_about_one_listed_user():
    from llmvero.core.records import is_single_user_question, profile_users

    facts = dict(summarize_records(ROWS, event="login"))
    users = profile_users(facts["user_profiles_1"])
    assert users == ["1", "2", "3"]
    good = [
        "When did user 1 first log in on November 6, 2026?",
        "What was User 2's usual login hour?",
        "How many times did user 3 log in?",
    ]
    bad = [  # real mistakes a model made from a partial sheet
        "Which users had the most logins on November 6, 2026?",
        "How many logins occurred between 09:00 and 09:59 on November 6, 2026?",
        "How many logins did the ten listed users make on November 6, 2026?",
        "Who logged in during the 08:00 hour on November 6, 2026?",
        "Did user 1 log in earlier than user 2?",
        "When did user 99 log in?",
    ]
    assert all(is_single_user_question(q, users) for q in good)
    assert not any(is_single_user_question(q, users) for q in bad)


def test_answers_must_only_use_numbers_from_the_facts():
    from llmvero.core.records import ungrounded_numbers

    facts = {"total": 145, "peak_hour": {"hour": "09:00-09:59", "count": 21, "share_percent": 14}, "avg": 3.6}
    assert ungrounded_numbers("There were 145 logins on November 6, 2026.", facts) == []
    assert ungrounded_numbers("Peak 9:00-09:59: 21 logins (14%), 3.6 per user, on 2026-11-06.", facts) == []
    assert ungrounded_numbers("There were 146 logins.", facts) == ["146"]
    # a total the model added up itself is not in an hourly result
    assert ungrounded_numbers("145 logins in total.", {"hours": [{"count": 21}]}) == ["145"]


def test_hourly_facts_include_the_total():
    from llmvero.core.records import analyze_records

    a = analyze_records(ROWS, event="login")
    assert a.hourly["total"] == 5 and "Total: 5 logins" in dict(a.sections)["hourly"]


def test_summary_sheet_questions_about_one_user_are_detected():
    from llmvero.core.records import names_a_known_user

    users = ["7", "23"]
    assert names_a_known_user("How many logins did user 23 make on November 6?", users)
    assert names_a_known_user("Was User 7's count the highest?", users)
    assert not names_a_known_user("Who was the most active user on November 6?", users)
    assert not names_a_known_user("How many user logins were there?", users)  # "user logins" is not an id


def test_user_ids_with_dots_and_emails_are_recognised():
    from llmvero.core.records import (
        analyze_records,
        is_single_user_question,
        names_a_known_user,
        profile_users,
    )

    rows = [
        {"email": "maria.santos@example.com", "date_time": "2026-11-06 09:01:00"},
        {"email": "maria.santos@example.com", "date_time": "2026-11-06 09:30:00"},
        {"email": "john.doe@example.com", "date_time": "2026-11-06 10:15:00"},
    ]
    facts = dict(summarize_records(rows, event="login"))
    users = profile_users(facts["user_profiles_1"])
    assert sorted(users) == ["john.doe@example.com", "maria.santos@example.com"]

    assert is_single_user_question("How many times did user maria.santos@example.com log in?", users)
    assert is_single_user_question(
        "When did user john.doe@example.com first log in on November 6, 2026?", users
    )
    assert is_single_user_question(
        "Was user john.doe@example.com.", users
    )  # sentence dot isn't part of the id
    assert not is_single_user_question(
        "Did user maria.santos@example.com log in before user john.doe@example.com?", users
    )
    assert not is_single_user_question("When did user ghost@example.com log in?", users)
    assert names_a_known_user("How many logins did user maria.santos@example.com make?", users)

    a = analyze_records(rows, event="login")
    call = a.tool_call_for("user_profiles_1", "How many times did user maria.santos@example.com log in?")
    assert (
        call is not None
        and call[0] == "get_user_activity"
        and call[1]["user_id"] == "maria.santos@example.com"
    )
