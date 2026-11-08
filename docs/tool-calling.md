# Tool calling: train the model to look numbers up

Fine-tuning is good at teaching a model **behaviour**: tone, format, which question needs which data. It's bad at
making a model **remember numbers**, and remembered numbers go out of date the next day. So for questions about
live data, train the model to call a function and answer from its result:

```
user:      What was the peak login hour on November 6?
assistant: → get_activity_summary({"date_from": "2026-11-06", "date_to": "2026-11-06"})
tool:      {"total": 145, "peak_hour": {"hour": "09:00-09:59", "count": 21, "users": 17}, ...}
assistant: The peak was 09:00–09:59, with 21 logins from 17 users.
```

After fine-tuning, the model asks your app for the numbers each time, so its answers are right for any day,
including today, without retraining.

## Generate the training data

Add `style=tool_calls`:

```bash
curl -s --data-binary @examples/login.json \
  "http://localhost:8000/v1/training-data?style=tool_calls&output=jsonl" > training-result.jsonl
```

Each line is one conversation in OpenAI's fine-tuning format, with the tool definitions included:

```json
{"messages": [
  {"role": "user", "content": "When was user 18's last login on 2026-11-06?"},
  {"role": "assistant", "tool_calls": [{"id": "call_...", "type": "function", "function": {"name": "get_user_activity", "arguments": "{\"user_id\": \"18\", \"date_from\": \"2026-11-06\", \"date_to\": \"2026-11-06\"}"}}]},
  {"role": "tool", "tool_call_id": "call_...", "content": "{\"user_id\": \"18\", \"count\": 3, \"first\": \"2026-11-06 09:14\", \"last\": \"2026-11-06 12:32\", ...}"},
  {"role": "assistant", "content": "User 18's last login was at 12:32."}
], "tools": [...]}
```

The tool results in the training data are computed exactly from your rows, and every number in the final answer
is checked against its tool result (examples that fail are dropped as `number_not_in_facts`).

Formats that support tool calls: `openai` (default), `sharegpt` (LLaMA-Factory's `function_call` / `observation`
turns), `raw` and `eval`.

## The functions your app implements

The trained model will call these five functions, so your app must provide them. Get their exact definitions
(JSON schema) from the `tools` field of a JSON response (`output=json`, the default).

All take `date_from` and `date_to` (`YYYY-MM-DD`; the same day for one day). Wording uses your event name
("login", "order"...).

| Function | Extra argument | Answers questions like |
|---|---|---|
| `get_activity_summary` | | How many logins? Peak hour? Busiest 15 minutes? First and last login? Morning vs evening? |
| `get_hourly_counts` | | How busy was 13:00? Which hours had no logins? |
| `get_top_users` | | Who logs in the most? What share do the top 10 have? Who logged in only once? |
| `get_daily_counts` | | Which day was busiest? Average per day? |
| `get_user_activity` | `user_id` | When did user 7 first log in? Does user 13 have a usual hour? |

### What each function returns

Return the same field names: the model learned to read them. Values below are from `examples/login.json`
(except `get_daily_counts`, which is illustrative because that file covers one day).

**`get_activity_summary`**

```json
{
  "event": "login", "date_from": "2026-11-06", "date_to": "2026-11-06",
  "total": 145, "unique_users": 40, "average_per_user": 3.6,
  "peak_hour": {"hour": "09:00-09:59", "count": 21, "users": 17, "share_percent": 14},
  "next_busiest_hours": [{"hour": "08:00-08:59", "count": 19}, {"hour": "13:00-13:59", "count": 18}],
  "busiest_15_minutes": {"from": "09:02", "to": "09:17", "count": 11},
  "quietest_active_hours": {"hours": ["19:00-19:59", "23:00-23:59"], "count_each": 2},
  "hours_with_none": ["00:00-00:59", "01:00-01:59", "...", "06:00-06:59"],
  "first": {"time": "2026-11-06 07:01", "user_id": "20"},
  "last": {"time": "2026-11-06 23:33", "user_id": "31"},
  "parts_of_day": {
    "night": {"hours": "00:00-05:59", "count": 0, "share_percent": 0},
    "morning": {"hours": "06:00-11:59", "count": 70, "share_percent": 48},
    "afternoon": {"hours": "12:00-17:59", "count": 56, "share_percent": 39},
    "evening": {"hours": "18:00-23:59", "count": 19, "share_percent": 13}
  }
}
```

**`get_hourly_counts`**: `total`, plus all 24 hours:

```json
{"event": "login", "date_from": "2026-11-06", "date_to": "2026-11-06", "total": 145,
 "hours": [{"hour": "00:00-00:59", "count": 0, "users": 0}, "...", {"hour": "09:00-09:59", "count": 21, "users": 17}, "..."]}
```

**`get_top_users`**: the top 10:

```json
{"event": "login", "date_from": "2026-11-06", "date_to": "2026-11-06", "total": 145, "unique_users": 40,
 "top_users": [{"rank": 1, "user_id": "7", "count": 14, "share_percent": 10}, {"rank": 2, "user_id": "23", "count": 9, "share_percent": 6}, "..."],
 "top_10_total": 70, "top_10_share_percent": 48,
 "users_with_one": ["39", "24", "3", "6", "30", "16", "28", "38"]}
```

**`get_daily_counts`**

```json
{"event": "login", "date_from": "2026-11-01", "date_to": "2026-11-07",
 "days": [{"date": "2026-11-06", "weekday": "Friday", "count": 145}, "..."],
 "busiest_day": {"date": "2026-11-06", "count": 145}, "quietest_day": {"date": "2026-11-01", "count": 38},
 "average_per_day": 102.4}
```

**`get_user_activity`**: `usual_hour` is `null` when there's no clear one (a tie, or a single login):

```json
{"event": "login", "date_from": "2026-11-06", "date_to": "2026-11-06", "user_id": "7",
 "count": 14, "first": "2026-11-06 07:42", "last": "2026-11-06 21:07",
 "usual_hour": {"hour": "14:00-14:59", "count": 4}, "distinct_hours": 9,
 "parts_of_day": ["morning", "afternoon", "evening"], "days_active": 1}
```

`llmvero/core/records.py` (`analyze_records`) is the reference implementation of every field, if you want to
match it exactly.

### Example queries (PostgreSQL)

For a table `logins(user_id, logged_in_at)`:

```sql
-- get_activity_summary: total, unique users, peak hour
SELECT count(*) AS total, count(DISTINCT user_id) AS unique_users
FROM logins WHERE logged_in_at >= $1::date AND logged_in_at < $2::date + 1;

SELECT extract(hour FROM logged_in_at) AS hour, count(*) AS count, count(DISTINCT user_id) AS users
FROM logins WHERE logged_in_at >= $1::date AND logged_in_at < $2::date + 1
GROUP BY 1 ORDER BY count DESC, hour LIMIT 3;          -- peak hour + next busiest

-- get_top_users
SELECT user_id, count(*) AS count
FROM logins WHERE logged_in_at >= $1::date AND logged_in_at < $2::date + 1
GROUP BY user_id ORDER BY count DESC, min(logged_in_at) LIMIT 10;

-- get_user_activity
SELECT count(*) AS count, min(logged_in_at) AS first, max(logged_in_at) AS last,
       count(DISTINCT extract(hour FROM logged_in_at)) AS distinct_hours
FROM logins WHERE user_id = $3 AND logged_in_at >= $1::date AND logged_in_at < $2::date + 1;
```

## Using the fine-tuned model

Send the same tool definitions with every request, run the function the model asks for, and send the result
back. Tell the model today's date so it can turn "yesterday" into `date_from` / `date_to`.

```js
// Node.js, OpenAI SDK
const messages = [
  { role: "system", content: `You are the analytics assistant. Today is ${today}.` },
  { role: "user", content: "Who logged in the most yesterday?" },
];
const first = await openai.chat.completions.create({ model: FINE_TUNED_MODEL, messages, tools });
const reply = first.choices[0].message;

if (reply.tool_calls) {
  messages.push(reply);
  for (const call of reply.tool_calls) {
    const args = JSON.parse(call.function.arguments);
    const result = await functions[call.function.name](args); // your DB queries
    messages.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify(result) });
  }
  const final = await openai.chat.completions.create({ model: FINE_TUNED_MODEL, messages, tools });
  console.log(final.choices[0].message.content); // "User 7 logged in the most, 14 times."
}
```

The base model (before fine-tuning) can already do this with the same tools. Fine-tuning on LLMvero's data makes
it more reliable at picking the right function and arguments, and makes the answers follow your wording.

## Testing it

Generate test cases with `format=eval&style=tool_calls`. Each line has the question, the expected function call
and the expected answer:

```json
{"question": "Who was the most active user on November 6, 2026?",
 "expected_tool_call": {"name": "get_top_users", "arguments": {"date_from": "2026-11-06", "date_to": "2026-11-06"}},
 "expected_answer": "User 7 was the most active, with 14 logins.",
 "facts": {"top_users": [{"rank": 1, "user_id": "7", "count": 14, "...": "..."}], "...": "..."}}
```

Run each question through your assistant and check it called the expected function with the expected arguments,
and that its answer contains the numbers in `expected_answer`.
