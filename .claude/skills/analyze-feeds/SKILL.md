---
name: analyze-feeds
user-invocable: false
allowed-tools: Bash, Read, Glob, Grep
---

# Feed Analysis Skill

You use this skill to analyze ASFINAG highway camera feeds using Gemini Vision API. Given a user's natural language filter query, you design a categorization scheme, run the analysis, and return matching camera IDs.

## Steps

### 1. Validate the prompt
Reject empty or nonsensical prompts with an error JSON response.

### 2. Design categorization
Based on the user's query, create 2-5 relevant categories. Always include fallback categories for unclear/broken feeds.

Examples:
- Traffic query ("stau", "viel verkehr"): `["empty", "light", "dense", "standing", "unclear", "no_feed"]`
- Weather query ("schnee", "bad weather"): `["clear", "rain", "snow", "fog", "unclear", "no_feed"]`
- Feed status ("broken", "kaputt"): `["working", "black_screen", "frozen", "test_pattern", "no_feed"]`

### 3. Generate Gemini prompt
Write a clear English prompt for Gemini Vision that:
- Describes what each category means visually
- Instructs Gemini to categorize each camera image
- References cameras by their `"Camera: {wcsid}"` labels
- Is specific about visual criteria for each category

### 4. Generate response schema
Create a JSON schema dict for Gemini structured output. The schema should have a `cameras` array where each item has:
- `wcsid`: string — the camera ID
- `category`: string enum — one of your designed categories

Example schema (as JSON string for CLI):
```json
{"type": "OBJECT", "properties": {"cameras": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"wcsid": {"type": "STRING"}, "category": {"type": "STRING", "enum": ["empty", "light", "dense", "standing", "unclear", "no_feed"]}}, "required": ["wcsid", "category"]}}}, "required": ["cameras"]}
```

### 5. Run analysis
GEMINI_API_KEY is already set in the environment (inherited from the control room server). Do NOT read .env or set it yourself. Just run:

```bash
uv run --with google-genai --with aiohttp --with pydantic analyze_feeds.py \
  --prompt "<your gemini prompt>" \
  --schema '<your schema json>' \
  --server-url http://localhost:8050
```

Choose `--frames-per-camera` based on the query:
- Use `1` (default) for static observations (weather, feed status, road conditions)
- Use `2-3` for temporal observations (traffic flow, movement, changes)

### 6. Process results
Parse the JSON output from analyze_feeds.py. Collect camera IDs (`wcsid` values) where the `category` matches the user's filter intent.

For example, if the user asked for "stau" (traffic jam), collect cameras with category `"standing"` or `"dense"`.

### 7. Output result
Return the final JSON:
```json
{"status": "success", "filter": {"camera_ids": ["CAM-1", "CAM-2"], "query": "original query", "description": "Showing 5 cameras with heavy traffic or standstill"}}
```

If no cameras match, return success with an empty array and appropriate description.

If analyze_feeds.py fails, return:
```json
{"status": "error", "message": "Analysis failed: <reason>"}
```
