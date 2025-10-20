---
title: Pandas OpenAI DataFrame Accessor
date: 2025-10-20
author: Shanaka DeSoysa
description: Extend pandas with a custom accessor so a DataFrame can call the OpenAI API row by row.
---

# Pandas Accessor for OpenAI Responses

Create a lightweight helper so `df.ai.generate_response()` can call the OpenAI API for each row in a DataFrame. This pattern is handy when you need to enrich data (summaries, classifications, translations) without scattering API plumbing throughout your notebooks.

## 1. Install and configure OpenAI

```sh
pip install openai pandas
export OPENAI_API_KEY="sk-..."  # make sure the key is available to the process
```

If you prefer to avoid environment variables, you can pass the key explicitly when the OpenAI client is created, but storing it in `OPENAI_API_KEY` keeps credentials out of your code.

## 2. Register the accessor

Save the snippet below as `ai_accessor.py` (or drop it into a utilities module that is imported before first use). It registers a custom accessor named `ai` on every DataFrame.

```python
import time
from typing import Optional

import pandas as pd
from openai import OpenAI

client = OpenAI()  # picks up OPENAI_API_KEY automatically


@pd.api.extensions.register_dataframe_accessor("ai")
class OpenAIFrameAccessor:
    """Adds AI helper methods to DataFrames."""

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def generate_response(
        self,
        prompt_column: str,
        model: str = "gpt-4o-mini",
        output_column: str = "ai_response",
        system_prompt: Optional[str] = "You are a helpful assistant.",
        temperature: float = 0.2,
        overwrite: bool = False,
        sleep: float = 0.0,
        max_retries: int = 3,
        **chat_kwargs,
    ) -> pd.DataFrame:
        """
        Call OpenAI for rows with missing output and return the mutated DataFrame.

        Parameters
        ----------
        prompt_column: str
            Column containing the user prompt for OpenAI.
        model: str
            Chat model to call.
        output_column: str
            Column where responses are stored. Created when missing.
        system_prompt: Optional[str]
            Optional system message prepended to each request.
        temperature: float
            Passed through to the chat completion API.
        overwrite: bool
            Force re-generation even if an answer already exists.
        sleep: float
            Optional delay (seconds) between calls for rate-limit headroom.
        max_retries: int
            Number of retry attempts before recording the error text.
        chat_kwargs:
            Forwarded verbatim to `client.chat.completions.create`.
        """
        df = self._obj

        if prompt_column not in df.columns:
            raise KeyError(f"Column '{prompt_column}' is missing.")

        if output_column not in df.columns or overwrite:
            df[output_column] = pd.NA

        pending = df[df[output_column].isna()]

        for idx, prompt in pending[prompt_column].items():
            messages = [{"role": "user", "content": str(prompt)}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            for attempt in range(1, max_retries + 1):
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        **chat_kwargs,
                    )
                    df.at[idx, output_column] = (
                        completion.choices[0].message.content.strip()
                    )
                    break
                except Exception as exc:
                    if attempt == max_retries:
                        df.at[idx, output_column] = f"ERROR: {exc}"
                    else:
                        time.sleep(min(2 ** attempt, 10))

            if sleep:
                time.sleep(sleep)

        return df
```

## 3. Use the accessor

```python
import pandas as pd
from ai_accessor import OpenAIFrameAccessor  # noqa: F401 - registers the accessor

df = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "text": [
            "Explain what a DataFrame accessor does.",
            "Summarize pandas' groupby operation in one sentence.",
            "List three use cases for joining DataFrames.",
        ],
    }
)

df.ai.generate_response(
    prompt_column="text",
    output_column="ai_summary",
    model="gpt-4o-mini",
    temperature=0.0,
)

print(df[["id", "ai_summary"]])
```

The accessor only calls OpenAI for rows where `ai_summary` is missing, so you can resume partially completed jobs or override responses by setting `overwrite=True`.

## 4. Extra ideas

- Pass `response_format={"type": "json_object"}` through `chat_kwargs` to coerce structured output that can be parsed into new columns.
- Move API calls into an async worker or queue if you need true concurrency; the accessor keeps things simple and sequential to avoid rate surprises.
- Cache results on disk (e.g., Parquet or SQLite) between runs to stay within usage quotas.

This approach keeps all OpenAI plumbing in one place while matching pandas' fluent style, making it easy to iterate on prompt engineering directly from your notebooks.
