"""
LLM Client (OpenRouter) with JSON Mode
=======================================
Same OpenRouter pattern as previous weeks, with one addition:
JSON mode forces the model to return a parseable structured verdict
instead of free text. This is what makes the LLM filter reliable enough
to act on programmatically.

Demo talking point:
    "Without JSON mode, the model might say 'I think this should be a
    timeout' or 'This message is borderline' — text we'd have to parse
    and might get wrong. With JSON mode we get a machine-readable object
    we can route directly to the right action."
"""

import os
import json
from openai import OpenAI

DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The schema the LLM filter expects back from the model.
# Pre-built so the instructor can focus on the system prompt, not boilerplate.
MODERATION_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "moderation_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["allow", "timeout", "ban"],
                    "description": "The moderation action to take",
                },
                "reason": {
                    "type": "string",
                    "description": "One sentence explaining the decision",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "How confident the model is in this decision",
                },
                "rule_triggered": {
                    "type": "string",
                    "description": "Which community rule this message violated, or 'none' if allowed",
                },
            },
            "required": ["decision", "reason", "confidence", "rule_triggered"],
            "additionalProperties": False,
        },
    },
}


class LLMClient:
    """
    OpenRouter client configured for moderation use.

    Parameters
    ----------
    api_key : str | None
        Reads from OPENROUTER_API_KEY env var if not provided.
    model : str | None
        Reads from OPENROUTER_MODEL env var, falls back to default.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY in your .env file."
            )
        self.model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=OPENROUTER_BASE_URL,
        )

    def moderate(self, system_prompt: str, message: str) -> dict:
        """
        Ask the LLM to make a moderation decision and return a structured verdict.

        Parameters
        ----------
        system_prompt : str
            The community rules and instructions for the moderator.
            Written live by the instructor during the demo.
        message : str
            The message content to evaluate.

        Returns
        -------
        dict with keys: decision, reason, confidence, rule_triggered
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Evaluate this message:\n\n{message}"},
            ],
            response_format=MODERATION_RESPONSE_SCHEMA,
            temperature=0.1,   # Low temperature for consistent decisions
        )
        raw = response.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback if JSON mode fails — treat as allow with low confidence
            return {
                "decision": "allow",
                "reason": f"JSON parse failed — raw response: {raw[:100]}",
                "confidence": "low",
                "rule_triggered": "none",
            }

    def __repr__(self):
        return f"LLMClient(model='{self.model}')"
