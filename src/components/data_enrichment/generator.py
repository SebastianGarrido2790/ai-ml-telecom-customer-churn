"""
This module provides the core generation logic for synthetic ticket notes.
It uses pydantic-ai to interface with either Gemini (Google) or local OpenAI-compatible
servers (Ollama), ensuring structured outputs via Pydantic validation.
Includes deterministic fallback logic for pipeline reliability.
"""

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from src.components.data_enrichment.prompts import ENRICHMENT_SYSTEM_PROMPT
from src.components.data_enrichment.schemas import CustomerInputContext, SyntheticNoteOutput
from src.entity.config_entity import DataEnrichmentConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Environment variables should be loaded by the entry point


class EnrichmentError(Exception):
    """Domain-specific exception for failures in the Data Enrichment pipeline."""

    pass


async def _call_google_llm(
    user_prompt: str, model_name: str, base_url: str | None
) -> SyntheticNoteOutput:
    """Invokes the Google Gemini model via pydantic-ai."""
    provider = GoogleProvider(base_url=base_url)
    model = GoogleModel(model_name, provider=provider)
    agent = Agent(
        model=model,
        output_type=SyntheticNoteOutput,
        system_prompt=ENRICHMENT_SYSTEM_PROMPT,
        retries=3,
    )
    result = await agent.run(user_prompt)
    return result.data


async def _call_ollama_llm(
    user_prompt: str, model_name: str, base_url: str | None
) -> SyntheticNoteOutput:
    """Invokes a local Ollama server via raw HTTP (workaround for pydantic-ai OpenAI issues)."""
    import json

    import httpx

    allowed_tags = (
        '["Frustrated", "Dissatisfied", "Neutral", "Satisfied", '
        '"Billing Inquiry", "Technical Issue"]'
    )
    system_instruction = (
        ENRICHMENT_SYSTEM_PROMPT
        + "\n\nYou MUST return ONLY valid JSON matching this schema: "
        + f'{{"ticket_note": "string", "primary_sentiment_tag": {allowed_tags}}} '
        + "without markdown blocks."
    )

    async with httpx.AsyncClient(timeout=300) as client:
        # Default to local if no base_url provided
        target_url = base_url if base_url else "http://localhost:11434/v1"
        ollama_url = target_url.replace("/v1", "/api/generate")
        if "/api/generate" not in ollama_url:
            ollama_url = ollama_url.rstrip("/") + "/api/generate"

        payload = {
            "model": model_name.replace("ollama:", ""),
            "prompt": system_instruction + "\n\n" + user_prompt,
            "format": "json",
            "stream": False,
        }

        response = await client.post(ollama_url, json=payload)
        response.raise_for_status()

        data = response.json()
        raw_json_str = data.get("response", "{}").strip()

        # Strip markdown JSON fences
        if raw_json_str.startswith("```json"):
            raw_json_str = raw_json_str[7:]
        if raw_json_str.startswith("```"):
            raw_json_str = raw_json_str[3:]
        if raw_json_str.endswith("```"):
            raw_json_str = raw_json_str[:-3]
        raw_json_str = raw_json_str.strip()

        # Rule: Agentic Healing - Handle list-to-string hallucination
        try:
            parsed = json.loads(raw_json_str)
            tag = parsed.get("primary_sentiment_tag")
            if tag:
                if isinstance(tag, list) and len(tag) > 0:
                    parsed["primary_sentiment_tag"] = str(tag[0]).strip()
                elif isinstance(tag, str):
                    parsed["primary_sentiment_tag"] = tag.strip()
                raw_json_str = json.dumps(parsed)
        except (json.JSONDecodeError, AttributeError, KeyError):
            pass

        return SyntheticNoteOutput.model_validate_json(raw_json_str)


async def generate_ticket_note(
    customer_context: CustomerInputContext, config: DataEnrichmentConfig
) -> SyntheticNoteOutput:
    """
    Generates a structured synthetic ticket note for a single customer.

    Implements a fallback chain:
    1. Primary LLM (Google or OpenAI)
    2. Secondary LLM (if model_provider is 'hybrid')
    3. Deterministic rule-based fallback (last resort)
    """
    user_prompt = f"""
    Customer Profile:
    - Tenure: {customer_context.tenure} months
    - Internet Service: {customer_context.InternetService}
    - Contract: {customer_context.Contract}
    - Monthly Charges: ${customer_context.MonthlyCharges:.2f}
    - Tech Support: {customer_context.TechSupport}
    - Churn Status: {customer_context.Churn}

    Generate the ticket note.
    """

    try:
        # --- PHASE 1: Try Primary Provider ---
        if config.model_provider == "google" or config.model_provider == "hybrid":
            try:
                return await _call_google_llm(user_prompt, config.model_name, config.base_url)
            except Exception as primary_err:
                if config.model_provider != "hybrid":
                    raise primary_err
                logger.warning(
                    f"Primary model (Google) failed for {customer_context.customerID}. "
                    f"Falling back to Secondary. Error: {str(primary_err)}"
                )

        if config.model_provider == "openai":
            return await _call_ollama_llm(user_prompt, config.model_name, config.base_url)

        # --- PHASE 2: Fallback to Secondary Model (Hybrid Mode) ---
        if config.model_provider == "hybrid" and config.secondary_model_name:
            try:
                return await _call_ollama_llm(
                    user_prompt, config.secondary_model_name, config.secondary_base_url
                )
            except Exception as secondary_err:
                logger.error(
                    f"Secondary model (Ollama) also failed for {customer_context.customerID}: "
                    f"{str(secondary_err)}"
                )
                # Fall through to deterministic logic

        if config.model_provider not in ["google", "openai", "hybrid"]:
            raise ValueError(f"Unsupported model provider: {config.model_provider}")

    except UnexpectedModelBehavior as e:
        logger.error(f"Structured output failure for {customer_context.customerID}: {str(e)}")
        # Fall through to deterministic

    except Exception as e:
        error_msg = f"{type(e).__name__}: {getattr(e, 'message', str(e))}"
        logger.warning(f"Total API failure for {customer_context.customerID}: {error_msg}")
        # Fall through to deterministic

    # --- PHASE 3: Deterministic Fallback (MLOps Principle) ---
    if customer_context.Churn == "Yes":
        note = (
            f"Customer with ID {customer_context.customerID} is frustrated with "
            f"{customer_context.InternetService} service. "
            f"High monthly charges of ${customer_context.MonthlyCharges}."
        )
        sentiment = "Frustrated"
    else:
        note = (
            f"Customer {customer_context.customerID} is satisfied with the "
            f"{customer_context.Contract} contract and "
            f"{customer_context.InternetService} internet."
        )
        sentiment = "Satisfied"

    return SyntheticNoteOutput(ticket_note=note, primary_sentiment_tag=sentiment)
