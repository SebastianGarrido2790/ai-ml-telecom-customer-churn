"""
Core generation logic for synthetic ticket notes.

Uses pydantic-ai to interface with either Gemini (Google) or local
OpenAI-compatible servers (Ollama), ensuring structured outputs via
Pydantic validation. Includes a feature-based deterministic fallback
for pipeline reliability when all LLM providers are unavailable.

Leakage Prevention (C1 Fix):
    - `Churn Status` has been removed from the `user_prompt` string. The
      formatted prompt now exposes only observable CRM fields.
    - The deterministic fallback (Phase 3) previously branched on
      `customer_context.Churn`. It now uses service profile signals only:
      internet type, tech support availability, contract type, and charges.
"""

from typing import Any, cast

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from src.components.data_enrichment.prompts import ENRICHMENT_SYSTEM_PROMPT
from src.components.data_enrichment.schemas import CustomerInputContext, SyntheticNoteOutput
from src.entity.config_entity import DataEnrichmentConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnrichmentError(Exception):
    """Domain-specific exception for failures in the Data Enrichment pipeline."""

    pass


async def _call_google_llm(user_prompt: str, model_name: str, base_url: str | None) -> SyntheticNoteOutput:
    """Invokes the Google Gemini model via pydantic-ai.

    Args:
        user_prompt: Formatted customer profile string for this row.
        model_name: Gemini model identifier (e.g., 'gemini-2.0-flash').
        base_url: Optional base URL override for the Google provider.

    Returns:
        SyntheticNoteOutput with validated ticket_note and primary_sentiment_tag.
    """
    provider = GoogleProvider(base_url=base_url)
    model = GoogleModel(model_name, provider=provider)
    agent = Agent(
        model=model,
        output_type=SyntheticNoteOutput,
        system_prompt=ENRICHMENT_SYSTEM_PROMPT,
        retries=3,
    )
    result = await agent.run(user_prompt)
    return cast(Any, result).data


async def _call_ollama_llm(user_prompt: str, model_name: str, base_url: str | None) -> SyntheticNoteOutput:
    """Invokes a local Ollama server via raw HTTP.

    Uses direct JSON mode to bypass pydantic-ai's OpenAI adapter quirks
    with local servers. Implements agentic healing for list-to-string
    hallucination in the primary_sentiment_tag field.

    Args:
        user_prompt: Formatted customer profile string for this row.
        model_name: Ollama model identifier (e.g., 'ollama:qwen2.5:7b').
        base_url: Base URL of the local Ollama server.

    Returns:
        SyntheticNoteOutput with validated ticket_note and primary_sentiment_tag.
    """
    import json

    import httpx

    allowed_tags = '["Frustrated", "Dissatisfied", "Neutral", "Satisfied", "Billing Inquiry", "Technical Issue"]'
    system_instruction = (
        ENRICHMENT_SYSTEM_PROMPT
        + "\n\nYou MUST return ONLY valid JSON matching this schema: "
        + f'{{"ticket_note": "string", "primary_sentiment_tag": {allowed_tags}}} '
        + "without markdown blocks."
    )

    async with httpx.AsyncClient(timeout=300) as client:
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

        # Agentic Healing: handle list-to-string hallucination in sentiment tag
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


def _deterministic_fallback(
    customer_context: CustomerInputContext,
) -> SyntheticNoteOutput:
    """Generates a rule-based ticket note from observable CRM signals only.

    Triggered as a last resort when all LLM providers are unavailable.
    Branching logic uses only service profile features — contract type,
    internet service, tech support availability, and monthly charges.
    The target variable (Churn) is never referenced.

    Args:
        customer_context: Validated customer CRM profile.

    Returns:
        SyntheticNoteOutput with a deterministically generated note and tag.
    """
    internet = customer_context.InternetService
    tech = customer_context.TechSupport
    contract = customer_context.Contract
    charges = customer_context.MonthlyCharges
    tenure = customer_context.tenure
    cid = customer_context.customerID

    # High-friction profile: fiber + no support + month-to-month
    if internet == "Fiber optic" and tech == "No" and contract == "Month-to-month":
        note = (
            f"Cust {cid} contacted support regarding recurring connectivity issues "
            f"with their Fiber optic service. Monthly charges of ${charges:.2f} "
            f"were flagged as a concern. Agent noted the absence of TechSupport "
            f"on the account."
        )
        sentiment = "Frustrated"

    # Price-sensitive: high charges + month-to-month
    elif charges > 70.0 and contract == "Month-to-month":
        note = (
            f"Cust called to inquire about their monthly invoice of ${charges:.2f}. "
            f"Requested information on available promotional plans or discounts. "
            f"Agent reviewed current plan details with the customer."
        )
        sentiment = "Billing Inquiry"

    # Technical issue: fiber or DSL, no security, no backup
    elif (
        internet in ("Fiber optic", "DSL")
        and customer_context.OnlineSecurity == "No"
        and customer_context.OnlineBackup == "No"
    ):
        note = (
            f"Cust reported intermittent issues with their {internet} connection. "
            f"Agent noted the account has no OnlineSecurity or OnlineBackup services. "
            f"Ticket logged for follow-up."
        )
        sentiment = "Technical Issue"

    # Loyal long-term: two-year contract
    elif contract == "Two year":
        note = (
            f"Cust with {tenure} months of tenure on a two-year contract called to "
            f"inquire about available service upgrades. "
            f"Expressed general satisfaction with current plan."
        )
        sentiment = "Satisfied"

    # New customer: short tenure
    elif tenure <= 6:
        note = (
            f"New customer ({tenure} months) called with questions about their "
            f"first invoice of ${charges:.2f} and service setup. "
            f"Agent walked through billing cycle and available add-on services."
        )
        sentiment = "Neutral"

    # Default: routine inquiry
    else:
        note = (
            f"Cust with {tenure} months tenure on a {contract} contract called "
            f"regarding their {internet} service plan. "
            f"Agent addressed the inquiry and noted no escalation required."
        )
        sentiment = "Neutral"

    return SyntheticNoteOutput(ticket_note=note, primary_sentiment_tag=sentiment)


async def generate_ticket_note(
    customer_context: CustomerInputContext, config: DataEnrichmentConfig
) -> SyntheticNoteOutput:
    """Generates a structured synthetic ticket note for a single customer.

    Implements a 3-tier fallback chain:
        1. Primary LLM (Google Gemini via pydantic-ai).
        2. Secondary LLM (Ollama local server, hybrid mode only).
        3. Deterministic rule-based fallback (feature-signals only, no label).

    Args:
        customer_context: Validated CRM context for this customer row.
        config: DataEnrichmentConfig with provider, model, and retry settings.

    Returns:
        SyntheticNoteOutput with ticket_note and primary_sentiment_tag.
    """
    user_prompt = f"""
Customer CRM Profile:
- Customer ID     : {customer_context.customerID}
- Tenure          : {customer_context.tenure} months
- Gender          : {customer_context.gender}
- Senior Citizen  : {"Yes" if customer_context.SeniorCitizen == 1 else "No"}
- Partner         : {customer_context.Partner}
- Dependents      : {customer_context.Dependents}
- Internet Service: {customer_context.InternetService}
- Online Security : {customer_context.OnlineSecurity}
- Online Backup   : {customer_context.OnlineBackup}
- Device Protect. : {customer_context.DeviceProtection}
- Tech Support    : {customer_context.TechSupport}
- Streaming TV    : {customer_context.StreamingTV}
- Streaming Movies: {customer_context.StreamingMovies}
- Contract        : {customer_context.Contract}
- Paperless Bill. : {customer_context.PaperlessBilling}
- Payment Method  : {customer_context.PaymentMethod}
- Monthly Charges : ${customer_context.MonthlyCharges:.2f}

Write the CRM interaction note for this customer.
"""

    try:
        # --- Tier 1: Primary Provider (Google Gemini) ---
        if config.model_provider in ("google", "hybrid"):
            try:
                return await _call_google_llm(user_prompt, config.model_name, config.base_url)
            except Exception as primary_err:
                if config.model_provider != "hybrid":
                    raise primary_err
                logger.warning(
                    f"Primary model (Google) failed for "
                    f"{customer_context.customerID}. "
                    f"Falling back to Secondary. Error: {primary_err!s}"
                )

        if config.model_provider == "openai":
            return await _call_ollama_llm(user_prompt, config.model_name, config.base_url)

        # --- Tier 2: Secondary Provider (Ollama, hybrid mode only) ---
        if config.model_provider == "hybrid" and config.secondary_model_name:
            try:
                return await _call_ollama_llm(
                    user_prompt,
                    config.secondary_model_name,
                    config.secondary_base_url,
                )
            except Exception as secondary_err:
                logger.error(
                    f"Secondary model (Ollama) also failed for {customer_context.customerID}: {secondary_err!s}"
                )
                # Fall through to deterministic

        if config.model_provider not in ("google", "openai", "hybrid"):
            raise ValueError(f"Unsupported model provider: {config.model_provider}")

    except UnexpectedModelBehavior as e:
        logger.error(f"Structured output failure for {customer_context.customerID}: {e!s}")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {getattr(e, 'message', str(e))}"
        logger.warning(f"Total API failure for {customer_context.customerID}: {error_msg}")

    # --- Tier 3: Deterministic Fallback (feature-signals only, no label) ---
    logger.warning(f"Using deterministic fallback for {customer_context.customerID}.")
    return _deterministic_fallback(customer_context)
