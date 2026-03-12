"""
This module provides the core generation logic for synthetic ticket notes.
It uses pydantic-ai to interface with the Gemini LLM, ensuring structured
outputs that match our validation schemas.
"""

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from src.components.data_enrichment.prompts import ENRICHMENT_SYSTEM_PROMPT
from src.components.data_enrichment.schemas import CustomerInputContext, SyntheticNoteOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Environment variables should be loaded by the entry point


class EnrichmentError(Exception):
    """Domain-specific exception for failures in the Data Enrichment pipeline."""

    pass


async def generate_ticket_note(
    customer_context: CustomerInputContext, model_name: str = "gemini-1.5-flash"
) -> SyntheticNoteOutput:
    """
    Generates a structured synthetic ticket note for a single customer.

    Args:
        customer_context (CustomerInputContext): Fully validated Pydantic model for a row.
        model_name (str): LLM model name to use.

    Returns:
        SyntheticNoteOutput: The synthesized ticket note and semantic tag.

    Raises:
        EnrichmentError: If generation fails after built-in retries and no fallback is available.
    """
    agent = Agent(
        model=model_name,
        result_type=SyntheticNoteOutput,
        system_prompt=ENRICHMENT_SYSTEM_PROMPT,
        retries=3,
    )
    try:
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
        # Execute the agent
        result = await agent.run(user_prompt)
        return result.data

    except UnexpectedModelBehavior as e:
        # Catch pydantic_ai specific validation issues
        # (e.g. model output didn't fit schema after 3 retries)
        raise EnrichmentError(
            f"Model failed to generate structured output for "
            f"{customer_context.customerID}: {str(e)}"
        ) from e
    except Exception as e:
        # Catch general network or other auth API issues
        logger.warning(
            f"API call failed, using deterministic fallback for "
            f"{customer_context.customerID}: {e}"
        )

        # MLOps Principle: Fallback to deterministic logic when probabilistic logic fails
        # to ensure pipeline continuity during development/testing.
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
