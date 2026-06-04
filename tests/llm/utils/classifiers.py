import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import openai
from autoevals import LLMClassifier, init
from braintrust import Span, SpanTypeAttribute
from braintrust.oai import wrap_openai

from tests.llm.utils.test_case_utils import _model_list_exists, create_eval_llm
from tests.llm.utils.test_env_vars import (
    AZURE_API_BASE,
    AZURE_API_KEY,
    AZURE_API_VERSION,
    CLASSIFIER_MODEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENROUTER_API_BASE,
    OPENROUTER_API_KEY,
)


@dataclass
class ClassifierModelParams:
    model: str
    api_key: Optional[str]
    api_base: Optional[str]
    api_version: Optional[str]

    @property
    def is_azure(self) -> bool:
        return bool(self.api_base and self.api_version)


def get_classifier_model_params() -> ClassifierModelParams:
    """Get classifier model parameters from model list or environment variables."""
    if _model_list_exists():
        llm = create_eval_llm(CLASSIFIER_MODEL)
        model_for_api = llm.model
        client_api_key = llm.api_key
        client_base_url = llm.api_base
        client_api_version = llm.api_version

        # The classifier talks to OpenAI/OpenRouter directly via the openai SDK
        # (autoevals doesn't go through litellm), so litellm-style provider
        # prefixes in the model name must be stripped before we send the call.
        if model_for_api and model_for_api.startswith("openrouter/"):
            model_for_api = model_for_api.split("/", 1)[1]
            # Fall back to OpenRouter's public endpoint if model_list didn't pin one.
            if not client_base_url:
                client_base_url = (
                    OPENROUTER_API_BASE or "https://openrouter.ai/api/v1"
                )
    else:
        if not OPENAI_API_KEY and not AZURE_API_KEY and not OPENROUTER_API_KEY:
            raise ValueError(
                "No API key found (AZURE_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY)"
            )
        model_for_api = CLASSIFIER_MODEL
        if AZURE_API_BASE:
            client_api_key = AZURE_API_KEY
            client_base_url = AZURE_API_BASE
        elif OPENAI_API_KEY:
            client_api_key = OPENAI_API_KEY
            client_base_url = OPENAI_API_BASE
        elif OPENROUTER_API_KEY:
            client_api_key = OPENROUTER_API_KEY
            client_base_url = OPENROUTER_API_BASE or "https://openrouter.ai/api/v1"
        else:
            raise ValueError(
                "No API key found (AZURE_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY)"
            )
        client_api_version = AZURE_API_VERSION

        # Strip provider prefixes for API calls
        if AZURE_API_BASE and CLASSIFIER_MODEL.startswith("azure/"):
            if len(CLASSIFIER_MODEL.split("/")) != 2:
                raise ValueError(
                    f"Current classifier model '{CLASSIFIER_MODEL}' does not meet the pattern 'azure/<deployment-name>' when using Azure AI Foundry."
                )
            model_for_api = CLASSIFIER_MODEL.split("/", 1)[1]
        elif CLASSIFIER_MODEL.startswith("openrouter/"):
            # Strip "openrouter/" prefix - OpenRouter expects "openai/gpt-4.1" not "openrouter/openai/gpt-4.1"
            model_for_api = CLASSIFIER_MODEL.split("/", 1)[1]

    return ClassifierModelParams(
        model=model_for_api,
        api_key=client_api_key,
        api_base=client_base_url,
        api_version=client_api_version,
    )


classifier_model = CLASSIFIER_MODEL


def create_llm_client():
    """Create OpenAI/Azure client with same logic used by tests"""
    params = get_classifier_model_params()

    if params.is_azure:
        deployment = (
            params.model.split("/", 1)[1] if "/" in params.model else params.model
        )
        if not params.api_key:
            raise ValueError("No AZURE_API_KEY")
        client = openai.AzureOpenAI(
            azure_endpoint=params.api_base,
            azure_deployment=deployment,
            api_version=params.api_version,
            api_key=params.api_key,
        )
        model_for_api = deployment
    else:
        if not params.api_key:
            raise ValueError("No OPENAI_API_KEY or OPENROUTER_API_KEY")
        client = openai.OpenAI(api_key=params.api_key, base_url=params.api_base)
        model_for_api = params.model

    return client, model_for_api


# Register client with autoevals
try:
    client, _ = create_llm_client()
    params = get_classifier_model_params()
    if params.is_azure:
        wrapped = wrap_openai(client)
        init(wrapped)  # type: ignore
except Exception:
    # If client creation fails, individual tests will be skipped due to the fixture, so client = None is OK
    client = None


class _BedrockScore:
    """Minimal stand-in for autoevals' Score: exposes .score and .metadata."""

    def __init__(self, score, metadata):
        self.score = score
        self.metadata = metadata or {}


def _bedrock_evaluate_correctness(expected_elements_str, output, evaluation_type):
    """Judge via Bedrock through litellm, bypassing the OpenAI SDK + autoevals.

    Local-eval patch: HolmesGPT's classifier talks to OpenAI/Azure directly and
    cannot use Bedrock. We have no OpenAI key, so when BEDROCK_CLASSIFIER is set we
    score with a Bedrock model (default Claude Sonnet) instead. Returns the same
    .score (1/0) / .metadata{rationale} shape callers expect. NOTE: this judge is
    NOT gpt-4.1, so absolute scores may differ from the published leaderboard;
    relative pass/fail shape should hold.
    """
    import json as _json
    import litellm

    model = os.environ.get(
        "BEDROCK_CLASSIFIER_MODEL",
        "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    if evaluation_type == "loose":
        criteria = (
            "A: The OUTPUT reasonably matches the EXPECTED content\n"
            "B: The OUTPUT does not match the EXPECTED content"
        )
    else:
        criteria = (
            "A: All elements are present\n"
            "B: Either no element is present or only some but not all elements are present"
        )
    prompt = f"""You are evaluating the correctness of an OUTPUT given by an LLM.
Correctness is defined by the presence of EXPECTED ELEMENTS in the OUTPUT. An element
need not appear verbatim; its essence must be present, even across multiple sentences.

# EXPECTED ELEMENTS
- {expected_elements_str}

# OUTPUT
{output}

Reason step by step, then return your verdict as STRICT JSON on the final line:
{{"choice": "A" or "B", "rationale": "<one sentence>"}}
Choices:
{criteria}
"""
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0,
        aws_region_name=os.environ.get("AWS_REGION", "us-west-2"),
    )
    text = (resp.choices[0].message.content or "").strip()
    choice, rationale = "B", text[:300]
    # Parse the last JSON object in the response.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = _json.loads(line)
                choice = obj.get("choice", "B").strip().upper()[:1]
                rationale = obj.get("rationale", rationale)
                break
            except Exception:
                pass
    score = 1 if choice == "A" else 0
    return _BedrockScore(score, {"rationale": rationale, "judge_model": model})


def evaluate_correctness(
    expected_elements: Union[str, List[str]],
    output: Optional[str],
    parent_span: Optional[Span],
    caplog,
    evaluation_type: str = "strict",
):
    expected_elements_str = "\n- ".join(expected_elements)

    caplog.set_level("INFO", logger="classifier")
    logger = logging.getLogger("classifier")

    if isinstance(expected_elements, str):
        expected_elements = [expected_elements]
    expected_elements_str = "\n- ".join(expected_elements)

    # Bedrock judge path (local-eval patch; see _bedrock_evaluate_correctness).
    if os.environ.get("BEDROCK_CLASSIFIER"):
        logger.info("Evaluating correctness with Bedrock judge (local patch)")
        result = _bedrock_evaluate_correctness(
            expected_elements_str, output, evaluation_type
        )
        if parent_span:
            with parent_span.start_span(
                name="Correctness", type=SpanTypeAttribute.SCORE
            ) as span:
                span.log(
                    input="(bedrock judge)",
                    output=result.metadata.get("rationale", ""),
                    expected=expected_elements_str,
                    scores={"correctness": result.score},
                    metadata=result.metadata,
                )
        return result

    prompt_prefix = """
You are evaluating the correctness of an OUTPUT given by a LLM. You must return a score that
represents the correctness of that OUTPUT.

The correctness is defined by the presence of EXPECTED ELEMENTS in the OUTPUT.
Make a judgement call whether each ELEMENT sufficiently matches the OUTPUT. ELEMENTS do
not need to appear verbatim or be a perfect match but their essence should be
present in the whole OUTPUT, even if it spans multiple sentences.

# EXPECTED ELEMENTS

- {{expected}}

# OUTPUT

{{output}}


Return a choice based on the number of EXPECTED ELEMENTS present in the OUTPUT.
Possible choices:
- A: All elements are presents
- B: Either no element is present or only some but not all elements are present
"""

    if evaluation_type == "loose":
        prompt_prefix = """
You are evaluating the correctness of an OUTPUT given by a LLM. You must return a score that
represents the correctness of that OUTPUT.

The correctness is defined by the presence of EXPECTED in the OUTPUT.
Make a judgement call whether each ELEMENT sufficiently matches the OUTPUT. ELEMENTS do
not need to appear verbatim or be a perfect match but their essence should be
present in the whole OUTPUT, even if it spans multiple sentences.

# EXPECTED

{{expected}}

# OUTPUT

{{output}}


Return a choice based on the number of EXPECTED presence in the OUTPUT.
Possible choices:
- A: The OUTPUT reasonably matches the EXPECTED content
- B: The OUTPUT does not match the EXPECTED content
"""
    params = get_classifier_model_params()
    if params.is_azure:
        logger.info(
            f"Evaluating correctness with Azure AI Foundry; base_url={params.api_base}, api_version={params.api_version}, model={params.model}, api_key ending with: {params.api_key[-4:] if params.api_key else None}"
        )
        logger.info(
            "To use OpenAI instead, unset the environment variable AZURE_API_BASE"
        )
    else:
        logger.info(
            f"Evaluating correctness with OpenAI; model={params.model}, api_key ending with: {params.api_key[-4:] if params.api_key else None}"
        )
        logger.info(
            "To use Azure AI Foundry instead, set the environment variables AZURE_API_BASE, AZURE_API_VERSION, and AZURE_API_KEY"
        )

    classifier = LLMClassifier(
        name="Correctness",
        prompt_template=prompt_prefix,
        choice_scores={"A": 1, "B": 0},
        use_cot=True,
        model=params.model,
        api_key=params.api_key if not params.is_azure else None,
        base_url=params.api_base if not params.is_azure else None,
        api_version=params.api_version if not params.is_azure else None,
    )
    if parent_span:
        with parent_span.start_span(
            name="Correctness", type=SpanTypeAttribute.SCORE
        ) as span:
            correctness_eval = classifier(
                input=prompt_prefix, output=output, expected=expected_elements_str
            )

            span.log(
                input=prompt_prefix,
                output=correctness_eval.metadata.get("rationale", ""),
                expected=expected_elements_str,
                scores={
                    "correctness": correctness_eval.score,
                },
                metadata=correctness_eval.metadata,
            )
            return correctness_eval
    else:
        return classifier(
            input=prompt_prefix, output=output, expected=expected_elements_str
        )


