"""
UserAgentLlmFn - Business Logic

Simulated user agent for pipeline testing. Generates realistic responses
as if a real user were interacting via WhatsApp/email. Receives only
user-visible conversation (no internal events) and generates the next
response based on configured persona, goals, and guidelines.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from api.orchestrator_requests import orchestrator_api_manager
from chask_foundation.backend.models import OrchestrationEvent
from chask_foundation.llm import LLMClient

logger = logging.getLogger(__name__)


CHANNEL_INSTRUCTIONS = {
    "whatsapp": (
        "You are chatting via WhatsApp. Keep messages short and conversational. "
        "Use informal language. You may use line breaks instead of paragraphs. "
        "Do not write long blocks of text."
    ),
    "email": (
        "You are communicating via email. Use a slightly more formal tone. "
        "Include a brief greeting and sign off naturally. "
        "You may write slightly longer responses than chat."
    ),
    "webhook": (
        "You are responding via a webhook. Keep the response concise and direct."
    ),
}


class FunctionBackend:
    """Backend for generating simulated user responses."""

    def __init__(
        self,
        orchestration_event: OrchestrationEvent,
        model: str = "gpt-4.1-mini",
        openai_api_key: str = None,
    ):
        self.orchestration_event = orchestration_event
        self.model = model
        self.openai_api_key = openai_api_key
        self.response_event_sent = False

    def process_request(self) -> str:
        """
        Generate a simulated user response.

        1. Extract and validate parameters
        2. Check pre_defined_responses for a pattern match (deterministic layer)
        3. If no match, call LLM to generate a natural response
        """
        tool_args = self._extract_tool_args()

        # Required params
        persona = tool_args.get("persona")
        goals = tool_args.get("goals")
        conversation_history_raw = tool_args.get("conversation_history")
        current_outbound_message = tool_args.get("current_outbound_message")
        channel_type = tool_args.get("channel_type", "whatsapp")

        # Optional params
        response_guidelines = tool_args.get("response_guidelines", "")
        pre_defined_responses_raw = tool_args.get("pre_defined_responses")

        # Validate required
        missing = []
        if not persona:
            missing.append("persona")
        if not goals:
            missing.append("goals")
        if not conversation_history_raw:
            missing.append("conversation_history")
        if not current_outbound_message:
            missing.append("current_outbound_message")
        if missing:
            error_msg = f"Missing required parameters: {', '.join(missing)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check pre-defined responses first (deterministic layer)
        pre_defined_match = self._check_pre_defined_responses(
            pre_defined_responses_raw, current_outbound_message
        )
        if pre_defined_match is not None:
            logger.info("Pre-defined response matched, skipping LLM call")
            self._send_response(pre_defined_match)
            return pre_defined_match

        # Parse conversation history
        conversation_history = self._parse_conversation_history(conversation_history_raw)

        # Build LLM messages
        messages = self._build_llm_messages(
            persona=persona,
            goals=goals,
            conversation_history=conversation_history,
            current_outbound_message=current_outbound_message,
            channel_type=channel_type,
            response_guidelines=response_guidelines,
        )

        # Call LLM
        llm_client = LLMClient(
            access_token=self.orchestration_event.access_token,
            organization_id=self.orchestration_event.organization.organization_id,
            orchestration_session_uuid=self.orchestration_event.orchestration_session_uuid,
            internal_orchestration_session_uuid=self.orchestration_event.internal_orchestration_session_uuid,
            orchestration_event_uuid=str(self.orchestration_event.event_id),
            default_model=self.model,
            openai_api_key=self._get_openai_api_key(),
        )

        try:
            response = llm_client.chat(
                messages=messages,
                temperature=0.9,
                caller_function="UserAgentLlmFn.process_request.user_agent_context",
            )

            if not response.get("success"):
                error_msg = f"LLM call failed: {response.get('error', 'Unknown error')}"
                logger.error(error_msg)
                self._send_response(error_msg, is_error=True)
                raise RuntimeError(error_msg)

            result = response["content"]
            logger.info("User agent response generated successfully")

            self._send_response(result)
            return result

        finally:
            llm_client.shutdown()

    # ── Pre-defined response matching ─────────────────────────────────

    def _check_pre_defined_responses(
        self, pre_defined_raw: Optional[str], outbound_message: str
    ) -> Optional[str]:
        """Check if any pre-defined response pattern matches the outbound message."""
        if not pre_defined_raw:
            return None

        try:
            pre_defined = json.loads(pre_defined_raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse pre_defined_responses JSON")
            return None

        if not isinstance(pre_defined, list):
            return None

        for entry in pre_defined:
            pattern = entry.get("pattern")
            response_text = entry.get("response")
            if not pattern or not response_text:
                continue

            try:
                if re.search(pattern, outbound_message, re.IGNORECASE):
                    file_url = entry.get("file_url")
                    if file_url:
                        return f"{response_text}\n[file: {file_url}]"
                    return response_text
            except re.error:
                logger.warning(f"Invalid regex pattern: {pattern}")
                continue

        return None

    # ── LLM message building ──────────────────────────────────────────

    def _build_llm_messages(
        self,
        persona: str,
        goals: str,
        conversation_history: List[Dict[str, str]],
        current_outbound_message: str,
        channel_type: str,
        response_guidelines: str,
    ) -> List[Dict[str, str]]:
        """Build the message array for the LLM call."""
        channel_instruction = CHANNEL_INSTRUCTIONS.get(
            channel_type, CHANNEL_INSTRUCTIONS["whatsapp"]
        )

        system_parts = [
            f"You are {persona}.",
            f"Your goals: {goals}.",
            channel_instruction,
        ]
        if response_guidelines:
            system_parts.append(f"Additional guidelines: {response_guidelines}.")
        system_parts.append(
            "Respond naturally as this person would. "
            "Do NOT reveal you are an AI or a simulated user. "
            "Do NOT mention that you are following instructions or guidelines."
        )

        messages = [{"role": "system", "content": " ".join(system_parts)}]

        # Add conversation history as alternating messages
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "user", "content": content})

        # Add the current outbound message as the latest assistant message
        messages.append({"role": "assistant", "content": current_outbound_message})

        return messages

    def _parse_conversation_history(self, raw: str) -> List[Dict[str, str]]:
        """Parse conversation_history JSON string into a list of messages."""
        try:
            history = json.loads(raw)
            if isinstance(history, list):
                return history
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse conversation_history JSON")
        return []

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_openai_api_key(self) -> str:
        return self.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    def _extract_tool_args(self) -> Dict[str, Any]:
        """Extract tool call arguments from orchestration event."""
        extra_params = self.orchestration_event.extra_params or {}
        tool_calls = extra_params.get("tool_calls", [])

        if not tool_calls:
            logger.warning("No tool calls found in orchestration event")
            return {}

        tool_call = tool_calls[0]
        return tool_call.get("args", {})

    def _send_response(self, message: str, is_error: bool = False) -> bool:
        """Send the function result back to the orchestrator via Kafka."""
        try:
            original_extra_params = self.orchestration_event.extra_params or {}
            tool_call_id = None
            tool_name = None
            if ("tool_calls" in original_extra_params and
                original_extra_params["tool_calls"]):
                tool_call = original_extra_params["tool_calls"][0]
                tool_call_id = tool_call.get("id")
                tool_name = tool_call.get("name")

            extra_params = {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "is_error": is_error
            }

            # ====================================================================
            # CRITICAL: Preserve test execution flags
            # ====================================================================
            if original_extra_params.get("is_test"):
                extra_params["is_test"] = True
                if original_extra_params.get("test_execution_uuid"):
                    extra_params["test_execution_uuid"] = original_extra_params["test_execution_uuid"]
            # ====================================================================

            evolve_response = orchestrator_api_manager.call(
                "evolve_event",
                parent_event_uuid=str(self.orchestration_event.event_id),
                event_type="function_call_response",
                source="agent",
                target="orchestrator",
                prompt=message,
                extra_params=extra_params,
                access_token=self.orchestration_event.access_token,
                organization_id=self.orchestration_event.organization.organization_id,
            )

            if evolve_response.get("status_code") not in (200, 201):
                error_msg = evolve_response.get("error", "Unknown error")
                raise Exception(f"Failed to evolve event: {error_msg}")

            evolved_uuid = evolve_response.get("uuid")
            if not evolved_uuid:
                raise Exception("API response missing uuid for evolved event")

            response_event = self.orchestration_event.model_copy(deep=True)
            response_event.event_id = evolved_uuid
            response_event.event_type = "function_call_response"
            response_event.source = "agent"
            response_event.target = "orchestrator"
            response_event.prompt = message
            response_event.extra_params = evolve_response.get("extra_params", extra_params)

            orchestrator_api_manager.call(
                "forward_oe_to_kafka",
                orchestration_event=response_event.model_dump(),
                topic="orchestrator",
                access_token=response_event.access_token,
                organization_id=response_event.organization.organization_id,
            )

            logger.info(
                f"Response sent to orchestrator via Kafka "
                f"[evolved from {self.orchestration_event.event_id} -> {evolved_uuid}]"
            )
            self.response_event_sent = True
            return True

        except Exception as e:
            logger.error(f"Failed to send response to orchestrator: {e}")
            return False

    def _extract_widget_params(self, param_names: list) -> Dict[str, Any]:
        """Extract widget parameters supporting both production and test formats."""
        widget_data = self.orchestration_event.extra_params.get("widget_data", {})

        widgets = widget_data.get("widgets", [])
        widget_values = {w.get("name"): w.get("value") for w in widgets}

        result = {}
        for param_name in param_names:
            result[param_name] = widget_values.get(param_name) or widget_data.get(param_name)

        return result
