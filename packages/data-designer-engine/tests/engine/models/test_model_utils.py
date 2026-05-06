# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from data_designer.engine.models.utils import ChatMessage, prompt_to_messages
from data_designer.engine.processing.ginja.record import sanitize_record


def test_prompt_to_messages() -> None:
    stub_system_prompt = "some system prompt"
    mult_modal_context = {"type": "image_url", "image_url": {"url": "http://example.com/image.png"}}
    assert prompt_to_messages(user_prompt="hello") == [ChatMessage.as_user("hello")]
    assert prompt_to_messages(user_prompt="hello", system_prompt=stub_system_prompt) == [
        ChatMessage.as_system(stub_system_prompt),
        ChatMessage.as_user("hello"),
    ]
    assert prompt_to_messages(user_prompt="hello", multi_modal_context=[mult_modal_context]) == [
        ChatMessage.as_user([mult_modal_context, {"type": "text", "text": "hello"}])
    ]
    assert prompt_to_messages(
        user_prompt="hello", system_prompt=stub_system_prompt, multi_modal_context=[mult_modal_context]
    ) == [
        ChatMessage.as_system(stub_system_prompt),
        ChatMessage.as_user([mult_modal_context, {"type": "text", "text": "hello"}]),
    ]


def test_chat_message_trace_dict_is_json_safe_for_record_rendering() -> None:
    @dataclass
    class ToolArguments:
        query: str

    message = ChatMessage.as_assistant(
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "arguments": ToolArguments(query="ray backend"),
                },
            }
        ]
    )

    record = {
        "topic_candidates": {"topics": ["ray"]},
        "topic_candidates__trace": [message.to_dict()],
    }

    sanitized = sanitize_record(record)

    assert sanitized["topic_candidates"]["topics"] == ["ray"]
    assert sanitized["topic_candidates__trace"][0]["tool_calls"][0]["function"]["arguments"] == {"query": "ray backend"}
