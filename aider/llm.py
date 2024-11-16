import json
import os

import requests
import warnings
from typing import Optional, Union, List, Literal, Dict, Any

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

model_costs = {
    "sample_spec": {
        "max_tokens": "set to max_output_tokens if provider specifies it. IF not set to max_tokens provider specifies",
        "max_input_tokens": "max input tokens, if the provider specifies it. if not default to max_tokens",
        "max_output_tokens": "max output tokens, if the provider specifies it. if not default to max_tokens",
        "input_cost_per_token": 0.0000,
        "output_cost_per_token": 0.000,
        "litellm_provider": "one of https://docs.litellm.ai/docs/providers",
        "mode": "one of chat, embedding, completion, image_generation, audio_transcription, audio_speech",
        "supports_function_calling": True,
        "supports_parallel_function_calling": True,
        "supports_vision": True,
        "supports_audio_input": True,
        "supports_audio_output": True,
        "supports_prompt_caching": True
    },
    "meta-llama/llama-3.2-3b-instruct": {
        "max_tokens": 128000,
        "input_cost_per_token": 0.00000059,
        "output_cost_per_token": 0.00000079,
        "edit_format": "diff",
        "use_repo_map": True,
        "send_undo_reply": True,
        "examples_as_sys_msg": True,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "max_tokens": 128000,
        "input_cost_per_token": 0.00000059,
        "output_cost_per_token": 0.00000079,
        "edit_format": "diff",
        "use_repo_map": True,
        "send_undo_reply": True,
        "examples_as_sys_msg": True,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "meta-llama/llama-3.1-405b-instruct": {
        "max_tokens": 128000,
        "max_input_tokens": 128000,
        "input_cost_per_token": 0.00000059,
        "output_cost_per_token": 0.00000079,
        "edit_format": "diff",
        "use_repo_map": True,
        "send_undo_reply": True,
        "examples_as_sys_msg": True,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
    "anthropic/claude-3.5-sonnet": {
        "max_tokens": 128000,
        "input_cost_per_token": 0.00000059,
        "output_cost_per_token": 0.00000079,
        "edit_format": "diff",
        "use_repo_map": True,
        "send_undo_reply": True,
        "examples_as_sys_msg": True,
        "litellm_provider": "openrouter",
        "mode": "chat"
    },
}


# Define a class to represent function calls used in the chat completion
class FunctionCall:
    def __init__(self, arguments: str, name: Optional[str] = None):
        # Initialize function call with its arguments and optional name
        self.arguments = arguments
        self.name = name


# Define a class for functions which may be used within the response
class Function:
    def __init__(self, arguments: Optional[Union[Dict, str]] = None, name: Optional[str] = None):
        # Initialize function with its arguments and optional name
        self.arguments = arguments
        self.name = name


# Define a class representing tool calls within a chat completion message
class ChatCompletionMessageToolCall:
    def __init__(self, function: Union[Dict, Function], id: Optional[str] = None, type: Optional[str] = None):
        # Initialize tool call with a function, and optionally an id and type
        self.function = function
        self.id = id
        self.type = type


# Define a class representing a message in the chat
class Message:
    def __init__(self, content: Optional[str], role: Literal["assistant", "user", "system", "tool", "function"],
                 tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None,
                 function_call: Optional[FunctionCall] = None):
        # Initialize message with content, role, optional tool calls, and optional function call
        self.content = content
        self.role = role
        self.tool_calls = tool_calls
        self.function_call = function_call


# Define a class for choices in the model response
class Choices:
    def __init__(self, finish_reason=None, index=0, message: Optional[Union[Message, dict]] = None,
                 logprobs=None, enhancements=None):
        # Initialize choices with finish reason, index, message, optional logprobs and enhancements
        self.finish_reason = finish_reason
        self.index = index
        self.message = message
        self.logprobs = logprobs
        self.enhancements = enhancements


# Define a class for delta tool calls in streaming chat completion
class ChatCompletionDeltaToolCall:
    def __init__(self, id: Optional[str] = None, function: Function = None,
                 type: Optional[str] = None, index: int = 0):
        # Initialize delta tool call with optional id, function, type, and index
        self.id = id
        self.function = function
        self.type = type
        self.index = index


# Define a class representing the delta in streaming chat completions
class Delta:
    def __init__(self, content=None, role=None,
                 function_call: Optional[Union[FunctionCall, Any]] = None,
                 tool_calls: Optional[List[Union[ChatCompletionDeltaToolCall, Any]]] = None):
        # Initialize delta with content, role, optional function call and tool calls
        self.content = content
        self.role = role
        self.function_call = function_call
        self.tool_calls = tool_calls


# Define a class for streaming choices in model response
class StreamingChoices:
    def __init__(self, finish_reason=None, index=0, delta: Optional[Delta] = None,
                 logprobs=None, enhancements=None):
        # Initialize streaming choices with finish reason, index, delta, optional logprobs and enhancements
        self.finish_reason = finish_reason
        self.index = index
        self.delta = delta
        self.logprobs = logprobs
        self.enhancements = enhancements


# Define a class to encapsulate the response from the model
class ModelResponse:
    choices: List[Union[Choices, StreamingChoices]]

    def __init__(self, choices: List[Union[Choices, StreamingChoices]]) -> None:
        # Initialize ModelResponse with a list of choices
        self.choices = choices

    def __iter__(self):
        # Allow iteration over choices
        if isinstance(self.choices, (list, tuple)):
            for choice in self.choices:
                yield ModelResponse(choices=[choice])
        else:
            for choice in self.choices:
                yield ModelResponse(choices=[choice])

    def __next__(self):
        # Return the next choice
        if isinstance(self.choices, (list, tuple)):
            return next(iter(self.choices))
        return next(self.choices)


# Define a class for interacting with the API
class ApiClient:
    model_cost = model_costs

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE")
        if not self.api_base:
            self.api_base = "https://api.openai.com"

    def completion(
            self,
            model: str,
            messages: list,
            stream: bool,
            temperature: int,
            tools: list = None,
            tool_choice: dict = None,
            max_tokens: int = None,
    ) -> ModelResponse:
        print("DEBUG -- -- completion call")

        # Prepare the request body for the API call
        request_body = {
            "model": model,
            "temperature": temperature,
            "stream": stream,
            "messages": messages,
        }
        if max_tokens:
            request_body["max_tokens"] = max_tokens
            request_body["n_tokens"] = max_tokens
        if tools:
            request_body["tools"] = tools
            # ignoring tool choices as callers to completion are always one tool
            request_body["tool_choice"] = "auto"
            request_body["parallel_tool_calls"] = False

        print(json.dumps(request_body, sort_keys=True, indent=2))  # DEBUG -- -- remove

        # Set request headers, including authentication
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key

        # Make a POST request to the API
        response = requests.post(
            self.api_base + "/v1/chat/completions",
            headers=headers,
            json=request_body
        )

        # Raise an error if the response is unsuccessful
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")

        # Handle streaming responses by generating streaming choices
        def generate_streaming_choices():
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        decoded_line = decoded_line[6:].strip()
                    if "OPENROUTER PROCESSING" in decoded_line or "[DONE]" in decoded_line:
                        continue
                    chunk = json.loads(decoded_line)
                    print("DEBUG -- -- ", decoded_line)
                    for c in chunk["choices"]:
                        # Handle tool calls in streaming response
                        tool_calls = None
                        if "tool_calls" in c.get("delta", {}):
                            tool_calls = []
                            for tool_call in c["delta"]["tool_calls"]:
                                function = Function(
                                    name=tool_call.get("function", {}).get("name"),
                                    arguments=tool_call.get("function", {}).get("arguments")
                                )
                                tool_calls.append(ChatCompletionDeltaToolCall(
                                    function=function,
                                    id=tool_call.get("id"),
                                    type=tool_call.get("type"),
                                    index=c["index"]
                                ))

                        yield StreamingChoices(
                            finish_reason=c["finish_reason"],
                            index=c["index"],
                            delta=Delta(
                                content=c["delta"].get("content"),
                                role=c["delta"].get("role"),
                                function_call=c["delta"].get("function_call"),
                                tool_calls=tool_calls
                            ),
                            logprobs=c.get("logprobs"),
                            enhancements=None
                        )

        if stream:
            return ModelResponse(choices=generate_streaming_choices())

        # For non-streaming responses, convert the response into a ModelResponse object
        print("DEBUG -- -- ", response.content)
        choices = []
        for choice_data in response.json()["choices"]:
            # Handle tool calls if present
            tool_calls = None
            if "tool_calls" in choice_data["message"]:
                tool_calls = []
                for tool_call in choice_data["message"]["tool_calls"]:
                    function = Function(
                        name=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"]
                    )
                    tool_calls.append(ChatCompletionMessageToolCall(
                        function=function,
                        id=tool_call["id"],
                        type=tool_call["type"]
                    ))

            message = Message(
                content=choice_data["message"].get("content"),
                role=choice_data["message"]["role"],
                tool_calls=tool_calls
            )
            choice = Choices(
                finish_reason=choice_data["finish_reason"],
                index=choice_data["index"],
                message=message
            )
            choices.append(choice)

        return ModelResponse(choices=choices)


apiclient = ApiClient()

__all__ = [apiclient]
