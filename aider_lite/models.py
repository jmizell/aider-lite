import difflib
import re
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import yaml

from aider_lite.dump import dump  # noqa: F401
from aider_lite.llm import apiclient

DEFAULT_MODEL_NAME = "anthropic/claude-3.5-sonnet"


@dataclass
class ModelSettings:
    # Model class needs to have each of these as well
    name: str
    edit_format: str = "whole"
    weak_model_name: Optional[str] = None
    use_repo_map: bool = False
    send_undo_reply: bool = False
    lazy: bool = False
    reminder: str = "user"
    examples_as_sys_msg: bool = False
    extra_params: Optional[dict] = None
    cache_control: bool = False
    caches_by_default: bool = False
    use_system_prompt: bool = True
    use_temperature: bool = True
    streaming: bool = True
    editor_model_name: Optional[str] = None
    editor_edit_format: Optional[str] = None


MODEL_SETTINGS = [
    ModelSettings(
        "meta-llama/llama-3.2-3b-instruct",
        "diff",
        weak_model_name="meta-llama/llama-3.2-3b-instruct",
        use_repo_map=False,
        send_undo_reply=False,
        examples_as_sys_msg=True,
    ),
    ModelSettings(
        "meta-llama/llama-3.2-70b-instruct",
        "diff",
        weak_model_name="meta-llama/llama-3.2-70b-instruct",
        use_repo_map=False,
        send_undo_reply=False,
        examples_as_sys_msg=True,
    ),
    ModelSettings(
        "meta-llama/llama-3.1-405b-instruct",
        "diff",
        weak_model_name="openrouter/meta-llama/llama-3.1-405b-instruct",
        use_repo_map=False,
        send_undo_reply=False,
        examples_as_sys_msg=True,
    ),
    ModelSettings(
        "anthropic/claude-3.5-sonnet",
        "diff",
        weak_model_name="anthropic/claude-3-5-haiku",
        editor_model_name="anthropic/claude-3.5-sonnet",
        editor_edit_format="editor-diff",
        use_repo_map=False,
        examples_as_sys_msg=True,
        reminder="user",
        cache_control=True,
    ),
    ModelSettings(
        "gpt-4o",
        "diff",
        weak_model_name="gpt-4o-mini",
        editor_model_name="gpt-4o",
        use_repo_map=False,
        examples_as_sys_msg=True,
        reminder="user",
        cache_control=True,
    ),
    ModelSettings(
        "gpt-4o-mini",
        "diff",
        use_repo_map=False,
        examples_as_sys_msg=True,
        reminder="user",
        cache_control=True,
    ),
]


class ModelInfoManager:
    CACHE_TTL = 60 * 60 * 24  # 24 hours

    def __init__(self):
        self.content = apiclient.model_parameters

    def get_model_info(self, model):
        info = self.content.get(model, dict())
        if info:
            return info
        return dict()


model_info_manager = ModelInfoManager()


class Model(ModelSettings):
    def __init__(self, model, weak_model=None, editor_model=None, editor_edit_format=None):
        self.name = model
        self.max_chat_history_tokens = 1024
        self.weak_model = None
        self.editor_model = None

        self.info = self.get_model_info(model)

        max_input_tokens = self.info.get("max_input_tokens") or 0
        if max_input_tokens < 32 * 1024:
            self.max_chat_history_tokens = 1024
        else:
            self.max_chat_history_tokens = 2 * 1024

        self.configure_model_settings(model)
        if weak_model is False:
            self.weak_model_name = None
        else:
            self.get_weak_model(weak_model)

        if editor_model is False:
            self.editor_model_name = None
        else:
            self.get_editor_model(editor_model, editor_edit_format)

    def get_model_info(self, model):
        return model_info_manager.get_model_info(model)

    def configure_model_settings(self, model):
        for ms in MODEL_SETTINGS:
            # direct match, or match "provider/<model>"
            if model == ms.name:
                for field in fields(ModelSettings):
                    val = getattr(ms, field.name)
                    setattr(self, field.name, val)
                return  # <--

        model = model.lower()

        if ("llama3" in model or "llama-3" in model) and "70b" in model:
            self.edit_format = "diff"
            self.use_repo_map = False
            self.send_undo_reply = True
            self.examples_as_sys_msg = True
            return  # <--

        # use the defaults
        if self.edit_format == "diff":
            self.use_repo_map = False

    def __str__(self):
        return self.name

    def get_weak_model(self, provided_weak_model_name):
        # If weak_model_name is provided, override the model settings
        if provided_weak_model_name:
            self.weak_model_name = provided_weak_model_name

        if not self.weak_model_name:
            self.weak_model = self
            return

        if self.weak_model_name == self.name:
            self.weak_model = self
            return

        self.weak_model = Model(
            self.weak_model_name,
            weak_model=False,
        )
        return self.weak_model

    def commit_message_models(self):
        return [self.weak_model, self]

    def get_editor_model(self, provided_editor_model_name, editor_edit_format):
        # If editor_model_name is provided, override the model settings
        if provided_editor_model_name:
            self.editor_model_name = provided_editor_model_name
        if editor_edit_format:
            self.editor_edit_format = editor_edit_format

        if not self.editor_model_name or self.editor_model_name == self.name:
            self.editor_model = self
        else:
            self.editor_model = Model(
                self.editor_model_name,
                editor_model=False,
            )

        if not self.editor_edit_format:
            self.editor_edit_format = self.editor_model.edit_format

        return self.editor_model

    def token_count(self, messages):
        """
        Estimate the number of tokens in a text string or list of messages.
        Wildly inaccurate, only a guess.
        """
        if isinstance(messages, list):
            text = " ".join(str(message) for message in messages)
        else:
            text = str(messages)

        # Split text on spaces and punctuation to approximate tokens
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return len(tokens)


def register_models(model_settings_fnames):
    files_loaded = []
    for model_settings_fname in model_settings_fnames:
        if not os.path.exists(model_settings_fname):
            continue

        try:
            with open(model_settings_fname, "r") as model_settings_file:
                model_settings_list = yaml.safe_load(model_settings_file)

            for model_settings_dict in model_settings_list:
                model_settings = ModelSettings(**model_settings_dict)
                existing_model_settings = next(
                    (ms for ms in MODEL_SETTINGS if ms.name == model_settings.name), None
                )

                if existing_model_settings:
                    MODEL_SETTINGS.remove(existing_model_settings)
                MODEL_SETTINGS.append(model_settings)
        except Exception as e:
            raise Exception(f"Error loading model settings from {model_settings_fname}: {e}")
        files_loaded.append(model_settings_fname)

    return files_loaded


def validate_variables(vars):
    missing = []
    for var in vars:
        if var not in os.environ:
            missing.append(var)
    if missing:
        return dict(keys_in_environment=False, missing_keys=missing)
    return dict(keys_in_environment=True, missing_keys=missing)


def sanity_check_models(io, main_model):
    problem_main = sanity_check_model(io, main_model)

    problem_weak = None
    if main_model.weak_model and main_model.weak_model is not main_model:
        problem_weak = sanity_check_model(io, main_model.weak_model)

    problem_editor = None
    if (
            main_model.editor_model
            and main_model.editor_model is not main_model
            and main_model.editor_model is not main_model.weak_model
    ):
        problem_editor = sanity_check_model(io, main_model.editor_model)

    return problem_main or problem_weak or problem_editor


def sanity_check_model(io, model):
    show = False

    if not model.info:
        show = True
        io.tool_warning(
            f"Warning for {model}: Unknown context window size using sane defaults."
        )

        possible_matches = fuzzy_match_models(model.name)
        if possible_matches:
            io.tool_output("Did you mean one of these?")
            for match in possible_matches:
                io.tool_output(f"- {match}")

    return show


def fuzzy_match_models(name):
    name = name.lower()

    chat_models = set()
    for model, attrs in apiclient.model_parameters.items():
        model = model.lower()
        if attrs.get("mode") != "chat":
            continue
        provider = (attrs["litellm_provider"] + "/").lower()

        if model.startswith(provider):
            fq_model = model
        else:
            fq_model = provider + model

        chat_models.add(fq_model)
        chat_models.add(model)

    chat_models = sorted(chat_models)
    # exactly matching model
    # matching_models = [
    #    (fq,m) for fq,m in chat_models
    #    if name == fq or name == m
    # ]
    # if matching_models:
    #    return matching_models

    # Check for model names containing the name
    matching_models = [m for m in chat_models if name in m]
    if matching_models:
        return sorted(set(matching_models))

    # Check for slight misspellings
    models = set(chat_models)
    matching_models = difflib.get_close_matches(name, models, n=3, cutoff=0.8)

    return sorted(set(matching_models))


def print_matching_models(io, search):
    matches = fuzzy_match_models(search)
    if matches:
        io.tool_output(f'Models which match "{search}":')
        for model in matches:
            io.tool_output(f"- {model}")
    else:
        io.tool_output(f'No models match "{search}".')


def get_model_settings_as_yaml():
    model_settings_list = []
    for ms in MODEL_SETTINGS:
        model_settings_dict = {
            field.name: getattr(ms, field.name) for field in fields(ModelSettings)
        }
        model_settings_list.append(model_settings_dict)

    return yaml.dump(model_settings_list, default_flow_style=False)


def register_apiclient_models(model_metadata_files):
    """Load model metadata files and update apiclient.model_parameters with the metadata."""
    files_loaded = []

    for fname in model_metadata_files:
        try:
            if not Path(fname).exists():
                continue

            with open(fname, "r") as f:
                metadata = yaml.safe_load(f)
                if metadata:
                    apiclient.model_parameters.update(metadata)
                    files_loaded.append(fname)
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    return files_loaded


def main():
    if len(sys.argv) < 2:
        print("Usage: python models.py <model_name> or python models.py --yaml")
        sys.exit(1)

    if sys.argv[1] == "--yaml":
        yaml_string = get_model_settings_as_yaml()
        print(yaml_string)
    else:
        model_name = sys.argv[1]
        matching_models = fuzzy_match_models(model_name)

        if matching_models:
            print(f"Matching models for '{model_name}':")
            for model in matching_models:
                print(model)
        else:
            print(f"No matching models found for '{model_name}'.")


if __name__ == "__main__":
    main()
