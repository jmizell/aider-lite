import unittest
from unittest.mock import ANY, MagicMock, patch

from aider_lite.models import (
    Model,
    ModelInfoManager,
    sanity_check_model,
    sanity_check_models,
)


class TestModels(unittest.TestCase):
    def test_get_model_info_nonexistent(self):
        manager = ModelInfoManager()
        info = manager.get_model_info("non-existent-model")
        self.assertEqual(info, {})

    def test_max_context_tokens(self):
        model = Model("meta-llama/llama-3.1-405b-instruct")
        self.assertEqual(model.info["max_input_tokens"], 128000)

    @patch("os.environ")
    def test_sanity_check_model_all_set(self, mock_environ):
        mock_environ.get.return_value = "dummy_value"
        mock_io = MagicMock()
        model = MagicMock()
        model.name = "test-model"
        model.info = {"some": "info"}
        sanity_check_model(mock_io, model)

    def test_sanity_check_models_bogus_editor(self):
        mock_io = MagicMock()
        main_model = Model("gpt-4")
        main_model.editor_model = Model("bogus-model")

        result = sanity_check_models(mock_io, main_model)

        self.assertTrue(
            result
        )  # Should return True because there's a problem with the editor model
        mock_io.tool_warning.assert_called_with(ANY)  # Ensure a warning was issued

        warning_messages = [call.args[0] for call in mock_io.tool_warning.call_args_list]
        print("Warning messages:", warning_messages)  # Add this line

        self.assertGreaterEqual(mock_io.tool_warning.call_count, 1)  # Expect two warnings
        self.assertTrue(
            any("bogus-model" in msg for msg in warning_messages)
        )  # Check that one of the warnings mentions the bogus model


if __name__ == "__main__":
    unittest.main()
