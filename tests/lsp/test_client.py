"""
Tests for aider_lite.lsp.client module.

This module tests the LSP client functionality including JSON-RPC communication,
threading, error handling, and LSP method implementations.
"""

import io
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from aider_lite.lsp.client import LspClient, to_uri, StderrHandler
from aider_lite.lsp.types import *


# Fixtures
@pytest.fixture
def mock_process():
    """Create a mock subprocess for testing."""
    mock_proc = Mock()
    mock_proc.stdin = Mock()
    mock_proc.stdout = Mock()
    mock_proc.stderr = Mock()
    mock_proc.poll.return_value = None  # Process is running
    mock_proc.returncode = None
    return mock_proc


@pytest.fixture
def mock_condition():
    """Create a mock condition that supports context manager."""
    condition = Mock()
    condition.__enter__ = Mock(return_value=condition)
    condition.__exit__ = Mock(return_value=None)
    return condition


# Test utility functions
def test_to_uri_absolute_path():
    """Test to_uri with absolute path."""
    result = to_uri("/home/user/test.py")
    assert result == "file:///home/user/test.py"


def test_to_uri_relative_path():
    """Test to_uri with relative path."""
    result = to_uri("test.py")
    expected = f"file://{Path('test.py').absolute()}"
    assert result == expected


def test_stderr_handler_values():
    """Test StderrHandler enum values."""
    assert StderrHandler.IGNORE == "ignore"
    assert StderrHandler.LOG == "log"
    assert StderrHandler.PRINT == "print"


# Test LspClient initialization
@patch('aider_lite.lsp.client.subprocess.Popen')
def test_init_default_params(mock_popen, mock_process):
    """Test LspClient initialization with default parameters."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    assert client.lsp_cmd == ["pylsp"]
    assert client.timeout == 60.0
    assert client.stderr_handler == StderrHandler.LOG
    assert client.shutdown_flag is False
    assert client.next_id == 0


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_init_custom_params(mock_popen, mock_process):
    """Test LspClient initialization with custom parameters."""
    mock_popen.return_value = mock_process
    
    client = LspClient(
        lsp_cmd=["custom-lsp"],
        timeout=30.0,
        stderr_handler=StderrHandler.PRINT
    )
    assert client.lsp_cmd == ["custom-lsp"]
    assert client.timeout == 30.0
    assert client.stderr_handler == StderrHandler.PRINT


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_init_invalid_stderr_handler(mock_popen, mock_process):
    """Test LspClient initialization with invalid stderr handler."""
    mock_popen.return_value = mock_process
    
    with pytest.raises(ValueError, match="Invalid stderr_handler"):
        LspClient(stderr_handler="invalid")


# Test message handling
@patch('aider_lite.lsp.client.subprocess.Popen')
def test_read_message_valid(mock_popen, mock_process):
    """Test reading a valid LSP message."""
    mock_popen.return_value = mock_process
    
    message = {"jsonrpc": "2.0", "method": "test", "params": {}}
    message_json = json.dumps(message)
    content_length = len(message_json.encode('utf-8'))
    
    mock_stdout = io.StringIO(f"Content-Length: {content_length}\r\n\r\n{message_json}")
    mock_process.stdout = mock_stdout
    
    client = LspClient()
    result = client._read_message()
    assert result == message


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_read_message_missing_content_length(mock_popen, mock_process):
    """Test reading message without Content-Length header."""
    mock_popen.return_value = mock_process
    
    mock_stdout = io.StringIO("Invalid-Header: 123\r\n\r\n{}")
    mock_process.stdout = mock_stdout
    
    client = LspClient()
    result = client._read_message()
    assert result is None


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_read_message_invalid_content_length(mock_popen, mock_process):
    """Test reading message with invalid Content-Length."""
    mock_popen.return_value = mock_process
    
    mock_stdout = io.StringIO("Content-Length: invalid\r\n\r\n{}")
    mock_process.stdout = mock_stdout
    
    client = LspClient()
    result = client._read_message()
    assert result is None


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_read_message_too_large(mock_popen, mock_process):
    """Test reading message that's too large."""
    mock_popen.return_value = mock_process
    
    mock_stdout = io.StringIO("Content-Length: 999999999\r\n\r\n{}")
    mock_process.stdout = mock_stdout
    
    client = LspClient()
    result = client._read_message()
    assert result is None


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_read_message_invalid_json(mock_popen, mock_process):
    """Test reading message with invalid JSON."""
    mock_popen.return_value = mock_process
    
    invalid_json = "{ invalid json }"
    content_length = len(invalid_json.encode('utf-8'))
    
    mock_stdout = io.StringIO(f"Content-Length: {content_length}\r\n\r\n{invalid_json}")
    mock_process.stdout = mock_stdout
    
    client = LspClient()
    result = client._read_message()
    assert result is None


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_write_message(mock_popen, mock_process):
    """Test writing an LSP message."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    message = {"jsonrpc": "2.0", "method": "test"}
    
    client._write_message(message)
    
    # Verify stdin.write was called
    mock_process.stdin.write.assert_called()
    mock_process.stdin.flush.assert_called()


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_write_message_after_shutdown(mock_popen, mock_process):
    """Test writing message after shutdown."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    client.shutdown_flag = True
    
    message = {"jsonrpc": "2.0", "method": "test"}
    client._write_message(message)
    
    # Should not call stdin.write when shutdown
    mock_process.stdin.write.assert_not_called()


# Test LSP method calls
@patch('aider_lite.lsp.client.subprocess.Popen')
def test_call_method_success(mock_popen, mock_process, mock_condition):
    """Test successful method call."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    client._send_request = Mock()
    
    result = {"capabilities": {}}
    mock_condition.wait.return_value = True
    
    with patch.object(client, 'dict_lock'), \
         patch.object(client, 'event_dict', {0: mock_condition}), \
         patch.object(client, 'response_dict', {0: (result, None)}), \
         patch('threading.Condition', return_value=mock_condition):
        
        response = client.call_method("initialize", rootPath="/test")
    
    assert response == result


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_call_method_timeout(mock_popen, mock_process, mock_condition):
    """Test method call timeout."""
    mock_popen.return_value = mock_process
    
    client = LspClient(timeout=0.1)
    client._send_request = Mock()
    
    mock_condition.wait.return_value = False
    
    with patch('threading.Condition', return_value=mock_condition):
        with pytest.raises(TimeoutError):
            client.call_method("initialize", rootPath="/test")


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_call_method_error_response(mock_popen, mock_process, mock_condition):
    """Test method call with error response."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    client._send_request = Mock()
    
    error = {"code": -32601, "message": "Method not found"}
    mock_condition.wait.return_value = True
    
    with patch.object(client, 'dict_lock'), \
         patch.object(client, 'event_dict', {0: mock_condition}), \
         patch.object(client, 'response_dict', {0: (None, error)}), \
         patch('threading.Condition', return_value=mock_condition):
        
        with pytest.raises(LanguageServerResponseError):
            client.call_method("nonexistent")


# Test specific LSP methods
@patch('aider_lite.lsp.client.subprocess.Popen')
def test_lsp_initialize(mock_popen, mock_process):
    """Test LSP initialize method."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    client.call_method = Mock(return_value={"capabilities": {}})
    
    result = client.lsp_initialize("/test/path")
    
    client.call_method.assert_called_once_with(
        "initialize",
        processId=None,
        rootPath="/test/path",
        rootUri="file:///test/path",
        capabilities=DEFAULT_CAPABILITIES
    )
    assert result == {"capabilities": {}}


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_lsp_did_open(mock_popen, mock_process):
    """Test LSP didOpen notification."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    client.send_notification = Mock()
    
    doc = TextDocumentItem(
        uri="file:///test.py",
        languageId=LanguageIdentifier.PYTHON,
        version=1,
        text="print('hello')"
    )
    
    client.lsp_did_open(doc)
    
    client.send_notification.assert_called_once_with(
        "textDocument/didOpen",
        textDocument=doc.to_dict()
    )


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_send_notification(mock_popen, mock_process):
    """Test sending notification."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    client._send_notification = Mock()
    
    client.send_notification("test/method", param1="value1")
    
    client._send_notification.assert_called_once_with(
        "test/method",
        {"param1": "value1"}
    )


# Test document symbols
@patch('aider_lite.lsp.client.subprocess.Popen')
def test_lsp_document_symbols_empty_response(mock_popen, mock_process):
    """Test document symbols with empty response."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    client.call_method = Mock(return_value=[])
    
    doc_id = TextDocumentIdentifier(uri="file:///test.py")
    result = client.lsp_document_symbols(doc_id)
    
    assert result == []


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_lsp_document_symbols_document_symbol_response(mock_popen, mock_process):
    """Test document symbols with DocumentSymbol response."""
    mock_popen.return_value = mock_process
    
    symbol_data = {
        "name": "test_function",
        "kind": 12,
        "range": {
            "start": {"line": 1, "character": 0},
            "end": {"line": 5, "character": 0}
        },
        "selectionRange": {
            "start": {"line": 1, "character": 4},
            "end": {"line": 1, "character": 17}
        }
    }
    
    client = LspClient()
    client.call_method = Mock(return_value=[symbol_data])
    
    doc_id = TextDocumentIdentifier(uri="file:///test.py")
    result = client.lsp_document_symbols(doc_id)
    
    assert len(result) == 1
    assert isinstance(result[0], DocumentSymbol)
    assert result[0].name == "test_function"


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_lsp_document_symbols_symbol_information_response(mock_popen, mock_process):
    """Test document symbols with SymbolInformation response."""
    mock_popen.return_value = mock_process
    
    symbol_data = {
        "name": "test_function",
        "kind": 12,
        "location": {
            "uri": "file:///test.py",
            "range": {
                "start": {"line": 1, "character": 0},
                "end": {"line": 5, "character": 0}
            }
        }
    }
    
    client = LspClient()
    client.call_method = Mock(return_value=[symbol_data])
    
    doc_id = TextDocumentIdentifier(uri="file:///test.py")
    result = client.lsp_document_symbols(doc_id)
    
    assert len(result) == 1
    assert isinstance(result[0], SymbolInformation)
    assert result[0].name == "test_function"


# Test type definition
@patch('aider_lite.lsp.client.subprocess.Popen')
def test_lsp_type_definition(mock_popen, mock_process):
    """Test type definition request."""
    mock_popen.return_value = mock_process
    
    location_data = {
        "uri": "file:///test.py",
        "range": {
            "start": {"line": 1, "character": 0},
            "end": {"line": 1, "character": 10}
        }
    }
    
    client = LspClient()
    client.call_method = Mock(return_value=[location_data])
    
    doc_id = TextDocumentIdentifier(uri="file:///test.py")
    pos = Position(line=5, character=10)
    result = client.lsp_type_definition(doc_id, pos)
    
    assert len(result) == 1
    assert isinstance(result[0], Location)
    assert result[0].uri == "file:///test.py"


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_lsp_type_definition_single_result(mock_popen, mock_process):
    """Test type definition with single result."""
    mock_popen.return_value = mock_process
    
    location_data = {
        "uri": "file:///test.py",
        "range": {
            "start": {"line": 1, "character": 0},
            "end": {"line": 1, "character": 10}
        }
    }
    
    client = LspClient()
    client.call_method = Mock(return_value=location_data)
    
    doc_id = TextDocumentIdentifier(uri="file:///test.py")
    pos = Position(line=5, character=10)
    result = client.lsp_type_definition(doc_id, pos)
    
    assert len(result) == 1
    assert isinstance(result[0], Location)


# Test shutdown functionality
@patch('aider_lite.lsp.client.subprocess.Popen')
def test_is_running(mock_popen, mock_process):
    """Test is_running method."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    assert client.is_running() is True
    
    client.shutdown_flag = True
    assert client.is_running() is False


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_shutdown_graceful(mock_popen, mock_process):
    """Test graceful shutdown."""
    mock_popen.return_value = mock_process
    mock_process.wait.return_value = 0
    
    client = LspClient()
    client.call_method = Mock()
    client.send_notification = Mock()
    
    client.shutdown()
    
    assert client.shutdown_flag is True
    client.call_method.assert_called_with("shutdown")
    client.send_notification.assert_called_with("exit")


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_shutdown_already_shutdown(mock_popen, mock_process):
    """Test shutdown when already shutdown."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    client.shutdown_flag = True
    
    client.shutdown()
    # Should not raise an exception


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_shutdown_process_already_dead(mock_popen, mock_process):
    """Test shutdown when process is already dead."""
    mock_popen.return_value = mock_process
    mock_process.poll.return_value = 0  # Process is dead
    
    client = LspClient()
    client.shutdown()
    
    assert client.shutdown_flag is True


# Test diagnostics
@patch('aider_lite.lsp.client.subprocess.Popen')
def test_handle_diagnostic_notification(mock_popen, mock_process):
    """Test handling diagnostic notification."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    
    params = {
        "uri": "file:///test.py",
        "diagnostics": [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 5}
                },
                "message": "Test error",
                "severity": 1
            }
        ]
    }
    
    client._handle_diagnostic_notification(params)
    
    assert "file:///test.py" in client.diagnostic_results
    assert client.diagnostic_results["file:///test.py"] == params


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_last_lsp_document_diagnostics(mock_popen, mock_process):
    """Test getting last diagnostics for document."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    
    params = {"uri": "file:///test.py", "diagnostics": []}
    client.diagnostic_results["file:///test.py"] = params
    
    result = client.last_lsp_document_diagnostics("file:///test.py")
    assert result == params


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_lsp_diagnostics_wait_timeout(mock_popen, mock_process, mock_condition):
    """Test waiting for diagnostics with timeout."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    client.send_notification = Mock()
    
    doc = TextDocumentItem(
        uri="file:///test.py",
        languageId=LanguageIdentifier.PYTHON,
        version=1,
        text="print('hello')"
    )
    
    mock_condition.wait.return_value = False  # Timeout
    
    with patch('threading.Condition', return_value=mock_condition):
        result = client.lsp_diagnostics_wait(doc, timeout=0.1)
    
    assert result is None
    client.send_notification.assert_called_once()


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_lsp_diagnostics_wait_success(mock_popen, mock_process, mock_condition):
    """Test waiting for diagnostics successfully."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    client.send_notification = Mock()
    
    doc = TextDocumentItem(
        uri="file:///test.py",
        languageId=LanguageIdentifier.PYTHON,
        version=1,
        text="print('hello')"
    )
    
    expected_result = {"uri": "file:///test.py", "diagnostics": []}
    mock_condition.wait.return_value = True  # Success
    
    with patch('threading.Condition', return_value=mock_condition), \
         patch.object(client, 'diagnostic_results', {doc.uri: expected_result}):
        
        result = client.lsp_diagnostics_wait(doc, timeout=1.0)
    
    assert result == expected_result


# Test error handling
@patch('aider_lite.lsp.client.subprocess.Popen')
def test_check_process_alive_running(mock_popen, mock_process):
    """Test checking if process is alive when running."""
    mock_popen.return_value = mock_process
    mock_process.poll.return_value = None  # Still running
    
    client = LspClient()
    client._check_process_alive()  # Should not raise


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_check_process_alive_dead(mock_popen, mock_process):
    """Test checking if process is alive when dead."""
    mock_popen.return_value = mock_process
    mock_process.poll.return_value = 1  # Process died
    
    client = LspClient()
    
    with pytest.raises(ConnectionError, match="LSP server process has died"):
        client._check_process_alive()


@patch('aider_lite.lsp.client.subprocess.Popen')
def test_operations_after_shutdown(mock_popen, mock_process):
    """Test that operations fail after shutdown."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    client.shutdown_flag = True
    
    with pytest.raises(ConnectionError, match="Cannot call methods after shutdown"):
        client.call_method("test")


# Integration-style tests (but still mocked)
@patch('aider_lite.lsp.client.subprocess.Popen')
def test_full_workflow_simulation(mock_popen, mock_process):
    """Test a full LSP workflow simulation."""
    mock_popen.return_value = mock_process
    
    client = LspClient()
    client.call_method = Mock()
    client.send_notification = Mock()
    
    # Simulate initialize
    client.call_method.return_value = {"capabilities": {}}
    init_result = client.lsp_initialize("/test")
    assert "capabilities" in init_result
    
    # Simulate didOpen
    doc = TextDocumentItem(
        uri="file:///test.py",
        languageId=LanguageIdentifier.PYTHON,
        version=1,
        text="def hello(): pass"
    )
    client.lsp_did_open(doc)
    
    # Simulate document symbols
    client.call_method.return_value = []
    symbols = client.lsp_document_symbols(TextDocumentIdentifier(uri=doc.uri))
    assert symbols == []
    
    # Verify calls were made
    assert client.call_method.call_count >= 2
    assert client.send_notification.call_count >= 1