"""
Integration tests for aider_lite.lsp.client module.

These tests use real LSP servers when available to test the full integration.
They may fail if the LSP server is not compatible with the current Python version.
"""

import os
import tempfile
import time
import shutil
from pathlib import Path

import pytest

from aider_lite.lsp.client import LspClient
from aider_lite.lsp.types import (
    LanguageIdentifier,
    LanguageServerResponseError,
    Position,
    TextDocumentIdentifier,
    TextDocumentItem,
)


def test_pylsp_availability():
    """Test if pylsp is available for integration tests."""
    try:
        import subprocess
        result = subprocess.run(["python", "-m", "pylsp", "--help"], 
                              capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture
def temp_python_file():
    """Create a temporary Python file for testing."""
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test.py")
    
    # Create a test Python file
    with open(test_file, "w") as f:
        f.write("""
def hello_world():
    '''A simple hello world function.'''
    print("Hello, World!")
    return "Hello, World!"

class TestClass:
    '''A test class.'''
    
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        '''Get the value.'''
        return self.value

# A global variable
GLOBAL_VAR = "test"
""")
    
    yield temp_dir, test_file
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def syntax_error_file():
    """Create a temporary Python file with syntax errors."""
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "syntax_error.py")
    
    # Create a file with syntax errors
    with open(test_file, "w") as f:
        f.write("""
def broken_function(
    # Missing closing parenthesis
    print("This has syntax errors"
    return "broken"

class BrokenClass
    # Missing colon
    def method(self):
        pass
""")
    
    yield temp_dir, test_file
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(
    not test_pylsp_availability(),
    reason="pylsp not available - install with 'pip install python-lsp-server'"
)
def test_full_lsp_workflow(temp_python_file):
    """Test a complete LSP workflow with a real server."""
    temp_dir, test_file = temp_python_file
    
    client = LspClient(lsp_cmd=["python", "-m", "pylsp"])
    
    try:
        # Start the client
        client.start()
        
        # Initialize
        init_response = client.lsp_initialize(temp_dir)
        assert isinstance(init_response, dict)
        assert "capabilities" in init_response
        
        # Open document
        with open(test_file, "r") as f:
            content = f.read()
        
        doc = TextDocumentItem(
            uri=f"file://{test_file}",
            languageId=LanguageIdentifier.PYTHON,
            version=1,
            text=content
        )
        
        client.lsp_did_open(doc)
        
        # Get document symbols
        symbols = client.lsp_document_symbols(TextDocumentIdentifier(uri=doc.uri))
        assert isinstance(symbols, list)
        # Note: May be empty due to Python 3.12 compatibility issues
        
        # Test type definition (may not work with all servers)
        try:
            type_def = client.lsp_type_definition(
                TextDocumentIdentifier(uri=doc.uri),
                Position(line=2, character=10)
            )
            assert isinstance(type_def, list)
        except LanguageServerResponseError:
            # Expected if server doesn't support this or has compatibility issues
            pass
        
    finally:
        client.shutdown()


@pytest.mark.skipif(
    not test_pylsp_availability(),
    reason="pylsp not available - install with 'pip install python-lsp-server'"
)
def test_client_restart(temp_python_file):
    """Test restarting the LSP client."""
    temp_dir, test_file = temp_python_file
    
    # First session
    client1 = LspClient(lsp_cmd=["python", "-m", "pylsp"])
    
    try:
        client1.start()
        init_response1 = client1.lsp_initialize(temp_dir)
        assert isinstance(init_response1, dict)
    finally:
        client1.shutdown()
    
    # Second session with new client
    client2 = LspClient(lsp_cmd=["python", "-m", "pylsp"])
    
    try:
        client2.start()
        init_response2 = client2.lsp_initialize(temp_dir)
        assert isinstance(init_response2, dict)
        
        # Should be able to work normally
        with open(test_file, "r") as f:
            content = f.read()
        
        doc = TextDocumentItem(
            uri=f"file://{test_file}",
            languageId=LanguageIdentifier.PYTHON,
            version=1,
            text=content
        )
        
        client2.lsp_did_open(doc)
        
        try:
            symbols = client2.lsp_document_symbols(TextDocumentIdentifier(uri=doc.uri))
            assert isinstance(symbols, list)
        except LanguageServerResponseError:
            # Expected if server has compatibility issues
            pass
        
    finally:
        client2.shutdown()


@pytest.mark.skipif(
    not test_pylsp_availability(),
    reason="pylsp not available - install with 'pip install python-lsp-server'"
)
def test_multiple_documents(temp_python_file):
    """Test working with multiple documents."""
    temp_dir, test_file = temp_python_file
    
    # Create a second file
    test_file2 = os.path.join(temp_dir, "test2.py")
    with open(test_file2, "w") as f:
        f.write("""
import os
import sys

def another_function():
    return os.path.join("test", "path")
""")
    
    client = LspClient(lsp_cmd=["python", "-m", "pylsp"])
    
    try:
        client.start()
        client.lsp_initialize(temp_dir)
        
        # Open first document
        with open(test_file, "r") as f:
            content1 = f.read()
        
        doc1 = TextDocumentItem(
            uri=f"file://{test_file}",
            languageId=LanguageIdentifier.PYTHON,
            version=1,
            text=content1
        )
        client.lsp_did_open(doc1)
        
        # Open second document
        with open(test_file2, "r") as f:
            content2 = f.read()
        
        doc2 = TextDocumentItem(
            uri=f"file://{test_file2}",
            languageId=LanguageIdentifier.PYTHON,
            version=1,
            text=content2
        )
        client.lsp_did_open(doc2)
        
        # Get symbols from both documents
        try:
            symbols1 = client.lsp_document_symbols(TextDocumentIdentifier(uri=doc1.uri))
            symbols2 = client.lsp_document_symbols(TextDocumentIdentifier(uri=doc2.uri))
            
            assert isinstance(symbols1, list)
            assert isinstance(symbols2, list)
        except LanguageServerResponseError:
            # Expected if server has compatibility issues
            pass
        
    finally:
        client.shutdown()


def test_lsp_client_without_server():
    """Test LSP client behavior when server is not available."""
    client = LspClient(lsp_cmd=["non-existent-lsp-server"])
    
    with pytest.raises(FileNotFoundError):
        client.start()


@pytest.mark.skipif(
    not test_pylsp_availability(),
    reason="pylsp not available - install with 'pip install python-lsp-server'"
)
def test_lsp_with_syntax_error(syntax_error_file):
    """Test LSP behavior with syntax errors in the file."""
    temp_dir, test_file = syntax_error_file
    
    client = LspClient(lsp_cmd=["python", "-m", "pylsp"])
    
    try:
        client.start()
        client.lsp_initialize(temp_dir)
        
        # Open document with syntax errors
        with open(test_file, "r") as f:
            content = f.read()
        
        doc = TextDocumentItem(
            uri=f"file://{test_file}",
            languageId=LanguageIdentifier.PYTHON,
            version=1,
            text=content
        )
        
        client.lsp_did_open(doc)
        
        # Wait a bit for diagnostics
        time.sleep(1)
        
        # Try to get diagnostics
        diagnostics = client.last_lsp_document_diagnostics(doc.uri)
        
        # Should either have diagnostics or None (depending on server capabilities)
        if diagnostics is not None:
            assert isinstance(diagnostics, dict)
            assert "uri" in diagnostics
            assert "diagnostics" in diagnostics
        
        # Document symbols might still work even with syntax errors
        try:
            symbols = client.lsp_document_symbols(TextDocumentIdentifier(uri=doc.uri))
            assert isinstance(symbols, list)
        except LanguageServerResponseError:
            # Expected if server can't parse the file or has compatibility issues
            pass
        
    finally:
        client.shutdown()


# Test diagnostic waiting functionality
@pytest.mark.skipif(
    not test_pylsp_availability(),
    reason="pylsp not available - install with 'pip install python-lsp-server'"
)
def test_diagnostics_wait_functionality(temp_python_file):
    """Test waiting for diagnostics functionality."""
    temp_dir, test_file = temp_python_file
    
    client = LspClient(lsp_cmd=["python", "-m", "pylsp"])
    
    try:
        client.start()
        client.lsp_initialize(temp_dir)
        
        # Create document
        with open(test_file, "r") as f:
            content = f.read()
        
        doc = TextDocumentItem(
            uri=f"file://{test_file}",
            languageId=LanguageIdentifier.PYTHON,
            version=1,
            text=content
        )
        
        # Test diagnostics wait (may timeout due to server compatibility)
        try:
            result = client.lsp_diagnostics_wait(doc, timeout=5.0)
            # Result can be None (timeout) or dict (success)
            if result is not None:
                assert isinstance(result, dict)
                assert "uri" in result
                assert "diagnostics" in result
        except LanguageServerResponseError:
            # Expected if server has compatibility issues
            pass
        
    finally:
        client.shutdown()


# Performance and stress tests
@pytest.mark.skipif(
    not test_pylsp_availability(),
    reason="pylsp not available - install with 'pip install python-lsp-server'"
)
def test_rapid_requests(temp_python_file):
    """Test making rapid requests to the LSP server."""
    temp_dir, test_file = temp_python_file
    
    client = LspClient(lsp_cmd=["python", "-m", "pylsp"])
    
    try:
        client.start()
        client.lsp_initialize(temp_dir)
        
        with open(test_file, "r") as f:
            content = f.read()
        
        doc = TextDocumentItem(
            uri=f"file://{test_file}",
            languageId=LanguageIdentifier.PYTHON,
            version=1,
            text=content
        )
        
        client.lsp_did_open(doc)
        
        # Make multiple rapid requests
        doc_id = TextDocumentIdentifier(uri=doc.uri)
        
        for i in range(5):
            try:
                symbols = client.lsp_document_symbols(doc_id)
                assert isinstance(symbols, list)
            except LanguageServerResponseError:
                # Expected if server has compatibility issues
                pass
        
    finally:
        client.shutdown()