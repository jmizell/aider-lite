"""
Tests for aider_lite.lsp.types module.

This module tests all the LSP type definitions, enums, and data structures
to ensure they work correctly for JSON-RPC communication.
"""

import pytest
from aider_lite.lsp.types import *


# Test LSP error code constants
def test_json_rpc_error_codes():
    """Test JSON-RPC standard error codes."""
    assert LanguageServerErrorCodes.ParseError == -32700
    assert LanguageServerErrorCodes.InvalidRequest == -32600
    assert LanguageServerErrorCodes.MethodNotFound == -32601
    assert LanguageServerErrorCodes.InvalidParams == -32602
    assert LanguageServerErrorCodes.InternalError == -32603


def test_server_error_range():
    """Test server error range."""
    assert LanguageServerErrorCodes.serverErrorStart == -32099
    assert LanguageServerErrorCodes.serverErrorEnd == -32000


def test_lsp_error_range():
    """Test LSP reserved error range."""
    assert LanguageServerErrorCodes.lspReservedErrorRangeStart == -32899
    # Note: serverErrorEnd is used as the end of LSP range
    assert LanguageServerErrorCodes.serverErrorEnd == -32000


def test_lsp_specific_errors():
    """Test LSP specific error codes."""
    assert LanguageServerErrorCodes.RequestFailed == -32803
    assert LanguageServerErrorCodes.ServerCancelled == -32802
    assert LanguageServerErrorCodes.ContentModified == -32801
    assert LanguageServerErrorCodes.RequestCancelled == -32800


# Test LanguageServerResponseError exception
def test_basic_error_creation():
    """Test basic error creation."""
    error = LanguageServerResponseError(-32601, "Method not found")
    assert error.code == -32601
    assert error.message == "Method not found"
    assert error.data is None


def test_error_with_data():
    """Test error creation with data."""
    data = {"traceback": ["line1", "line2"]}
    error = LanguageServerResponseError(-32603, "Internal error", data)
    assert error.code == -32603
    assert error.message == "Internal error"
    assert error.data == data


def test_error_inheritance():
    """Test that error inherits from Exception."""
    error = LanguageServerResponseError(-32601, "Method not found")
    assert isinstance(error, Exception)


# Test LanguageIdentifier enum
def test_common_languages():
    """Test common programming languages."""
    assert LanguageIdentifier.PYTHON == "python"
    assert LanguageIdentifier.JAVASCRIPT == "javascript"
    assert LanguageIdentifier.TYPE_SCRIPT == "typescript"
    assert LanguageIdentifier.JAVA == "java"
    assert LanguageIdentifier.CPP == "cpp"
    assert LanguageIdentifier.C == "c"
    assert LanguageIdentifier.CSHARP == "csharp"
    assert LanguageIdentifier.GO == "go"
    assert LanguageIdentifier.RUST == "rust"
    assert LanguageIdentifier.PHP == "php"
    assert LanguageIdentifier.RUBY == "ruby"


def test_markup_languages():
    """Test markup and config languages."""
    assert LanguageIdentifier.HTML == "html"
    assert LanguageIdentifier.CSS == "css"
    assert LanguageIdentifier.JSON == "json"
    assert LanguageIdentifier.XML == "xml"
    assert LanguageIdentifier.YAML == "yaml"
    assert LanguageIdentifier.MARKDOWN == "markdown"


def test_string_behavior():
    """Test that enum values behave like strings."""
    lang = LanguageIdentifier.PYTHON
    assert lang == "python"  # Direct comparison works
    # Note: str() returns the enum representation, not the value


# Test DataclassInstance base functionality
def test_to_dict_method():
    """Test that to_dict method works on dataclass instances."""
    pos = Position(line=10, character=5)
    result = pos.to_dict()
    expected = {"line": 10, "character": 5}
    assert result == expected


# Test Position dataclass
def test_position_creation():
    """Test Position creation."""
    pos = Position(line=10, character=5)
    assert pos.line == 10
    assert pos.character == 5


def test_position_zero_values():
    """Test Position with zero values."""
    pos = Position(line=0, character=0)
    assert pos.line == 0
    assert pos.character == 0


def test_position_to_dict():
    """Test Position to_dict method."""
    pos = Position(line=10, character=5)
    result = pos.to_dict()
    expected = {"line": 10, "character": 5}
    assert result == expected


# Test Range dataclass
def test_range_creation():
    """Test Range creation."""
    start = Position(line=1, character=0)
    end = Position(line=1, character=10)
    range_obj = Range(start=start, end=end)
    assert range_obj.start == start
    assert range_obj.end == end


def test_range_to_dict():
    """Test Range to_dict method."""
    start = Position(line=1, character=0)
    end = Position(line=1, character=10)
    range_obj = Range(start=start, end=end)
    result = range_obj.to_dict()
    expected = {
        "start": {"line": 1, "character": 0},
        "end": {"line": 1, "character": 10}
    }
    assert result == expected


def test_range_from_dict():
    """Test Range from_dict method."""
    data = {
        "start": {"line": 1, "character": 0},
        "end": {"line": 1, "character": 10}
    }
    range_obj = Range.from_dict(data)
    assert range_obj.start.line == 1
    assert range_obj.start.character == 0
    assert range_obj.end.line == 1
    assert range_obj.end.character == 10


# Test Location dataclass
def test_location_creation():
    """Test Location creation."""
    range_obj = Range(
        start=Position(line=1, character=0),
        end=Position(line=1, character=10)
    )
    location = Location(uri="file:///test.py", range=range_obj)
    assert location.uri == "file:///test.py"
    assert location.range == range_obj


def test_location_from_dict():
    """Test Location from_dict method."""
    data = {
        "uri": "file:///test.py",
        "range": {
            "start": {"line": 1, "character": 0},
            "end": {"line": 1, "character": 10}
        }
    }
    location = Location.from_dict(data)
    assert location.uri == "file:///test.py"
    assert location.range.start.line == 1
    assert location.range.start.character == 0
    assert location.range.end.line == 1
    assert location.range.end.character == 10


def test_location_from_dict_without_range():
    """Test Location from_dict method without range."""
    data = {"uri": "file:///test.py"}
    location = Location.from_dict(data)
    assert location.uri == "file:///test.py"
    assert location.range is None


# Test SymbolKind enum
def test_symbol_kind_values():
    """Test SymbolKind enum values."""
    assert SymbolKind.File == 1
    assert SymbolKind.Module == 2
    assert SymbolKind.Namespace == 3
    assert SymbolKind.Package == 4
    assert SymbolKind.Class == 5
    assert SymbolKind.Method == 6
    assert SymbolKind.Property == 7
    assert SymbolKind.Field == 8
    assert SymbolKind.Constructor == 9
    assert SymbolKind.Enum == 10
    assert SymbolKind.Interface == 11
    assert SymbolKind.Function == 12
    assert SymbolKind.Variable == 13
    assert SymbolKind.Constant == 14
    assert SymbolKind.String == 15
    assert SymbolKind.Number == 16
    assert SymbolKind.Boolean == 17
    assert SymbolKind.Array == 18
    assert SymbolKind.Object == 19
    assert SymbolKind.Key == 20
    assert SymbolKind.Null == 21
    assert SymbolKind.EnumMember == 22
    assert SymbolKind.Struct == 23
    assert SymbolKind.Event == 24
    assert SymbolKind.Operator == 25
    assert SymbolKind.TypeParameter == 26


# Test SymbolTag enum
def test_symbol_tag_values():
    """Test SymbolTag enum values."""
    assert SymbolTag.Deprecated == 1


# Test TextDocumentItem dataclass
def test_text_document_item_creation():
    """Test TextDocumentItem creation."""
    doc = TextDocumentItem(
        uri="file:///test.py",
        languageId=LanguageIdentifier.PYTHON,
        version=1,
        text="print('hello')"
    )
    assert doc.uri == "file:///test.py"
    assert doc.languageId == LanguageIdentifier.PYTHON
    assert doc.version == 1
    assert doc.text == "print('hello')"


def test_text_document_item_to_dict():
    """Test TextDocumentItem to_dict method."""
    doc = TextDocumentItem(
        uri="file:///test.py",
        languageId=LanguageIdentifier.PYTHON,
        version=1,
        text="print('hello')"
    )
    result = doc.to_dict()
    expected = {
        "uri": "file:///test.py",
        "languageId": LanguageIdentifier.PYTHON,
        "version": 1,
        "text": "print('hello')"
    }
    assert result == expected


# Test TextDocumentIdentifier dataclass
def test_text_document_identifier_creation():
    """Test TextDocumentIdentifier creation."""
    doc_id = TextDocumentIdentifier(uri="file:///test.py")
    assert doc_id.uri == "file:///test.py"


def test_text_document_identifier_to_dict():
    """Test TextDocumentIdentifier to_dict method."""
    doc_id = TextDocumentIdentifier(uri="file:///test.py")
    result = doc_id.to_dict()
    expected = {"uri": "file:///test.py"}
    assert result == expected


# Test DocumentSymbol dataclass
def test_document_symbol_minimal():
    """Test DocumentSymbol with minimal required fields."""
    symbol = DocumentSymbol(
        name="test_function",
        kind=SymbolKind.Function,
        range=Range(
            start=Position(line=1, character=0),
            end=Position(line=5, character=0)
        ),
        selectionRange=Range(
            start=Position(line=1, character=4),
            end=Position(line=1, character=17)
        )
    )
    assert symbol.name == "test_function"
    assert symbol.kind == SymbolKind.Function


def test_document_symbol_full():
    """Test DocumentSymbol with all fields."""
    symbol = DocumentSymbol(
        name="test_function",
        detail="def test_function():",
        kind=SymbolKind.Function,
        tags=[SymbolTag.Deprecated],
        deprecated=True,
        range=Range(
            start=Position(line=1, character=0),
            end=Position(line=5, character=0)
        ),
        selectionRange=Range(
            start=Position(line=1, character=4),
            end=Position(line=1, character=17)
        ),
        children=[]
    )
    assert symbol.name == "test_function"
    assert symbol.detail == "def test_function():"
    assert symbol.kind == SymbolKind.Function
    assert symbol.tags == [SymbolTag.Deprecated]
    assert symbol.deprecated is True
    assert symbol.children == []


def test_document_symbol_from_dict_minimal():
    """Test DocumentSymbol from_dict with minimal data."""
    data = {
        "name": "test_function",
        "kind": 12,  # SymbolKind.Function
        "range": {
            "start": {"line": 1, "character": 0},
            "end": {"line": 5, "character": 0}
        },
        "selectionRange": {
            "start": {"line": 1, "character": 4},
            "end": {"line": 1, "character": 17}
        }
    }
    symbol = DocumentSymbol.from_dict(data)
    assert symbol.name == "test_function"
    assert symbol.kind == SymbolKind.Function


def test_document_symbol_from_dict():
    """Test DocumentSymbol from_dict with full data."""
    data = {
        "name": "test_function",
        "detail": "def test_function():",
        "kind": 12,  # SymbolKind.Function
        "tags": [1],  # SymbolTag.Deprecated
        "deprecated": True,
        "range": {
            "start": {"line": 1, "character": 0},
            "end": {"line": 5, "character": 0}
        },
        "selectionRange": {
            "start": {"line": 1, "character": 4},
            "end": {"line": 1, "character": 17}
        },
        "children": []
    }
    symbol = DocumentSymbol.from_dict(data)
    assert symbol.name == "test_function"
    assert symbol.detail == "def test_function():"
    assert symbol.kind == SymbolKind.Function
    assert symbol.tags == [SymbolTag.Deprecated]
    assert symbol.deprecated is True
    assert symbol.children == []


# Test SymbolInformation dataclass
def test_symbol_information_minimal():
    """Test SymbolInformation with minimal required fields."""
    symbol = SymbolInformation(
        name="test_function",
        kind=SymbolKind.Function,
        location=Location(
            uri="file:///test.py",
            range=Range(
                start=Position(line=1, character=0),
                end=Position(line=5, character=0)
            )
        )
    )
    assert symbol.name == "test_function"
    assert symbol.kind == SymbolKind.Function


def test_symbol_information_full():
    """Test SymbolInformation with all fields."""
    symbol = SymbolInformation(
        name="test_function",
        kind=SymbolKind.Function,
        deprecated=True,
        location=Location(
            uri="file:///test.py",
            range=Range(
                start=Position(line=1, character=0),
                end=Position(line=5, character=0)
            )
        ),
        containerName="TestClass"
    )
    assert symbol.name == "test_function"
    assert symbol.kind == SymbolKind.Function
    assert symbol.deprecated is True
    assert symbol.containerName == "TestClass"


def test_symbol_information_from_dict_minimal():
    """Test SymbolInformation from_dict with minimal data."""
    data = {
        "name": "test_function",
        "kind": 12,  # SymbolKind.Function
        "location": {
            "uri": "file:///test.py",
            "range": {
                "start": {"line": 1, "character": 0},
                "end": {"line": 5, "character": 0}
            }
        }
    }
    symbol = SymbolInformation.from_dict(data)
    assert symbol.name == "test_function"
    assert symbol.kind == SymbolKind.Function


def test_symbol_information_from_dict():
    """Test SymbolInformation from_dict with full data."""
    data = {
        "name": "test_function",
        "kind": 12,  # SymbolKind.Function
        "deprecated": True,
        "location": {
            "uri": "file:///test.py",
            "range": {
                "start": {"line": 1, "character": 0},
                "end": {"line": 5, "character": 0}
            }
        },
        "containerName": "TestClass"
    }
    symbol = SymbolInformation.from_dict(data)
    assert symbol.name == "test_function"
    assert symbol.kind == SymbolKind.Function
    assert symbol.deprecated is True
    assert symbol.containerName == "TestClass"


# Test default capabilities structure
def test_default_capabilities_structure():
    """Test that default capabilities have expected structure."""
    caps = DEFAULT_CAPABILITIES
    assert "textDocument" in caps
    assert "completion" in caps["textDocument"]
    assert "completionItem" in caps["textDocument"]["completion"]
    assert caps["textDocument"]["completion"]["completionItem"]["snippetSupport"] is True


# Parametrized tests for various symbol kinds
@pytest.mark.parametrize("symbol_kind,expected_value", [
    (SymbolKind.File, 1),
    (SymbolKind.Module, 2),
    (SymbolKind.Namespace, 3),
    (SymbolKind.Package, 4),
    (SymbolKind.Class, 5),
    (SymbolKind.Method, 6),
    (SymbolKind.Property, 7),
    (SymbolKind.Field, 8),
    (SymbolKind.Constructor, 9),
    (SymbolKind.Enum, 10),
    (SymbolKind.Interface, 11),
    (SymbolKind.Function, 12),
    (SymbolKind.Variable, 13),
    (SymbolKind.Constant, 14),
    (SymbolKind.String, 15),
    (SymbolKind.Number, 16),
    (SymbolKind.Boolean, 17),
    (SymbolKind.Array, 18),
    (SymbolKind.Object, 19),
    (SymbolKind.Key, 20),
    (SymbolKind.Null, 21),
    (SymbolKind.EnumMember, 22),
    (SymbolKind.Struct, 23),
    (SymbolKind.Event, 24),
    (SymbolKind.Operator, 25),
    (SymbolKind.TypeParameter, 26),
])
def test_symbol_kind_parametrized(symbol_kind, expected_value):
    """Test SymbolKind enum values using parametrized test."""
    assert symbol_kind == expected_value


# Parametrized tests for various language identifiers
@pytest.mark.parametrize("language,expected_value", [
    (LanguageIdentifier.PYTHON, "python"),
    (LanguageIdentifier.JAVASCRIPT, "javascript"),
    (LanguageIdentifier.TYPE_SCRIPT, "typescript"),
    (LanguageIdentifier.JAVA, "java"),
    (LanguageIdentifier.CPP, "cpp"),
    (LanguageIdentifier.C, "c"),
    (LanguageIdentifier.CSHARP, "csharp"),
    (LanguageIdentifier.GO, "go"),
    (LanguageIdentifier.RUST, "rust"),
    (LanguageIdentifier.PHP, "php"),
    (LanguageIdentifier.RUBY, "ruby"),
    (LanguageIdentifier.HTML, "html"),
    (LanguageIdentifier.CSS, "css"),
    (LanguageIdentifier.JSON, "json"),
    (LanguageIdentifier.XML, "xml"),
    (LanguageIdentifier.YAML, "yaml"),
    (LanguageIdentifier.MARKDOWN, "markdown"),
])
def test_language_identifier_parametrized(language, expected_value):
    """Test LanguageIdentifier enum values using parametrized test."""
    assert language == expected_value