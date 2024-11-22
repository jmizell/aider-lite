from dataclasses import dataclass, asdict
from enum import Enum, IntEnum
from typing import Any, Optional, List

DEFAULT_CAPABILITIES = {
    'textDocument': {
        'completion': {
            'completionItem': {
                'commitCharactersSupport': True,
                'documentationFormat': ['markdown', 'plaintext'],
                'snippetSupport': True
            }
        }
    }
}


class LanguageServerErrorCodes(IntEnum):
    """
    Enumerates error codes defined by JSON-RPC and the LSP protocol.

    Attributes:
        ParseError: (-32700) The message sent to the server could not be parsed.
        InvalidRequest: (-32600) The request was not valid JSON-RPC.
        MethodNotFound: (-32601) The requested method is not supported by the server.
        InvalidParams: (-32602) Invalid method parameters were provided.
        InternalError: (-32603) An internal error occurred within the server.
        serverErrorStart: (-32099) Start range for JSON-RPC server error codes.
        serverErrorEnd: (-32000) End range for JSON-RPC server error codes.
        ServerNotInitialized: (-32002) A request or notification was sent before the server was initialized.
        UnknownErrorCode: (-32001) An unknown error occurred.

        RequestFailed: (-32803) The request failed for semantically invalid reasons.
        ServerCancelled: (-32802) The server cancelled the request.
        RequestCancelled: (-32800) The request was cancelled by the client.
        ContentModified: (-32801) The document content was modified unexpectedly.
        lspReservedErrorRangeStart: (-32899) Start range for LSP-specific error codes.
        lspReservedErrorRangeEnd: (-32800) End range for LSP-specific error codes.
    """
    # JSON-RPC errors
    ParseError = -32700
    InvalidRequest = -32600
    MethodNotFound = -32601
    InvalidParams = -32602
    InternalError = -32603

    # Reserved range for JSON-RPC server errors
    serverErrorStart = -32099
    serverErrorEnd = -32000
    ServerNotInitialized = -32002
    UnknownErrorCode = -32001

    # LSP-specific errors
    RequestFailed = -32803
    ServerCancelled = -32802
    RequestCancelled = -32800
    ContentModified = -32801

    # Reserved range for LSP-specific error codes
    lspReservedErrorRangeStart = -32899
    lspReservedErrorRangeEnd = -32800


class LanguageServerResponseError(Exception):
    """
    Represents an error in the response from the language server.

    Attributes:
        code: An error code from the LanguageServerErrorCodes enumeration.
        message: A human-readable error message describing the issue.
        data: Additional information about the error (optional).
    """
    def __init__(self, code: LanguageServerErrorCodes, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data  # Always set data, even if None


class LanguageIdentifier(str, Enum):
    """
    Enumerates language identifiers for text documents.
    """
    BAT = "bat"
    BIBTEX = "bibtex"
    CLOJURE = "clojure"
    COFFEESCRIPT = "coffeescript"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    CSS = "css"
    DIFF = "diff"
    DOCKERFILE = "dockerfile"
    FSHARP = "fsharp"
    GIT_COMMIT = "git-commit"
    GIT_REBASE = "git-rebase"
    GO = "go"
    GROOVY = "groovy"
    HANDLEBARS = "handlebars"
    HTML = "html"
    INI = "ini"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    JSON = "json"
    LATEX = "latex"
    LESS = "less"
    LUA = "lua"
    MAKEFILE = "makefile"
    MARKDOWN = "markdown"
    OBJECTIVE_C = "objective-c"
    OBJECTIVE_CPP = "objective-cpp"
    Perl = "perl"
    PHP = "php"
    POWERSHELL = "powershell"
    PUG = "jade"
    PYTHON = "python"
    R = "r"
    RAZOR = "razor"
    RUBY = "ruby"
    RUST = "rust"
    SASS = "sass"
    SCSS = "scss"
    ShaderLab = "shaderlab"
    SHELL_SCRIPT = "shellscript"
    SQL = "sql"
    SWIFT = "swift"
    TYPE_SCRIPT = "typescript"
    TEX = "tex"
    VB = "vb"
    XML = "xml"
    XSL = "xsl"
    YAML = "yaml"


@dataclass
class DataclassInstance:
    """
    A base class providing a method to convert data classes to dictionaries.
    """
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TextDocumentItem(DataclassInstance):
    """
    Represents a text document.

    Attributes:
        uri: The document's URI as a string.
        languageId: The programming language of the document.
        version: The version of the document, incremented after changes.
        text: The content of the document.
    """
    uri: str
    languageId: LanguageIdentifier
    version: int
    text: str


@dataclass
class TextDocumentIdentifier(DataclassInstance):
    """
    Identifies a text document uniquely using its URI.

    Attributes:
        uri: The document's URI as a string.
    """
    uri: str


@dataclass
class Position(DataclassInstance):
    """
    Represents a specific position in a text document.

    Attributes:
        line: Zero-based line index in the document.
        character: Zero-based character index in the line.
    """
    line: int
    character: int


@dataclass
class Range(DataclassInstance):
    """
    Defines a range of text within a document.

    Attributes:
        start: The starting position of the range.
        end: The ending position of the range.
    """
    start: Position
    end: Position


@dataclass
class Location(DataclassInstance):
    """
    Represents a specific location in a resource.

    Attributes:
        uri: The resource's URI as a string.
        range: The range within the resource.
    """
    uri: str
    range: Range

    @classmethod
    def from_dict(cls, data: dict):
        if 'range' in data:
            data['range'] = Range(**data['range'])
        return cls(**data)


class SymbolKind(IntEnum):
    """
    Enumerates symbol kinds, such as classes, methods, and variables.
    """
    File = 1
    Module = 2
    Namespace = 3
    Package = 4
    Class = 5
    Method = 6
    Property = 7
    Field = 8
    Constructor = 9
    Enum = 10
    Interface = 11
    Function = 12
    Variable = 13
    Constant = 14
    String = 15
    Number = 16
    Boolean = 17
    Array = 18
    Object = 19
    Key = 20
    Null = 21
    EnumMember = 22
    Struct = 23
    Event = 24
    Operator = 25
    TypeParameter = 26


class SymbolTag(IntEnum):
    """
    Represents additional attributes for a symbol, such as deprecation.
    """
    Deprecated = 1


@dataclass
class DocumentSymbol(DataclassInstance):
    """
    Represents programming constructs like variables, classes, and methods.

    Attributes:
        name: The name of the symbol.
        detail: Additional details about the symbol.
        kind: The type of symbol, specified by SymbolKind.
        tags: Additional attributes of the symbol (e.g., deprecation).
        deprecated: Whether the symbol is deprecated.
        range: The range covering the symbol's definition.
        selectionRange: The range used for the symbol's selection.
        children: Optional child symbols.
    """
    name: str
    detail: Optional[str] = None
    kind: Optional[SymbolKind] = None
    tags: Optional[List[SymbolTag]] = None
    deprecated: Optional[bool] = None
    range: Optional[Range] = None
    selectionRange: Optional[Range] = None
    children: Optional[List['DocumentSymbol']] = None

    @classmethod
    def from_dict(cls, data: dict):
        if 'range' in data and data['range']:
            data['range'] = Range(**data['range'])
        if 'selectionRange' in data and data['selectionRange']:
            data['selectionRange'] = Range(**data['selectionRange'])
        if 'children' in data and data['children']:
            data['children'] = [cls.from_dict(child) for child in data['children']]
        return cls(**data)


@dataclass
class SymbolInformation(DataclassInstance):
    """
    Represents metadata about programming constructs.

    Attributes:
        name: The symbol's name.
        kind: The type of symbol, specified by SymbolKind.
        deprecated: Whether the symbol is deprecated.
        location: The location of the symbol in a resource.
        containerName: The name of the containing construct.
    """
    name: str
    kind: SymbolKind
    deprecated: Optional[bool] = None
    location: Optional[Location] = None
    containerName: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict):
        if 'location' in data:
            loc_data = data['location']
            if 'range' in loc_data and loc_data['range']:
                loc_data['range'] = Range(**loc_data['range'])
            data['location'] = Location(**loc_data)
        return cls(**data)
