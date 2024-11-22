import logging
import os
import sys
import json
import subprocess
import threading
from enum import auto, Enum
from typing import Union, List
from .types import (
    DEFAULT_CAPABILITIES,
    DocumentSymbol,
    LanguageIdentifier,
    LanguageServerErrorCodes,
    LanguageServerResponseError,
    Location,
    Position,
    SymbolInformation,
    SymbolKind,
    TextDocumentIdentifier,
    TextDocumentItem
)


def to_uri(file_path):
    """Convert a file path to a URI."""
    return f"file://{os.path.abspath(file_path)}"


class StderrHandler(Enum):
    """Enum for handling subprocess stderr output."""
    STDERR = auto()  # Write to sys.stderr
    LOG = auto()     # Write to log file
    DISCARD = auto() # Read and discard


class LspClient(threading.Thread):
    """
    A Language Server Protocol (LSP) client implementation that facilitates communication with LSP servers.

    The LspClient class manages bidirectional communication with a language server using the LSP protocol.
    It handles request-response cycles, notifications, and diagnostic messages asynchronously through
    separate threads for message processing and error handling.

    Features:
    - Asynchronous communication with LSP server
    - Handles requests, responses, and notifications
    - Manages document diagnostics
    - Thread-safe message processing
    - Configurable error handling and logging
    - Timeout management for requests
    - Graceful shutdown handling

    Args:
        lsp_cmd (list[str], optional): Command to start the LSP server. Defaults to ["python", "-m", "pylsp"].
        method_callbacks (dict, optional): Mapping of method names to callback functions for handling server requests.
        notify_callbacks (dict, optional): Mapping of notification names to callback functions for handling server notifications.
        timeout (int, optional): Maximum time in seconds to wait for server responses. Defaults to 10.
        stderr_handler (StderrHandler, optional): How to handle server stderr output. Defaults to StderrHandler.STDERR.
        log_level (int, optional): Logging level for the client. Defaults to logging.INFO.

    Attributes:
        logger (logging.Logger): Logger instance for the client.
        process (subprocess.Popen): Subprocess running the LSP server.
        shutdown_flag (bool): Flag indicating if the client is shutting down.
        latest_diagnostics (dict): Storage for the most recent diagnostics by document URI.

    Example:
        >>> client = LspClient()
        >>> client.start()
        >>> client.lsp_initialize("/path/to/workspace")
        >>> # Open a document
        >>> client.lsp_did_open(TextDocumentItem(uri="file:///path/to/file.py", ...))
        >>> # Get document symbols
        >>> symbols = client.lsp_document_symbols(TextDocumentIdentifier(uri="file:///path/to/file.py"))
        >>> client.shutdown()

    Note:
        The client runs in a separate thread and maintains internal thread-safe state management.
        Always call shutdown() when finished to ensure proper cleanup of resources.
    """

    def __init__(self, lsp_cmd=None, method_callbacks=None, notify_callbacks=None, timeout=10,
                 stderr_handler=StderrHandler.STDERR, log_level=logging.INFO):
        """
        Constructs a new LspClient instance with LSP endpoint functionalities integrated.

        :param lsp_cmd: Command to start pylsp.
        :param method_callbacks: Callbacks for method requests.
        :param notify_callbacks: Callbacks for notifications.
        :param timeout: Timeout for waiting responses.
        """
        threading.Thread.__init__(self)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Only add handlers if they haven't been added already
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            self.logger.addHandler(console_handler)

        self.logger.debug("Initializing LSP client")

        self.lsp_cmd = lsp_cmd or ["python", "-m", "pylsp"]
        self.process = subprocess.Popen(self.lsp_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
        self.stdin = self.process.stdin
        self.stdout = self.process.stdout
        self.stderr = self.process.stderr

        if not isinstance(stderr_handler, StderrHandler):
            raise ValueError(f"stderr_handler must be a StderrHandler enum, got {type(stderr_handler)}")
        self.stderr_handler = stderr_handler

        self.stderr_thread = threading.Thread(target=self._handle_stderr)
        self.stderr_thread.daemon = True
        self.stderr_thread.start()

        self.notify_callbacks = notify_callbacks or {}
        self.notify_callbacks["textDocument/publishDiagnostics"] = lambda params: self._handle_diagnostic_notification(params)
        self.method_callbacks = method_callbacks or {}

        # Shared dictionaries and associated locks
        self.dict_lock = threading.Lock()
        self.diagnostic_events = {}
        self.diagnostic_results = {}
        self.latest_diagnostics = {}
        self.event_dict = {}
        self.response_dict = {}

        self.next_id = 0
        self.next_id_lock = threading.Lock()
        self.timeout = timeout
        self.shutdown_flag = False
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()

    def _check_process_alive(self):
        """Check if the server process is still alive and raise an error if not."""
        if self.process.poll() is not None:
            exit_code = self.process.poll()
            self.shutdown_flag = True  # Set shutdown flag to prevent further operations
            raise ConnectionError(f"Language server process died unexpectedly with exit code: {exit_code}")

    def _read_message(self):
        """
        Read and parse a message from the language server.

        Returns:
            dict: The parsed JSON-RPC message, or None if connection closed

        Raises:
            ConnectionError: If server dies during read
            LanguageServerResponseError: If message parsing fails
        """
        with self.read_lock:
            self._check_process_alive()
            message_size = None
            lines_read = 0
            max_header_lines = 100  # Maximum number of header lines to read

            # Read headers
            while lines_read < max_header_lines:
                try:
                    line = self.stdout.readline()
                    if not line:
                        if self.process.poll() is not None:
                            raise ConnectionError("Language server process died while reading message")
                        return None
                    self.logger.debug(f"Read line: {line}")
                    lines_read += 1

                    try:
                        line = line.decode("utf-8")
                    except UnicodeDecodeError as e:
                        raise LanguageServerResponseError(
                            LanguageServerErrorCodes.ParseError,
                            f"Unicode decode error in header: {e}"
                        )

                    if not line.endswith("\r\n"):
                        raise LanguageServerResponseError(
                            LanguageServerErrorCodes.ParseError,
                            "Bad header: missing newline or partial line"
                        )

                    line = line[:-2]  # Remove \r\n
                    if line == "":  # Empty line marks end of headers
                        break
                    elif line.startswith("Content-Length: "):
                        line = line[len("Content-Length: "):]
                        if not line.isdigit():
                            raise LanguageServerResponseError(
                                LanguageServerErrorCodes.ParseError,
                                "Bad header: size is not int"
                            )
                        message_size = int(line)
                        # 10MB max message size
                        if message_size > 10 * 1024 * 1024:
                            raise LanguageServerResponseError(
                                LanguageServerErrorCodes.ParseError,
                                "Message too large"
                            )
                    elif line.startswith("Content-Type: "):
                        # Just ignore Content-Type header
                        pass
                    else:
                        # Ignore unknown headers instead of raising error
                        pass

                except (IOError, OSError) as e:
                    if self.process.poll() is not None:
                        raise ConnectionError(f"Language server process died while reading headers: {str(e)}")
                    raise

            if lines_read >= max_header_lines:
                raise LanguageServerResponseError(
                    LanguageServerErrorCodes.ParseError,
                    "Too many header lines"
                )

            if not message_size:
                raise LanguageServerResponseError(
                    LanguageServerErrorCodes.ParseError,
                    "Bad header: missing size"
                )

            # Read message body
            try:
                jsonrpc_res = self.stdout.read(message_size)
                if not jsonrpc_res:
                    if self.process.poll() is not None:
                        raise ConnectionError("Language server process died while reading message body")
                    return None

                try:
                    jsonrpc_res = jsonrpc_res.decode("utf-8")
                except UnicodeDecodeError as e:
                    raise LanguageServerResponseError(
                        LanguageServerErrorCodes.ParseError,
                        f"Unicode decode error in body: {e}"
                    )

                self.logger.debug(f"Read message: {jsonrpc_res}")
                try:
                    return json.loads(jsonrpc_res)
                except json.JSONDecodeError as e:
                    raise LanguageServerResponseError(
                        LanguageServerErrorCodes.ParseError,
                        f"JSON decode error: {e}"
                    )

            except (IOError, OSError) as e:
                if self.process.poll() is not None:
                    raise ConnectionError(f"Language server process died while reading message body: {str(e)}")
                raise

    def run(self):
        """
        The main thread execution method for the LSPClient, inherited from threading.Thread.

        This method is automatically called when the thread starts (after .start()).
        It continuously processes incoming messages from the language server by calling
        _process_messages() until shutdown_flag is set.
        """
        self._process_messages()

    def _process_messages(self):
        """
        Main message processing loop that handles incoming messages from the language server.

        This method runs in a separate thread and continuously processes messages until shutdown.
        It handles requests, responses, and notifications from the server while managing error cases
        and unexpected server termination.
        """
        while not self.shutdown_flag:
            rpc_id = None
            try:
                self._check_process_alive()

                try:
                    jsonrpc_message = self._read_message()
                    if jsonrpc_message is None:
                        self.logger.info("Server connection closed")
                        break

                    method = jsonrpc_message.get("method")
                    result = jsonrpc_message.get("result")
                    error = jsonrpc_message.get("error")
                    rpc_id = jsonrpc_message.get("id")
                    params = jsonrpc_message.get("params")

                    if method:  # This is a request or notification from server
                        if rpc_id:  # This is a request
                            if method not in self.method_callbacks:
                                self.logger.warning(f"Method not found: {method}")
                                error_exc = LanguageServerResponseError(
                                    LanguageServerErrorCodes.MethodNotFound,
                                    f"Method not found: {method}"
                                )
                                self._send_response(rpc_id, None, error_exc)
                                continue

                            try:
                                response_result = self.method_callbacks[method](params)
                                self._send_response(rpc_id, response_result, None)
                            except Exception as e:
                                self.logger.exception(f"Error executing method callback {method}")
                                error_exc = LanguageServerResponseError(
                                    LanguageServerErrorCodes.InternalError,
                                    f"Error in method callback: {str(e)}"
                                )
                                self._send_response(rpc_id, None, error_exc)

                        else:  # This is a notification
                            callback = self.notify_callbacks.get(method)
                            if callback is None:
                                self.logger.debug(f"No callback registered for notification: {method}")
                            else:
                                try:
                                    callback(params)
                                except Exception as e:
                                    self.logger.exception(
                                        f"Error in notification callback for method '{method}': {str(e)}"
                                    )

                    elif rpc_id is not None:  # This is a response message
                        with self.dict_lock:
                            self.response_dict[rpc_id] = (result, error)
                            cond = self.event_dict.get(rpc_id)
                            if cond is None:
                                self.logger.warning(f"No pending request found for response id: {rpc_id}")
                                continue
                        with cond:
                            cond.notify()

                    else:
                        self.logger.error("Invalid message: not a request, response or notification")
                        raise LanguageServerResponseError(
                            LanguageServerErrorCodes.InvalidRequest,
                            "Invalid message: not a request, response or notification"
                        )

                except ConnectionError as e:
                    self.logger.error(f"Language server connection terminated unexpectedly: {str(e)}")
                    self.shutdown_flag = True
                    # Notify any waiting threads
                    with self.dict_lock:
                        for event in self.event_dict.values():
                            with event:
                                event.notify()
                        for event in self.diagnostic_events.values():
                            with event:
                                event.notify()
                    break

                except LanguageServerResponseError as e:
                    self.logger.error(f"LSP protocol error: {str(e)}")
                    if rpc_id is not None:
                        self._send_response(rpc_id, None, e)

            except Exception as e:
                self.logger.exception(f"Fatal error in message processing loop: {str(e)}")
                self.shutdown_flag = True
                # Notify waiting threads before breaking
                with self.dict_lock:
                    for event in self.event_dict.values():
                        with event:
                            event.notify()
                    for event in self.diagnostic_events.values():
                        with event:
                            event.notify()
                break

    def _handle_stderr(self):
        """Handle stderr output from the subprocess based on configured handler."""
        for line in iter(self.stderr.readline, b''):
            try:
                line_decoded = line.decode('utf-8', errors='replace').rstrip()
            except UnicodeDecodeError:
                line_decoded = "<binary data?>"
            if self.stderr_handler == StderrHandler.STDERR:
                print(f"LSP subprocess error: {line_decoded}", file=sys.stderr)
            elif self.stderr_handler == StderrHandler.LOG:
                # Use proper logging rather than a separate file
                self.logger.error(f"LSP subprocess error: {line_decoded}")
            # For DISCARD, we just read and drop the output

    def _write_message(self, message):
        if self.shutdown_flag:
            # If shutdown is flagged, do not write messages
            self.logger.debug("Attempted to write message after shutdown, ignoring.")
            return

        self._check_process_alive()

        json_string = json.dumps(message)
        jsonrpc_req = f"Content-Length: {len(json_string)}\r\n\r\n{json_string}"
        with self.write_lock:
            self.logger.debug(f"Writing message: {jsonrpc_req.encode()}")
            try:
                self.stdin.write(jsonrpc_req.encode())
                self.stdin.flush()
            except BrokenPipeError:
                if self.process.poll() is not None:
                    raise ConnectionError("Language server process died - broken pipe")
                raise
            except Exception as e:
                if self.process.poll() is not None:
                    raise ConnectionError(f"Language server process died while writing: {str(e)}")
                raise

    def _send_response(self, request_id, result, error):
        message_dict = {
            "jsonrpc": "2.0",
            "id": request_id
        }
        if error is not None:
            message_dict["error"] = {
                "code": error.code,
                "message": error.message,
                "data": error.data
            }
        else:
            message_dict["result"] = result
        self._write_message(message_dict)

    def _send_request(self, method_name, params, message_id=None):
        if self.shutdown_flag:
            # Prevent sending requests after shutdown
            self.logger.debug("Attempted to send request after shutdown, ignoring.")
            return
        message_dict = {
            "jsonrpc": "2.0",
            "method": method_name,
        }
        if message_id is not None:
            message_dict["id"] = message_id
        if params is not None:
            message_dict["params"] = params
        self._write_message(message_dict)

    def _handle_diagnostic_notification(self, params):
        """Handle incoming diagnostic notifications."""
        uri = params.get("uri")
        with self.dict_lock:
            self.latest_diagnostics[uri] = params  # Store latest diagnostics
            if uri in self.diagnostic_events:
                self.diagnostic_results[uri] = params
                cond = self.diagnostic_events[uri]
        if uri in self.diagnostic_events:
            with cond:
                cond.notify()

    def send_notification(self, method_name, **kwargs):
        """
        Sends a notification to the language server (LSP notification).

        Notifications are messages sent from client to server that do not require a response.
        As per LSP spec, notifications cannot be canceled and do not return results.

        LSP Spec:
        - Notifications have no 'id' field in the JSON-RPC message.
        - Server must not reply to notifications.

        Args:
            method_name (str): The LSP notification method name (e.g. "textDocument/didOpen")
            **kwargs: Arbitrary keyword arguments representing the notification parameters.
        """
        if self.shutdown_flag:
            self.logger.debug(f"Attempted to send notification '{method_name}' after shutdown, ignoring.")
            return
        try:
            self._send_request(method_name, kwargs)
        except Exception as e:
            self.logger.exception(f"Error sending notification '{method_name}': {str(e)}")
            raise

    def call_method(self, method_name, **kwargs):
        """
        Calls a method on the language server and waits for the response (LSP request).

        This method handles the full request-response cycle for LSP method calls:
        1. Assigns a unique request ID
        2. Sends the request to the server
        3. Waits for the response with the matching ID
        4. Returns the result or raises any errors

        LSP Spec:
        - Requests must include an 'id' field to match responses
        - Server must respond with either a result or error
        - Responses must include the same 'id' as the request

        Args:
            method_name (str): The LSP method name to call (e.g. "textDocument/definition")
            **kwargs: Arbitrary keyword arguments representing the method parameters

        Returns:
            The result field from the server's response

        Raises:
            TimeoutError: If no response is received within the timeout period
            LanguageServerResponseError: If the server returns an error response
        """
        if self.shutdown_flag:
            self.logger.error(f"Attempted to call method '{method_name}' after shutdown.")
            raise ConnectionError("Cannot call methods after shutdown.")

        with self.next_id_lock:
            current_id = self.next_id
            self.next_id += 1

        cond = threading.Condition()
        with self.dict_lock:
            self.event_dict[current_id] = cond

        with cond:
            try:
                self._send_request(method_name, kwargs, current_id)
                if self.shutdown_flag:
                    return None

                if not cond.wait(timeout=self.timeout):
                    with self.dict_lock:
                        self.event_dict.pop(current_id, None)
                        self.response_dict.pop(current_id, None)
                    self.logger.error(f"Timeout waiting for response to {method_name}")
                    raise TimeoutError(f"Timeout waiting for response to {method_name}")

                with self.dict_lock:
                    result, error = self.response_dict.pop(current_id)
                    self.event_dict.pop(current_id)

                if error:
                    self.logger.error(f"Server error response to {method_name}: {error}")
                    raise LanguageServerResponseError(error.get("code"), error.get("message"), error.get("data"))
                return result

            except (TimeoutError, LanguageServerResponseError):
                # Let these expected exceptions propagate up
                raise

            except Exception as e:
                # Log and wrap unexpected exceptions
                self.logger.exception(f"Unexpected error in call_method '{method_name}'")
                raise ConnectionError(f"LSP communication error: {str(e)}") from e

    def shutdown(self):
        """
        Sends a `shutdown` request followed by an `exit` notification to terminate the language server session.

        The `shutdown` request informs the server that the client intends to shut down. The server should
        perform any cleanup tasks and return a response. After receiving the response, the client sends
        the `exit` notification to terminate the connection.

        LSP Spec:
        - Method: `shutdown`:
            - Request to notify the server of an impending shutdown.
            - No parameters.
        - Method: `exit` (notification):
            - Notification to cleanly exit the server process.
            - No parameters or response.

        Returns:
            None
        """
        if self.shutdown_flag:
            self.logger.debug("Shutdown already in progress")
            return

        self.shutdown_flag = True

        # Only attempt graceful shutdown if process is still alive
        if self.process.poll() is None:
            try:
                self.call_method("shutdown")
                self.send_notification("exit")
            except Exception as e:
                self.logger.warning(f"Error during graceful shutdown: {str(e)}")

        # Terminate the process if still alive
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)  # Wait up to 5 seconds for termination
                except subprocess.TimeoutExpired:
                    self.logger.warning("Process did not terminate, forcing kill")
                    self.process.kill()  # Force kill if terminate doesn't work
                    self.process.wait()
        except Exception as e:
            self.logger.error(f"Error killing process: {str(e)}")

        # Clean up threads
        try:
            if self.is_alive():
                self.join(timeout=5.0)
        except Exception as e:
            self.logger.error(f"Error joining main thread: {str(e)}")

        try:
            if not self.stderr_thread.join(timeout=5.0):
                self.logger.warning("stderr_thread did not terminate within timeout")
        except Exception as e:
            self.logger.error(f"Error joining stderr thread: {str(e)}")

        # Clear any pending requests
        with self.dict_lock:
            self.event_dict.clear()
            self.response_dict.clear()
            self.diagnostic_events.clear()
            self.diagnostic_results.clear()

    def lsp_initialize(self, root_path, initialization_options=None, capabilities=DEFAULT_CAPABILITIES,
                       trace="off",
                       workspace_folders=None):
        """
        Sends an `initialize` request to the language server to begin a session.

        This is the first request sent to the server during setup. It initializes the connection
        by providing the client's configuration and environment details. After receiving the server's
        response, it sends the `initialized` notification to indicate that initialization is complete.

        LSP Spec:
        - Method: `initialize`
        - Params:
            - `processId` (integer | null): The client's process ID or `null` if unavailable.
            - `rootPath` (string | null): Deprecated, use `rootUri`. Represents the root path of the workspace.
            - `rootUri` (string | null): The root URI of the workspace folder.
            - `initializationOptions` (any): Additional options for server initialization.
            - `capabilities` (object): Client capabilities.
            - `trace` (string): The initial trace setting (`off`, `messages`, or `verbose`).
            - `workspaceFolders` (array | null): The workspace folders the client manages.
        - Response:
            - Returns server capabilities, such as supported features and options.

        Args:
            root_path (str | None): The workspace's root path (deprecated, use root_uri).
            initialization_options (dict | None): Additional options for initialization.
            capabilities (dict): Client capabilities as defined by the LSP spec.
            trace (str): Trace level (`off`, `messages`, or `verbose`).
            workspace_folders (list | None): A list of workspace folders.

        Returns:
            dict: The server's response, typically containing its capabilities and configuration.
        """
        try:
            process_id = os.getpid()  # Get the current process ID
            resp = self.call_method(
                "initialize",
                processId=process_id,
                rootPath=root_path,
                rootUri=to_uri(root_path),
                initializationOptions=initialization_options,
                capabilities=capabilities,
                trace=trace,
                workspaceFolders=workspace_folders,
            )
            self.send_notification("initialized")
            return resp
        except Exception as e:
            self.logger.exception(f"Error during LSP initialization: {str(e)}")
            raise

    def lsp_did_open(self, text_document: TextDocumentItem):
        """
        Sends a `textDocument/didOpen` notification to the language server.

        This notification informs the server that a text document is now open in the client.
        The server is expected to start tracking the document and maintain its state.
        Typically used for real-time features like diagnostics or hover information.

        LSP Spec:
        - Method: `textDocument/didOpen`
        - Params:
            - `textDocument`: The text document that was opened.

        Args:
            text_document (TextDocumentItem): The text document to notify the server about.
        """
        self.send_notification("textDocument/didOpen", textDocument=text_document.to_dict())

    def last_lsp_document_diagnostics(self, uri: str) -> Union[dict, None]:
        """
        Get the last diagnostics for a document URI.

        Args:
            uri (str): The document URI to get diagnostics for.

        Returns:
            dict | None: The latest diagnostic results for the URI, or None if no diagnostics exist.
        """
        with self.dict_lock:
            return self.latest_diagnostics.get(uri)

    def lsp_diagnostics_wait(self, text_document: TextDocumentItem, timeout: float = 60.0) -> Union[dict, None]:
        """
        Sends a `textDocument/didOpen` notification and waits for diagnostic results from the server.

        Diagnostics provide feedback such as errors, warnings, or informational messages about
        the document's content. This method handles the opening of the document and waits for
        the diagnostics result from the server, up to the specified timeout.

        LSP Spec:
        - Method: `textDocument/didOpen` (notification).
        - Diagnostics are provided asynchronously by the server using `textDocument/publishDiagnostics`.

        Args:
            text_document (TextDocumentItem): The text document to open and analyze.
            timeout (float): Maximum time (in seconds) to wait for the diagnostic results.

        Returns:
            dict: Diagnostic results for the text document, or None if it times out.
        """
        if self.shutdown_flag:
            self.logger.debug("Attempted to wait for diagnostics after shutdown, returning None.")
            return None

        cond = threading.Condition()
        with self.dict_lock:
            self.diagnostic_events[text_document.uri] = cond

        self.send_notification("textDocument/didOpen", textDocument=text_document.to_dict())

        with cond:
            if not cond.wait(timeout=timeout):
                with self.dict_lock:
                    self.diagnostic_events.pop(text_document.uri, None)
                    self.diagnostic_results.pop(text_document.uri, None)
                return None
            with self.dict_lock:
                result = self.diagnostic_results.pop(text_document.uri, None)
                self.diagnostic_events.pop(text_document.uri, None)
            return result

    def lsp_document_symbols(self, text_document: TextDocumentIdentifier) -> Union[List[DocumentSymbol], List[SymbolInformation]]:
        """
        Sends a `textDocument/documentSymbol` request to retrieve symbols in a document.

        Document symbols provide hierarchical or flat representations of programming constructs,
        such as functions, variables, and classes, defined within the text document.

        LSP Spec:
        - Method: `textDocument/documentSymbol`
        - Params:
            - `textDocument`: The text document identifier.
        - Result:
            - A list of `DocumentSymbol` (hierarchical symbols) or `SymbolInformation` (flat symbols).

        Args:
            text_document (TextDocumentIdentifier): The identifier of the text document.

        Returns:
            list[DocumentSymbol] | list[SymbolInformation]:
            - A list of hierarchical `DocumentSymbol` objects if supported.
            - Otherwise, a list of flat `SymbolInformation` objects.
        """
        if self.shutdown_flag:
            self.logger.debug("Attempted to get document symbols after shutdown, returning empty list.")
            return []

        result_dict = self.call_method("textDocument/documentSymbol", textDocument=text_document.to_dict())

        if not result_dict:
            return []

        # Check if it matches DocumentSymbol structure
        is_document_symbol = True
        for sym in result_dict:
            if 'selectionRange' not in sym or 'range' not in sym:
                is_document_symbol = False
                break

        if is_document_symbol:
            return [DocumentSymbol.from_dict(sym) for sym in result_dict]
        else:
            return [SymbolInformation.from_dict(sym) for sym in result_dict]

    def lsp_type_definition(self, text_document: TextDocumentIdentifier, position: Position) -> List[Location]:
        """
        Sends a `textDocument/typeDefinition` request to retrieve the type definitions at a specific position.

        This request is used to find where a type is defined, such as the definition of a class,
        interface, or other type construct.

        LSP Spec:
        - Method: `textDocument/typeDefinition`
        - Params:
            - `textDocument`: The text document identifier.
            - `position`: The position in the document for which type definition is requested.
        - Result:
            - A list of `Location` objects pointing to the type definitions.

        Args:
            text_document (TextDocumentIdentifier): The identifier of the text document.
            position (Position): The position within the document to query for type definitions.

        Returns:
            list[Location]: A list of locations where the type definitions are found.
        """
        if self.shutdown_flag:
            self.logger.debug("Attempted to get type definitions after shutdown, returning empty list.")
            return []
        result = self.call_method("textDocument/typeDefinition", textDocument=text_document.to_dict(), position=position.to_dict())
        if not result:
            return []
        if not isinstance(result, list):
            result = [result]
        return [Location.from_dict(loc) for loc in result]

    def is_running(self) -> bool:
        """
        Checks if the LSP client is still running and not shut down.

        Returns:
            bool: True if running and process alive, False if shutdown or process ended.
        """
        if self.shutdown_flag:
            return False
        if self.process.poll() is not None:
            return False
        return True


if __name__ == "__main__":

    def main():
        """
        NOTES
        lsp spec https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/
        lsp client reference https://github.com/yeger00/pylspclient/tree/main
        lsp client reference https://github.com/microsoft/multilspy/tree/main
        """

        lsp_client = LspClient(lsp_cmd=["python", "-m", "pylsp"])
        lsp_client.start()

        # Initialize our lsp client with our current working directory
        root_path = os.path.dirname(os.path.abspath(__file__))
        initialize_response = lsp_client.lsp_initialize(root_path)
        print(json.dumps(initialize_response, indent=2, sort_keys=True))

        # Inspect our file
        #
        # We signal didOpen and wait for the result. This is potentially a race condition, but not many LSP support
        # requesting diagnostics directly. So this is the only method to get diagnostics out of an LSP, that's expected
        # to work with most LSPs.
        language_id = LanguageIdentifier.PYTHON
        file_path = os.path.join(root_path, "client.py")
        text = open(file_path, "r").read()
        uri = to_uri(file_path)
        print(
            json.dumps(
                lsp_client.lsp_diagnostics_wait(
                    TextDocumentItem(
                        uri=uri,
                        languageId=language_id,
                        version=1,
                        text=text,
                    ),
                ),
                indent=2,
                sort_keys=True
            )
        )

        # Get all symbols from file
        symbols = lsp_client.lsp_document_symbols(TextDocumentIdentifier(uri=uri))
        for symbol in symbols:
            print(f"- {symbol.name}:")
            print(f"  Kind: {SymbolKind(symbol.kind).name}")  # Convert int to enum name

            # Only print detail if it exists (DocumentSymbol has it, SymbolInformation doesn't)
            if hasattr(symbol, 'detail') and symbol.detail:
                print(f"  Detail: {symbol.detail}")

            # Only print children if it exists (DocumentSymbol has it, SymbolInformation doesn't)
            if hasattr(symbol, 'children') and symbol.children:
                print("  Children:")
                for child in symbol.children:
                    print(f"    - {child.name} ({child.kind})")

            # Print container name if it exists (SymbolInformation has it, DocumentSymbol doesn't)
            if hasattr(symbol, 'containerName') and symbol.containerName:
                print(f"  Container: {symbol.containerName}")
        lsp_client.shutdown()

    main()
