"""Comprehensive error handling for the backend API."""

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import logging
import traceback
from typing import Union
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ChatbotException(Exception):
    """Base exception for chatbot errors."""

    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class InitializationError(ChatbotException):
    """Raised when chatbot initialization fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=500, details=details)


class ModelNotFoundError(ChatbotException):
    """Raised when required model files are not found."""

    def __init__(self, message: str = "Required model files not found", details: dict = None):
        super().__init__(message, status_code=404, details=details)


class ChatProcessingError(ChatbotException):
    """Raised when message processing fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=500, details=details)


class ValidationError(ChatbotException):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=422, details=details)


class SessionError(ChatbotException):
    """Raised when session management fails."""

    def __init__(self, message: str = "Session error occurred", details: dict = None):
        super().__init__(message, status_code=400, details=details)


def create_error_response(
    status_code: int,
    message: str,
    details: dict = None,
    request_id: str = None
) -> dict:
    """Create standardized error response."""
    error_response = {
        "error": True,
        "status_code": status_code,
        "message": message,
        "timestamp": None,  # Would add timestamp in production
    }

    if details:
        error_response["details"] = details

    if request_id:
        error_response["request_id"] = request_id

    return error_response


async def chatbot_exception_handler(request: Request, exc: ChatbotException):
    """Handle custom chatbot exceptions."""
    logger.error(
        f"ChatbotException: {exc.message}",
        extra={
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path
        }
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            details=exc.details
        )
    )


async def validation_exception_handler(request: Request, exc: Union[RequestValidationError, ValidationError]):
    """Handle validation errors."""
    logger.warning(
        f"Validation error: {str(exc)}",
        extra={"path": request.url.path}
    )

    details = {}
    if isinstance(exc, RequestValidationError):
        details = {"validation_errors": exc.errors()}

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=create_error_response(
            status_code=422,
            message="Invalid input data",
            details=details
        )
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={"path": request.url.path}
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            status_code=exc.status_code,
            message=str(exc.detail)
        )
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={
            "path": request.url.path,
            "traceback": traceback.format_exc()
        }
    )

    # In production, don't expose internal errors
    message = "An unexpected error occurred. Please try again later."

    # In development, include more details
    details = {}
    if logger.level == logging.DEBUG:
        details = {
            "error_type": type(exc).__name__,
            "error_details": str(exc)
        }

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            status_code=500,
            message=message,
            details=details
        )
    )


def safe_execute(func, *args, fallback_value=None, error_message="Operation failed", **kwargs):
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        fallback_value: Value to return on error
        error_message: Error message to log
        **kwargs: Keyword arguments for the function

    Returns:
        Result of function or fallback_value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}", exc_info=True)
        return fallback_value


class ErrorHandler:
    """Context manager for consistent error handling."""

    def __init__(self, operation_name: str, fallback_value=None):
        self.operation_name = operation_name
        self.fallback_value = fallback_value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(
                f"Error in {self.operation_name}: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
            # Return True to suppress exception
            return False
        return True


def validate_input(data: dict, required_fields: list, field_types: dict = None) -> tuple[bool, str]:
    """
    Validate input data.

    Args:
        data: Data dictionary to validate
        required_fields: List of required field names
        field_types: Optional dictionary mapping field names to expected types

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    # Check field types if specified
    if field_types:
        for field, expected_type in field_types.items():
            if field in data and not isinstance(data[field], expected_type):
                return False, f"Field '{field}' must be of type {expected_type.__name__}"

    return True, ""


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        text = str(text)

    # Trim to max length
    text = text[:max_length]

    # Remove potential injection attempts (basic)
    dangerous_patterns = ['<script', 'javascript:', 'onerror=']
    for pattern in dangerous_patterns:
        text = text.replace(pattern, '')

    return text.strip()
