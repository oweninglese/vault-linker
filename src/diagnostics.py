from pydantic import BaseModel
from enum import Enum
from typing import Optional, Any

class Severity(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Diagnostic(BaseModel):
    code: str
    message: str
    severity: Severity
    path: Optional[str] = None
    context: Optional[Any] = None
