"""Public, stable ZeSolver API v1.

Import the whole surface with::

    from zesolver.api.v1 import (
        API_VERSION, get_api_info, probe,
        create_solver_runtime, SolverRuntime, SolverSession,
        SolveRequest, SolveHints, SolveOptions, SolveResult,
        BackendPolicy, GpuPolicy, NetworkPolicy, WritePolicy,
        SolveStatus, FailureCode, CanonicalWcsHeader,
        CancellationToken, ProgressPhase, ProgressEvent,
        CapabilityState, CapabilityAvailability, CapabilityUnavailableReason,
        ReadinessReport, ConfigurationSession,
        ZeSolverApiError, SolverClosedError, InvalidRequestError,
    )
"""

from .cancellation import CancellationToken
from .errors import InvalidRequestError, SolverClosedError, ZeSolverApiError
from .models import (
    API_MAJOR,
    API_MINOR,
    API_VERSION,
    ApiInfo,
    BackendPolicy,
    CanonicalWcsHeader,
    CapabilityAvailability,
    CapabilityState,
    CapabilityUnavailableReason,
    FailureCode,
    GpuPolicy,
    NetworkPolicy,
    ProgressEvent,
    ProgressPhase,
    RuntimeProbe,
    ReadinessReport,
    SolveHints,
    SolveOptions,
    SolveRequest,
    SolveResult,
    SolveStatus,
    WritePolicy,
)
from .probe import get_api_info, probe
from .readiness import open_configuration, readiness
from .session import ConfigurationSession
from .runtime import SolverRuntime, SolverSession, create_solver_runtime

__all__ = [
    # version
    "API_VERSION",
    "API_MAJOR",
    "API_MINOR",
    # metadata / probe
    "ApiInfo",
    "get_api_info",
    "RuntimeProbe",
    "ReadinessReport",
    "probe",
    # readiness / configuration access
    "readiness",
    "open_configuration",
    "ConfigurationSession",
    # capabilities
    "CapabilityAvailability",
    "CapabilityUnavailableReason",
    "CapabilityState",
    # policies
    "BackendPolicy",
    "GpuPolicy",
    "NetworkPolicy",
    "WritePolicy",
    # solve models
    "SolveHints",
    "SolveOptions",
    "SolveRequest",
    "CanonicalWcsHeader",
    "SolveStatus",
    "FailureCode",
    "SolveResult",
    # progress / cancellation
    "ProgressPhase",
    "ProgressEvent",
    "CancellationToken",
    # errors
    "ZeSolverApiError",
    "SolverClosedError",
    "InvalidRequestError",
    # lifecycle
    "create_solver_runtime",
    "SolverRuntime",
    "SolverSession",
]
