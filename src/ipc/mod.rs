//! Local authenticated IPC and durable job control.
//!
//! The public JSON documents in this module are versioned automation
//! contracts. The transport is loopback-only and every operation, including
//! read-only inspection, requires an explicit bearer grant. Processing is
//! delegated to the normal `denoize plan` and finite-execution CLI paths so the
//! service cannot bypass resource admission, regular-file validation, atomic
//! publication, or signed execution receipts.

mod contracts;
mod control;
mod server;
mod storage;
mod transport;

pub use contracts::{
    IpcCapability, IpcDestinationSummary, IpcDiscovery, IpcDryRunReport, IpcError,
    IpcGrantDocument, IpcGrantPolicy, IpcGrantSummary, IpcHistoryEntry, IpcHistoryReport,
    IpcJobKind, IpcJobSpec, IpcJobState, IpcJobStatus, IpcLimits, IpcOperation, IpcRequestEnvelope,
    IpcResourceSummary, IpcResponseEnvelope, IpcResponseResult, IPC_CAPABILITY_SCHEMA,
    IPC_DISCOVERY_SCHEMA, IPC_DRY_RUN_SCHEMA, IPC_GRANT_SCHEMA, IPC_HISTORY_SCHEMA,
    IPC_JOB_STATUS_SCHEMA, IPC_REQUEST_SCHEMA, IPC_RESPONSE_SCHEMA, IPC_SCHEMA_VERSION,
};
pub use server::{initialize_ipc_state, run_ipc_server, IpcServerConfig};
pub use transport::IpcClient;

#[doc(hidden)]
pub use control::{check_process_control_boundary, install_process_control, ProcessControlGuard};

pub(crate) use control::{check_publication_fence, write_control_file, ControlAction};
pub(crate) use storage::{
    authorize_job_paths, unix_millis, validate_bound_job_paths, write_private_bytes,
    write_private_json, AuthenticatedGrant, StateStore,
};
pub(crate) use transport::{read_request, write_response};
