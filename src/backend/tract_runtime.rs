//! Shared tract runtime preparation for CPU, Metal, and CUDA execution.

use std::sync::Arc;

use crate::AcceleratorRuntime;
use tract_onnx::prelude::TypedModel;
use tract_onnx::tract_core::runtime::{runtime_for_name, Runnable};

pub(crate) type SharedRunnable = Arc<dyn Runnable>;

pub(crate) fn prepare(
    model: TypedModel,
    runtime: AcceleratorRuntime,
    context: &str,
) -> Result<SharedRunnable, String> {
    let runtime_name = runtime.name();
    let runtime = runtime_for_name(runtime_name)
        .map_err(|error| format!("select {runtime_name} runtime for {context}: {error:#}"))?
        .ok_or_else(|| format!("{runtime_name} runtime is not registered for {context}"))?;
    runtime
        .prepare(model)
        .map(Arc::from)
        .map_err(|error| format!("prepare {context} with {runtime_name}: {error:#}"))
}
