#![no_main]

use std::io::Write as _;

use denoize::{
    ExecutionPlan, ReceiptPublicKey, ReceiptSecretKey, ReceiptTrustPolicy, SignedExecutionReceipt,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 1024 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let _ = serde_json::from_slice::<ExecutionPlan>(data);
    let _ = serde_json::from_slice::<SignedExecutionReceipt>(data);
    let _ = serde_json::from_slice::<ReceiptPublicKey>(data);
    let _ = serde_json::from_slice::<ReceiptSecretKey>(data);
    let _ = serde_json::from_slice::<ReceiptTrustPolicy>(data);

    let mut document = tempfile::Builder::new()
        .prefix("denoize-fuzz-")
        .suffix(".json")
        .tempfile()
        .expect("create fuzz document");
    document
        .write_all(data)
        .expect("write bounded fuzz document");
    document.flush().expect("flush fuzz document");
    let _ = ExecutionPlan::from_file(document.path());
    let _ = SignedExecutionReceipt::from_file(document.path());
    let _ = ReceiptPublicKey::from_file(document.path());
    let _ = ReceiptSecretKey::from_file(document.path());
    let _ = ReceiptTrustPolicy::from_file(document.path());

    let mut bundle = tempfile::Builder::new()
        .prefix("denoize-fuzz-")
        .suffix(".dmb")
        .tempfile()
        .expect("create fuzz bundle");
    bundle.write_all(data).expect("write bounded fuzz bundle");
    bundle.flush().expect("flush fuzz bundle");
    let _ = denoize::models::inspect_offline_bundle(bundle.path());
});
