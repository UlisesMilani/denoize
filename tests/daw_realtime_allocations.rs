#[cfg(feature = "aec")]
use denoize::{
    sign_aec_promotion_evidence, AecConfig, AecEvidenceMetric, AecEvidenceMetricOperator,
    AecEvidenceStratum, AecEvidenceStratumKind, AecPromotionEvidencePayload, AecSession,
};
use denoize::{DawParameters, DawRealtimeProcessor};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

thread_local! {
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: the request is forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the allocation originated from the system allocator above.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: the request is forwarded unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        // SAFETY: the allocation originated from the system allocator above.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[inline]
fn record_allocation() {
    if RECORDING
        .try_with(|recording| recording.get())
        .unwrap_or(false)
    {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn callback_processing_performs_zero_allocations() {
    let mut processor = DawRealtimeProcessor::new(48_000.0, 2).unwrap();
    let parameters = DawParameters::default();
    let input_left = vec![0.0125_f32; 256];
    let input_right = vec![-0.009_f32; 256];
    let mut output_left = vec![0.0_f32; 256];
    let mut output_right = vec![0.0_f32; 256];

    // Initialize code paths and the thread-local gate before measuring. Only
    // allocations on this callback thread belong to the real-time contract;
    // libtest and other process threads may allocate concurrently.
    RECORDING.with(|recording| recording.set(false));
    {
        let inputs = [input_left.as_slice(), input_right.as_slice()];
        let mut outputs = [output_left.as_mut_slice(), output_right.as_mut_slice()];
        processor
            .process_f32(&inputs, &mut outputs, &parameters)
            .unwrap();
    }

    ALLOCATIONS.store(0, Ordering::Relaxed);
    RECORDING.with(|recording| recording.set(true));
    let result = 'processing: {
        let inputs = [input_left.as_slice(), input_right.as_slice()];
        let mut outputs = [output_left.as_mut_slice(), output_right.as_mut_slice()];
        for _ in 0..1_000 {
            if let Err(error) = processor.process_f32(&inputs, &mut outputs, &parameters) {
                break 'processing Err(error);
            }
        }
        break 'processing Ok::<(), String>(());
    };
    RECORDING.with(|recording| recording.set(false));

    result.unwrap();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(
        allocations, 0,
        "audio callback allocated {allocations} times"
    );
}

#[cfg(feature = "aec")]
fn aec_metric(name: &str, operator: AecEvidenceMetricOperator, limit: f64) -> AecEvidenceMetric {
    AecEvidenceMetric {
        metric: name.into(),
        value: limit,
        operator,
        limit,
        passed: true,
    }
}

#[cfg(feature = "aec")]
fn aec_metrics(kind: AecEvidenceStratumKind) -> Vec<AecEvidenceMetric> {
    use AecEvidenceMetricOperator::{GreaterOrEqual, LessOrEqual};
    let mut metrics = vec![
        aec_metric("latency.algorithmic-plus-buffering-ms", LessOrEqual, 20.0),
        aec_metric("output.duration-error-frames", LessOrEqual, 0.0),
        aec_metric("output.non-finite-samples", LessOrEqual, 0.0),
    ];
    match kind {
        AecEvidenceStratumKind::FarEndOnly => {
            metrics.push(aec_metric("echo.erle-db", GreaterOrEqual, 10.0));
            metrics.push(aec_metric("perceptual.aecmos-far-end", GreaterOrEqual, 3.5));
        }
        AecEvidenceStratumKind::NearEndOnly => {
            metrics.push(aec_metric(
                "content.word-accuracy-regression",
                LessOrEqual,
                0.02,
            ));
            metrics.push(aec_metric("near-end.attenuation-db", LessOrEqual, 1.0));
        }
        AecEvidenceStratumKind::DoubleTalk => {
            metrics.push(aec_metric(
                "content.word-accuracy-regression",
                LessOrEqual,
                0.02,
            ));
            metrics.push(aec_metric("near-end.attenuation-db", LessOrEqual, 1.5));
            metrics.push(aec_metric(
                "perceptual.aecmos-double-talk",
                GreaterOrEqual,
                3.2,
            ));
        }
        AecEvidenceStratumKind::Transition => {
            metrics.push(aec_metric("near-end.attenuation-db", LessOrEqual, 1.5));
            metrics.push(aec_metric("reset.stale-output-frames", LessOrEqual, 0.0));
            metrics.push(aec_metric(
                "transition.reconvergence-ms",
                LessOrEqual,
                500.0,
            ));
        }
        AecEvidenceStratumKind::Impairment => {
            metrics.push(aec_metric(
                "content.word-accuracy-regression",
                LessOrEqual,
                0.02,
            ));
            metrics.push(aec_metric("echo.erle-db", GreaterOrEqual, 6.0));
            metrics.push(aec_metric("near-end.attenuation-db", LessOrEqual, 1.5));
            metrics.push(aec_metric("perceptual.aecmos", GreaterOrEqual, 3.0));
        }
    }
    metrics.sort_by(|left, right| left.metric.cmp(&right.metric));
    metrics
}

#[cfg(feature = "aec")]
fn promoted_aec_session() -> AecSession {
    const STRATA: &[(&str, AecEvidenceStratumKind)] = &[
        ("background-noise", AecEvidenceStratumKind::Impairment),
        ("clipping", AecEvidenceStratumKind::Impairment),
        ("clock-drift-negative", AecEvidenceStratumKind::Transition),
        ("clock-drift-positive", AecEvidenceStratumKind::Transition),
        ("delay-jump", AecEvidenceStratumKind::Transition),
        ("delay-negative", AecEvidenceStratumKind::Transition),
        ("delay-positive", AecEvidenceStratumKind::Transition),
        ("double-talk", AecEvidenceStratumKind::DoubleTalk),
        ("far-end-clean", AecEvidenceStratumKind::FarEndOnly),
        ("linear-path", AecEvidenceStratumKind::FarEndOnly),
        ("music-playback", AecEvidenceStratumKind::Impairment),
        ("near-end-clean", AecEvidenceStratumKind::NearEndOnly),
        ("nonlinear-speaker", AecEvidenceStratumKind::Impairment),
        ("real-device", AecEvidenceStratumKind::Impairment),
        ("reference-loss", AecEvidenceStratumKind::Transition),
        ("room-change", AecEvidenceStratumKind::Transition),
        ("route-change", AecEvidenceStratumKind::Transition),
    ];
    let config = AecConfig {
        sample_rate: 16_000,
        block_size_samples: 128,
        tail_samples: 1_024,
        maximum_delay_samples: 256,
        delay_analysis_samples: 2_048,
        ..AecConfig::default()
    };
    let payload = AecPromotionEvidencePayload {
        completed_at_unix_seconds: 1,
        implementation: "native-pfdnlms-v1".into(),
        implementation_source_revision: "fixture".into(),
        implementation_source_sha256: "1".repeat(64),
        configuration_sha256: config.digest().unwrap(),
        corpus_manifest_sha256: "2".repeat(64),
        evaluation_result_sha256: "3".repeat(64),
        listening_result_sha256: "4".repeat(64),
        sample_rate: config.sample_rate,
        block_size_samples: config.block_size_samples,
        tail_samples: config.tail_samples,
        maximum_delay_samples: config.maximum_delay_samples,
        strata: STRATA
            .iter()
            .map(|(id, kind)| AecEvidenceStratum {
                id: (*id).into(),
                kind: *kind,
                cases: 100,
                metrics: aec_metrics(*kind),
            })
            .collect(),
        real_device_cases: 100,
        nonlinear_device_cases: 100,
        delay_transition_cases: 100,
        paced_realtime_blocks: 10_000,
        worst_case_realtime_factor: 0.5,
        callback_allocations: 0,
        callback_locks: 0,
        callback_waits: 0,
        callback_io_operations: 0,
        callback_log_operations: 0,
        deadline_misses: 0,
        stale_frames_after_reset: 0,
        minimum_listeners: 20,
        listener_count: 20,
        listener_preference: 0.5,
        listener_preference_limit: 0.5,
        accepted: true,
    };
    let (secret, public) = denoize::generate_receipt_keypair().unwrap();
    let evidence = sign_aec_promotion_evidence(payload, &secret).unwrap();
    AecSession::prepare(&evidence, &public, config).unwrap()
}

#[cfg(feature = "aec")]
#[test]
fn aec_exact_block_processing_performs_zero_allocations() {
    let session = promoted_aec_session();
    let mut stream = session.stream(1, true).unwrap();
    let frames = stream.block_size_samples();
    let reference = (0..frames)
        .map(|frame| ((frame as f32 * 0.071).sin() * 0.25).clamp(-1.0, 1.0))
        .collect::<Vec<_>>();
    let microphone = reference
        .iter()
        .map(|sample| sample * 0.4)
        .collect::<Vec<_>>();
    let mut output = vec![0.0_f32; frames];

    RECORDING.with(|recording| recording.set(false));
    stream
        .process_block(&microphone, &reference, &mut output)
        .unwrap();
    ALLOCATIONS.store(0, Ordering::Relaxed);
    RECORDING.with(|recording| recording.set(true));
    let result = (|| {
        for _ in 0..1_000 {
            stream.process_block(&microphone, &reference, &mut output)?;
        }
        Ok::<(), String>(())
    })();
    RECORDING.with(|recording| recording.set(false));

    result.unwrap();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(allocations, 0, "AEC callback allocated {allocations} times");

    let mut adapter = session.realtime_adapter(2, true).unwrap();
    let microphone_quantum = vec![0.05_f32; 37];
    let reference_quantum = vec![0.02_f32; 37];
    let mut output_quantum = vec![0.0_f32; 37];
    for _ in 0..4 {
        adapter
            .process(&microphone_quantum, &reference_quantum, &mut output_quantum)
            .unwrap();
    }
    ALLOCATIONS.store(0, Ordering::Relaxed);
    RECORDING.with(|recording| recording.set(true));
    let result = (|| {
        for _ in 0..1_000 {
            adapter.process(&microphone_quantum, &reference_quantum, &mut output_quantum)?;
        }
        Ok::<(), String>(())
    })();
    RECORDING.with(|recording| recording.set(false));
    result.unwrap();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(
        allocations, 0,
        "AEC arbitrary-quantum callback allocated {allocations} times"
    );
}
