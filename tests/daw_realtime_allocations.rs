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
