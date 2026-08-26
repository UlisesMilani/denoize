# Accessible plug-in editor

v0.79.0 implements the Stage 28c custom editor gate for both native CLAP
descriptors. `denoize` exposes all seven DSP parameters and `denoize Neural`
exposes all four neural parameters. The editor owns no private processing
setting: every visible value is the same stable CLAP parameter used by generic
host controls, automation, and portable state.

## Host support and fallback

The editor accepts only a native embedded child-window configuration:

- X11 on Linux;
- Win32 on Windows; and
- Cocoa on macOS.

Floating windows and Wayland custom windows return unsupported. A failed or
unsupported custom-editor request leaves `clap.params`, activation, audio
processing, automation, and state available, so a host can render its generic
parameter editor without a denoize-specific recovery path. Stage 28c does not
claim that every host chooses that fallback automatically.

The window starts at 640 × 400 logical pixels. The accepted range is 480 × 300
through 1280 × 800. `adjust_size` never expands an offer made by the host;
undersized offers are rejected, oversized offers are bounded, and `set_size`
rejects values outside the negotiated range. Scale factors must be finite and
within 0.5 through 4.0.

## Keyboard, pointer, and accessibility

Every control has a visible label, current value, non-overlapping hit target,
and high-contrast focus indicator. The complete editor is usable without a
pointer:

- `Tab` and `Shift+Tab` move focus with bounded wraparound;
- arrow keys make one parameter step;
- `Page Up` and `Page Down` make one page step;
- `Home` and `End` select the parameter bounds; and
- `Space` or `Enter` toggles a binary parameter.

Pointer click/drag and wheel input update the same model. A deterministic
software renderer avoids GPU/device initialization in the plug-in and makes
focus/content frames testable byte-for-byte.

AccessKit exposes the editor root and each parameter as a native toggle,
slider, or choice with name, numeric range, current value, supported actions,
and keyboard focus. Platform adapters are AT-SPI on Linux, UI Automation on
Windows, and NSAccessibility on macOS. Assistive `SetValue`, increment,
decrement, and focus actions enter the same bounded automation path as keyboard
and pointer gestures. The implementation follows the semantics of the
[WAI-ARIA slider pattern](https://www.w3.org/WAI/ARIA/apg/patterns/slider/) and
uses the native adapter architecture documented by
[AccessKit](https://github.com/AccessKit/accesskit).

## Threading and automation boundary

Window creation, drawing, accessibility, and input remain on the host-approved
main/UI thread. Audio callbacks never open a window, draw, allocate editor
state, call an accessibility API, or wait for UI work.

Parameter mirrors are atomic. UI changes enter a fixed 128-entry lock-free
queue. If it fills, one 64-bit overflow mask retains the final value for each of
at most 63 controls; the current editor has at most seven. The next host flush
emits one ordered `Begin` → `Value` → `End` gesture. If the host output queue
accepts only part of that gesture, the main-thread adapter resumes at the exact
unwritten stage without duplicating `Begin` or applying the value twice. Host
automation updates the editor atomically but never generate feedback
automation.

The pinned `clack-extensions` 0.1.1 GUI adapter maps an inner `Result` through
`Option::is_some` for several callbacks and can therefore report rejected
`set_size` and `set_parent` calls as success. denoize supplies a narrowly scoped
standard `clap.gui` vtable that retains Clack's instance and panic handling but
returns the actual boolean contract. This boundary is covered through the raw
host-facing ABI, not only by direct Rust calls. The upstream implementation is
auditable in the [official Clack repository](https://github.com/prokopyl/clack).

## Lifecycle and failure isolation

Creation rejects duplicate instances. Show/hide before parenting fails;
parenting is accepted once; show and hide are idempotent afterward; destroy is
safe after any exercised failure. Native host handles are retained only for the
strictly shorter editor lifetime. Two documented unsafe bridges cover that
lifetime and native accessibility handles; the plug-in crate otherwise denies
unsafe code.

Editor construction, rendering, resizing, accessibility, or host-callback
failure is separate from the audio processor lifecycle. The official CLAP
validator still exercises both descriptors independently of GUI creation.

## Evidence and limits

CI and the tagged release workflow run:

- model, layout, renderer, accessibility-action, lifecycle, resize, queue
  overflow, host-feedback, and partial-output retry unit tests;
- editor crate type checks for Windows x86-64, macOS x86-64, and macOS arm64;
- the pinned official CLAP validator contract (81 results: 68 success, 13
  capability skips, zero other results); and
- a real `clack-host` 0.1.1 process with a baseview X11 parent under Xvfb.

The real-host gate opens both editors, rejects unsupported/invalid lifecycle
calls, embeds and renders a child, samples its pixels, injects an XTEST bypass
click, receives exactly three host automation events, and completes repeated
hide/show/destroy. A closed `denoize-plugin-editor-evidence-v1` document binds
the tag, source commit, dependency revisions, per-descriptor observations, and
report SHA-256. The JSON and report share one GitHub Sigstore attestation and
the final release verifier authenticates both against the tag and release
workflow.

The signed real-host claim is Linux X11 x86-64 only. Windows and macOS adapters
are compile-checked but do not yet have signed proprietary-host matrices.
Wayland and floating custom windows are unsupported. VST3 custom-view parity is
not inferred from the CLAP result; the v0.78 VST3 matrix continues to leave
`custom_editor` false until a native VST3 host opens and exercises that view.
