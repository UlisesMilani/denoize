# Desktop app

The Tauri desktop app provides local access to the main denoize workflows
without sending audio to a service. Prebuilt packages are available on the
[release page](https://github.com/penguin425/denoize/releases/latest).

## Included workflows

- single-file and batch denoising
- bounded previews with original, processed, and removed-signal audition
- diagnostics, assessment, deterministic restoration, and supported gated workflows
- portable projects, execution-plan previews, and signed receipts
- managed model installation, verification, repair, and catalog inspection
- live input-to-output processing for live-capable backends
- DAW preset, session, model, and latency inspection
- recoverable application updates

The interface supports English and Japanese, keyboard navigation, visible
focus, reduced motion, forced colors, and screen-reader semantics.

## Local safety boundary

Final file, batch, and preview work runs in supervised child processes. A
worker crash is isolated from the UI, and cancelled or rejected work cannot
publish a later output. Recovery records are owner-private and apply only to
verified denoize staging files; existing outputs and batch journals are not
deleted by recovery.

Model credentials and alternate download settings are session-only. They are
not saved in presets or exported configuration. Diagnostic exports omit audio,
paths, URLs, credentials, device names, and free-form errors.

Resource controls cover aggregate RAM, temporary output, GPU-memory
reservation, and GPU-worker concurrency. Reproducibility mode serializes work
and uses stable model seeds where applicable.

## Development

```sh
cd apps/desktop
npm ci
npm run tauri -- dev

# UI and accessibility checks
npm run check:ui
npm run build
npm run test:a11y:webview

# Platform package
npm run tauri -- build
```

FDK-AAC remains an explicit build-time opt-in because it has separate license
terms:

```sh
npm run tauri -- build --features fdk-aac-encoder
```

Linux development requires WebKitGTK 4.1 and GTK 3 development packages. On
Ubuntu 24.04 or later:

```sh
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev patchelf xvfb
```

CLI-compatible behavior and file contracts are documented in the
[CLI reference](cli.md), [Stable JSON contracts](json.md), and the individual
workflow guides in the [documentation index](README.md).
