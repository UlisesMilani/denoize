//! Accessible, bounded and audio-thread-independent plug-in editor.
//!
//! The crate deliberately keeps windowing, software drawing, native
//! accessibility adapters and the two documented unsafe lifetime bridges out
//! of the real-time CLAP adapter. Parameter values are atomic, editor gestures
//! use a fixed-capacity queue, and overflow collapses to one final gesture per
//! parameter.

mod accessibility;
mod layout;
mod model;
mod renderer;
mod window;

pub use model::{
    AutomationGesture, ControlKind, DisplayUnit, EditorModel, MAX_PARAMETERS, ModelError,
    ParameterSpec,
};
pub use window::PluginEditor;
