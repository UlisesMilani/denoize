use crossbeam_queue::ArrayQueue;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

pub const MAX_PARAMETERS: usize = 63;
const AUTOMATION_QUEUE_CAPACITY: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlKind {
    Toggle,
    Continuous,
    Choice(&'static [&'static str]),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisplayUnit {
    Plain,
    Percent,
    Decibels,
    Milliseconds,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParameterSpec {
    pub id: u32,
    pub name: &'static str,
    pub minimum: f64,
    pub maximum: f64,
    pub default: f64,
    pub step: f64,
    pub page_step: f64,
    pub kind: ControlKind,
    pub unit: DisplayUnit,
}

impl ParameterSpec {
    pub fn clamp(self, value: f64) -> f64 {
        if !value.is_finite() {
            return self.default;
        }
        let value = value.clamp(self.minimum, self.maximum);
        match self.kind {
            ControlKind::Toggle => {
                if value >= (self.minimum + self.maximum) * 0.5 {
                    self.maximum
                } else {
                    self.minimum
                }
            }
            ControlKind::Choice(_) => value.round().clamp(self.minimum, self.maximum),
            ControlKind::Continuous => value,
        }
    }

    pub fn normalized(self, value: f64) -> f64 {
        let range = self.maximum - self.minimum;
        if !range.is_finite() || range <= 0.0 {
            0.0
        } else {
            ((self.clamp(value) - self.minimum) / range).clamp(0.0, 1.0)
        }
    }

    pub fn value_from_normalized(self, normalized: f64) -> f64 {
        self.clamp(self.minimum + normalized.clamp(0.0, 1.0) * (self.maximum - self.minimum))
    }

    pub fn display(self, value: f64) -> String {
        let value = self.clamp(value);
        if let ControlKind::Choice(labels) = self.kind {
            let index = (value - self.minimum).round().max(0.0) as usize;
            return labels.get(index).copied().unwrap_or("Unknown").to_owned();
        }
        if matches!(self.kind, ControlKind::Toggle) {
            return if value >= (self.minimum + self.maximum) * 0.5 {
                "On".to_owned()
            } else {
                "Off".to_owned()
            };
        }
        match self.unit {
            DisplayUnit::Plain => format!("{value:.2}"),
            DisplayUnit::Percent => format!("{:.1} %", value * 100.0),
            DisplayUnit::Decibels => format!("{value:.1} dB"),
            DisplayUnit::Milliseconds => format!("{value:.1} ms"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutomationGesture {
    pub parameter_id: u32,
    pub value: f64,
}

#[derive(Debug, Eq, PartialEq)]
pub enum ModelError {
    Empty,
    TooManyParameters,
    InitialValueCount,
    InvalidSpec(&'static str),
    DuplicateId(u32),
}

impl Display for ModelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => formatter.write_str("the editor requires at least one parameter"),
            Self::TooManyParameters => write!(
                formatter,
                "the editor supports at most {MAX_PARAMETERS} parameters"
            ),
            Self::InitialValueCount => {
                formatter.write_str("editor initial values do not match parameter specifications")
            }
            Self::InvalidSpec(name) => write!(formatter, "invalid editor parameter spec: {name}"),
            Self::DuplicateId(id) => write!(formatter, "duplicate editor parameter ID {id}"),
        }
    }
}

impl Error for ModelError {}

pub struct EditorModel {
    title: &'static str,
    specs: &'static [ParameterSpec],
    values: Box<[AtomicU32]>,
    gestures: ArrayQueue<AutomationGesture>,
    overflow_mask: AtomicU64,
    focus: AtomicUsize,
    revision: AtomicU64,
    viewport: AtomicU64,
}

impl EditorModel {
    pub fn new(
        title: &'static str,
        specs: &'static [ParameterSpec],
        initial_values: &[f64],
    ) -> Result<Arc<Self>, ModelError> {
        validate_specs(specs, initial_values)?;
        let values = specs
            .iter()
            .zip(initial_values)
            .map(|(spec, value)| AtomicU32::new((spec.clamp(*value) as f32).to_bits()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Arc::new(Self {
            title,
            specs,
            values,
            gestures: ArrayQueue::new(AUTOMATION_QUEUE_CAPACITY),
            overflow_mask: AtomicU64::new(0),
            focus: AtomicUsize::new(0),
            revision: AtomicU64::new(1),
            viewport: AtomicU64::new(pack_viewport(640, 400)),
        }))
    }

    pub const fn title(&self) -> &'static str {
        self.title
    }

    pub const fn specs(&self) -> &'static [ParameterSpec] {
        self.specs
    }

    pub fn value(&self, index: usize) -> Option<f64> {
        self.values
            .get(index)
            .map(|value| f64::from(f32::from_bits(value.load(Ordering::Relaxed))))
    }

    pub fn value_by_id(&self, id: u32) -> Option<f64> {
        self.index_of(id).and_then(|index| self.value(index))
    }

    pub fn set_host_value(&self, id: u32, value: f64) -> bool {
        let Some(index) = self.index_of(id) else {
            return false;
        };
        self.store(index, value);
        true
    }

    pub fn set_editor_value(&self, index: usize, value: f64) -> Option<f64> {
        let spec = *self.specs.get(index)?;
        let value = spec.clamp(value);
        self.store(index, value);
        let gesture = AutomationGesture {
            parameter_id: spec.id,
            value,
        };
        if self.gestures.push(gesture).is_err() {
            self.mark_overflow(index);
        }
        Some(value)
    }

    pub fn adjust_editor_value(&self, index: usize, direction: f64, page: bool) -> Option<f64> {
        let spec = *self.specs.get(index)?;
        let current = self.value(index)?;
        let step = if page { spec.page_step } else { spec.step };
        self.set_editor_value(index, current + direction.signum() * step)
    }

    pub fn toggle_editor_value(&self, index: usize) -> Option<f64> {
        let spec = *self.specs.get(index)?;
        match spec.kind {
            ControlKind::Toggle => {
                let value = self.value(index)?;
                let next = if value >= (spec.minimum + spec.maximum) * 0.5 {
                    spec.minimum
                } else {
                    spec.maximum
                };
                self.set_editor_value(index, next)
            }
            _ => None,
        }
    }

    pub fn reset_editor_value(&self, index: usize) -> Option<f64> {
        let spec = *self.specs.get(index)?;
        self.set_editor_value(index, spec.default)
    }

    pub fn pop_gesture(&self) -> Option<AutomationGesture> {
        self.gestures.pop()
    }

    pub fn take_overflow_mask(&self) -> u64 {
        self.overflow_mask.swap(0, Ordering::AcqRel)
    }

    pub fn restore_overflow_mask(&self, mask: u64) {
        self.overflow_mask.fetch_or(mask, Ordering::Release);
    }

    pub fn overflow_gesture(&self, index: usize) -> Option<AutomationGesture> {
        Some(AutomationGesture {
            parameter_id: self.specs.get(index)?.id,
            value: self.value(index)?,
        })
    }

    pub fn focus(&self) -> usize {
        self.focus.load(Ordering::Acquire).min(self.specs.len() - 1)
    }

    pub fn set_focus(&self, index: usize) -> bool {
        if index >= self.specs.len() {
            return false;
        }
        self.focus.store(index, Ordering::Release);
        self.revision.fetch_add(1, Ordering::Relaxed);
        true
    }

    pub fn focus_next(&self, backwards: bool) -> usize {
        let current = self.focus();
        let next = if backwards {
            current.checked_sub(1).unwrap_or(self.specs.len() - 1)
        } else {
            (current + 1) % self.specs.len()
        };
        self.set_focus(next);
        next
    }

    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    pub fn set_viewport(&self, width: u32, height: u32) {
        self.viewport
            .store(pack_viewport(width, height), Ordering::Release);
        self.revision.fetch_add(1, Ordering::Relaxed);
    }

    pub fn viewport(&self) -> (u32, u32) {
        unpack_viewport(self.viewport.load(Ordering::Acquire))
    }

    pub fn index_of(&self, id: u32) -> Option<usize> {
        self.specs.iter().position(|spec| spec.id == id)
    }

    fn store(&self, index: usize, value: f64) {
        let Some(spec) = self.specs.get(index) else {
            return;
        };
        let Some(target) = self.values.get(index) else {
            return;
        };
        target.store((spec.clamp(value) as f32).to_bits(), Ordering::Relaxed);
        self.revision.fetch_add(1, Ordering::Release);
    }

    fn mark_overflow(&self, index: usize) {
        if index < MAX_PARAMETERS {
            self.overflow_mask
                .fetch_or(1_u64 << index, Ordering::Release);
        }
    }
}

fn validate_specs(specs: &[ParameterSpec], initial_values: &[f64]) -> Result<(), ModelError> {
    if specs.is_empty() {
        return Err(ModelError::Empty);
    }
    if specs.len() > MAX_PARAMETERS {
        return Err(ModelError::TooManyParameters);
    }
    if specs.len() != initial_values.len() {
        return Err(ModelError::InitialValueCount);
    }
    for (index, spec) in specs.iter().enumerate() {
        if spec.name.is_empty()
            || !spec.minimum.is_finite()
            || !spec.maximum.is_finite()
            || !spec.default.is_finite()
            || !spec.step.is_finite()
            || !spec.page_step.is_finite()
            || spec.minimum >= spec.maximum
            || !(spec.minimum..=spec.maximum).contains(&spec.default)
            || spec.step <= 0.0
            || spec.page_step <= 0.0
        {
            return Err(ModelError::InvalidSpec(spec.name));
        }
        if specs[..index].iter().any(|other| other.id == spec.id) {
            return Err(ModelError::DuplicateId(spec.id));
        }
        if let ControlKind::Choice(labels) = spec.kind {
            let count = spec.maximum - spec.minimum + 1.0;
            if labels.is_empty()
                || spec.minimum.fract() != 0.0
                || spec.maximum.fract() != 0.0
                || count > usize::MAX as f64
                || labels.len() != count as usize
            {
                return Err(ModelError::InvalidSpec(spec.name));
            }
        }
    }
    Ok(())
}

const fn pack_viewport(width: u32, height: u32) -> u64 {
    width as u64 | ((height as u64) << 32)
}

const fn unpack_viewport(value: u64) -> (u32, u32) {
    (value as u32, (value >> 32) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECS: &[ParameterSpec] = &[
        ParameterSpec {
            id: 0,
            name: "Bypass",
            minimum: 0.0,
            maximum: 1.0,
            default: 0.0,
            step: 1.0,
            page_step: 1.0,
            kind: ControlKind::Toggle,
            unit: DisplayUnit::Plain,
        },
        ParameterSpec {
            id: 1,
            name: "Mix",
            minimum: 0.0,
            maximum: 1.0,
            default: 1.0,
            step: 0.01,
            page_step: 0.1,
            kind: ControlKind::Continuous,
            unit: DisplayUnit::Percent,
        },
    ];

    #[test]
    fn host_updates_never_generate_automation() -> Result<(), ModelError> {
        let model = EditorModel::new("test", SPECS, &[0.0, 1.0])?;
        assert!(model.set_host_value(1, 0.25));
        assert_eq!(model.value_by_id(1), Some(0.25));
        assert_eq!(model.pop_gesture(), None);
        Ok(())
    }

    #[test]
    fn editor_updates_are_clamped_and_bounded() -> Result<(), ModelError> {
        let model = EditorModel::new("test", SPECS, &[0.0, 1.0])?;
        for index in 0..(AUTOMATION_QUEUE_CAPACITY + 8) {
            assert!(model.set_editor_value(1, index as f64).is_some());
        }
        assert_eq!(model.value(1), Some(1.0));
        assert_ne!(model.take_overflow_mask() & 0b10, 0);
        assert_eq!(
            model.overflow_gesture(1).map(|gesture| gesture.value),
            Some(1.0)
        );
        Ok(())
    }

    #[test]
    fn keyboard_focus_wraps_without_allocation_or_growth() -> Result<(), ModelError> {
        let model = EditorModel::new("test", SPECS, &[0.0, 1.0])?;
        assert_eq!(model.focus_next(true), 1);
        assert_eq!(model.focus_next(false), 0);
        assert_eq!(model.toggle_editor_value(0), Some(1.0));
        Ok(())
    }

    #[test]
    fn choice_labels_cover_the_exact_inclusive_integer_range() -> Result<(), ModelError> {
        const LABELS: &[&str] = &["Dry", "Gain", "Silence"];
        const CHOICE: &[ParameterSpec] = &[ParameterSpec {
            id: 9,
            name: "Fallback",
            minimum: 2.0,
            maximum: 4.0,
            default: 2.0,
            step: 1.0,
            page_step: 1.0,
            kind: ControlKind::Choice(LABELS),
            unit: DisplayUnit::Plain,
        }];
        let model = EditorModel::new("choice", CHOICE, &[4.0])?;
        assert_eq!(
            model.value(0).map(|value| CHOICE[0].display(value)),
            Some("Silence".to_owned())
        );
        Ok(())
    }
}
