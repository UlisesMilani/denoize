use crate::layout::control_rect;
use crate::model::{ControlKind, EditorModel};
#[cfg(target_os = "linux")]
use accesskit::DeactivationHandler;
use accesskit::{
    Action, ActionData, ActionHandler, ActionRequest, ActivationHandler, Node, NodeId, Rect, Role,
    Toggled, Tree, TreeId, TreeUpdate,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

const ROOT_ID: NodeId = NodeId(1);
const CONTROL_ID_BASE: u64 = 100;

pub type FlushCallback = Arc<dyn Fn() + Send + Sync + 'static>;

pub fn build_tree(model: &EditorModel, include_tree: bool) -> TreeUpdate {
    let (width, height) = model.viewport();
    let control_ids = (0..model.specs().len())
        .map(control_node_id)
        .collect::<Vec<_>>();
    let mut root = Node::new(Role::Window);
    root.set_label(model.title());
    root.set_bounds(Rect::new(0.0, 0.0, f64::from(width), f64::from(height)));
    root.set_children(control_ids);

    let mut nodes = Vec::with_capacity(model.specs().len() + 1);
    nodes.push((ROOT_ID, root));
    for (index, spec) in model.specs().iter().enumerate() {
        let value = model.value(index).unwrap_or(spec.default);
        let bounds = control_rect(index, model.specs().len(), width, height);
        let mut node = match spec.kind {
            ControlKind::Toggle => Node::new(Role::CheckBox),
            ControlKind::Continuous | ControlKind::Choice(_) => Node::new(Role::Slider),
        };
        node.set_label(spec.name);
        node.set_value(spec.display(value));
        node.set_bounds(Rect::new(
            bounds.x,
            bounds.y,
            bounds.x + bounds.width,
            bounds.y + bounds.height,
        ));
        node.add_action(Action::Focus);
        match spec.kind {
            ControlKind::Toggle => {
                node.set_toggled(if value >= (spec.minimum + spec.maximum) * 0.5 {
                    Toggled::True
                } else {
                    Toggled::False
                });
                node.add_action(Action::Click);
                node.add_action(Action::SetValue);
            }
            ControlKind::Continuous | ControlKind::Choice(_) => {
                node.set_numeric_value(value);
                node.set_min_numeric_value(spec.minimum);
                node.set_max_numeric_value(spec.maximum);
                node.set_numeric_value_step(spec.step);
                node.set_numeric_value_jump(spec.page_step);
                node.add_action(Action::Decrement);
                node.add_action(Action::Increment);
                node.add_action(Action::SetValue);
            }
        }
        nodes.push((control_node_id(index), node));
    }

    TreeUpdate {
        nodes,
        tree: include_tree.then(|| Tree::new(ROOT_ID)),
        tree_id: TreeId::ROOT,
        focus: control_node_id(model.focus()),
    }
}

fn control_node_id(index: usize) -> NodeId {
    NodeId(CONTROL_ID_BASE + index as u64)
}

fn control_index(id: NodeId, count: usize) -> Option<usize> {
    let raw = id.0.checked_sub(CONTROL_ID_BASE)?;
    let index = usize::try_from(raw).ok()?;
    (index < count).then_some(index)
}

pub struct EditorActivationHandler {
    model: Arc<EditorModel>,
}

impl EditorActivationHandler {
    pub fn new(model: Arc<EditorModel>) -> Self {
        Self { model }
    }
}

impl ActivationHandler for EditorActivationHandler {
    fn request_initial_tree(&mut self) -> Option<TreeUpdate> {
        Some(build_tree(&self.model, true))
    }
}

pub struct EditorActionHandler {
    model: Arc<EditorModel>,
    flush: FlushCallback,
    dirty: Arc<AtomicBool>,
}

impl EditorActionHandler {
    pub fn new(model: Arc<EditorModel>, flush: FlushCallback, dirty: Arc<AtomicBool>) -> Self {
        Self {
            model,
            flush,
            dirty,
        }
    }

    fn set_value(&self, index: usize, value: f64) {
        if self.model.set_editor_value(index, value).is_some() {
            self.notify_changed();
        }
    }

    fn notify_changed(&self) {
        (self.flush)();
        self.dirty.store(true, Ordering::Release);
    }
}

impl ActionHandler for EditorActionHandler {
    fn do_action(&mut self, request: ActionRequest) {
        let Some(index) = control_index(request.target_node, self.model.specs().len()) else {
            return;
        };
        match request.action {
            Action::Focus => {
                self.model.set_focus(index);
                self.dirty.store(true, Ordering::Release);
            }
            Action::Click => {
                if self.model.toggle_editor_value(index).is_some() {
                    self.notify_changed();
                }
            }
            Action::Increment => {
                if self.model.adjust_editor_value(index, 1.0, false).is_some() {
                    self.notify_changed();
                }
            }
            Action::Decrement => {
                if self.model.adjust_editor_value(index, -1.0, false).is_some() {
                    self.notify_changed();
                }
            }
            Action::SetValue => {
                let value = match request.data {
                    Some(ActionData::NumericValue(value)) => Some(value),
                    Some(ActionData::Value(value)) => value.parse::<f64>().ok(),
                    _ => None,
                };
                if let Some(value) = value {
                    self.set_value(index, value);
                }
            }
            _ => {}
        }
    }
}

#[cfg(target_os = "linux")]
pub struct EditorDeactivationHandler;

#[cfg(target_os = "linux")]
impl DeactivationHandler for EditorDeactivationHandler {
    fn deactivate_accessibility(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DisplayUnit, ModelError, ParameterSpec};
    use std::sync::atomic::AtomicUsize;

    const SPECS: &[ParameterSpec] = &[ParameterSpec {
        id: 7,
        name: "Mix",
        minimum: 0.0,
        maximum: 1.0,
        default: 1.0,
        step: 0.01,
        page_step: 0.1,
        kind: ControlKind::Continuous,
        unit: DisplayUnit::Percent,
    }];

    #[test]
    fn tree_exposes_value_range_actions_and_focus() -> Result<(), ModelError> {
        let model = EditorModel::new("denoize", SPECS, &[0.75])?;
        let tree = build_tree(&model, true);
        assert_eq!(tree.tree.as_ref().map(|value| value.root), Some(ROOT_ID));
        assert_eq!(tree.focus, control_node_id(0));
        let control = &tree.nodes[1].1;
        assert_eq!(control.role(), Role::Slider);
        assert_eq!(control.numeric_value(), Some(0.75));
        assert!(control.supports_action(Action::SetValue));
        Ok(())
    }

    #[test]
    fn assistive_action_queues_one_bounded_gesture() -> Result<(), ModelError> {
        let model = EditorModel::new("denoize", SPECS, &[0.75])?;
        let flushes = Arc::new(AtomicUsize::new(0));
        let callback: FlushCallback = {
            let flushes = Arc::clone(&flushes);
            Arc::new(move || {
                flushes.fetch_add(1, Ordering::Relaxed);
            })
        };
        let mut handler = EditorActionHandler::new(
            Arc::clone(&model),
            callback,
            Arc::new(AtomicBool::new(false)),
        );
        handler.do_action(ActionRequest {
            action: Action::SetValue,
            target_tree: TreeId::ROOT,
            target_node: control_node_id(0),
            data: Some(ActionData::NumericValue(0.25)),
        });
        assert_eq!(model.value(0), Some(0.25));
        assert_eq!(model.pop_gesture().map(|gesture| gesture.value), Some(0.25));
        assert_eq!(flushes.load(Ordering::Relaxed), 1);
        Ok(())
    }
}
