//! Stable SDK capability and mobile lifecycle contracts.

use serde::{Deserialize, Serialize};

pub const SDK_CAPABILITIES_SCHEMA: &str = "denoize-sdk-capabilities-v1";
pub const MOBILE_LIFECYCLE_SCHEMA: &str = "denoize-mobile-lifecycle-v1";
pub const SDK_SCHEMA_VERSION: u32 = 1;

pub fn sdk_capabilities_json() -> &'static str {
    include_str!("../sdk/capabilities.json").trim()
}

pub fn mobile_lifecycle_json() -> &'static str {
    include_str!("../sdk/mobile-lifecycle.json").trim()
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum MobileLifecycleState {
    Backgrounded,
    Closed,
    Idle,
    Interrupted,
    Ready,
    RebuildRequired,
    Running,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct MobileAudioRoute {
    pub sample_rate: u32,
    pub buffer_frames: u32,
    pub channels: u32,
}

impl MobileAudioRoute {
    pub fn validate(self) -> Result<Self, String> {
        if !(1..=768_000).contains(&self.sample_rate) {
            return Err("mobile route sample_rate must be in 1..=768000".into());
        }
        if !(1..=1_048_576).contains(&self.buffer_frames) {
            return Err("mobile route buffer_frames must be in 1..=1048576".into());
        }
        if !(1..=32).contains(&self.channels) {
            return Err("mobile route channels must be in 1..=32".into());
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MobileLifecycleEvent {
    Background,
    Close,
    Configure(MobileAudioRoute),
    Interrupt,
    MemoryWarning,
    Resume(MobileAudioRoute),
    RouteChange(MobileAudioRoute),
    Start,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MobileLifecycle {
    state: MobileLifecycleState,
    route_generation: u64,
    route: Option<MobileAudioRoute>,
}

impl Default for MobileLifecycle {
    fn default() -> Self {
        Self {
            state: MobileLifecycleState::Idle,
            route_generation: 0,
            route: None,
        }
    }
}

impl MobileLifecycle {
    pub fn state(self) -> MobileLifecycleState {
        self.state
    }

    pub fn route_generation(self) -> u64 {
        self.route_generation
    }

    pub fn route(self) -> Option<MobileAudioRoute> {
        self.route
    }

    pub fn transition(&mut self, event: MobileLifecycleEvent) -> Result<(), String> {
        if self.state == MobileLifecycleState::Closed {
            return Err("mobile lifecycle is closed".into());
        }
        let (next, route, rebuild) = match event {
            MobileLifecycleEvent::Configure(route) if self.state == MobileLifecycleState::Idle => {
                (MobileLifecycleState::Ready, Some(route.validate()?), true)
            }
            MobileLifecycleEvent::Start if self.state == MobileLifecycleState::Ready => {
                (MobileLifecycleState::Running, self.route, false)
            }
            MobileLifecycleEvent::RouteChange(route)
                if matches!(
                    self.state,
                    MobileLifecycleState::Ready | MobileLifecycleState::Running
                ) =>
            {
                (MobileLifecycleState::Ready, Some(route.validate()?), true)
            }
            MobileLifecycleEvent::Interrupt
                if matches!(
                    self.state,
                    MobileLifecycleState::Ready | MobileLifecycleState::Running
                ) =>
            {
                (MobileLifecycleState::Interrupted, None, false)
            }
            MobileLifecycleEvent::Background
                if matches!(
                    self.state,
                    MobileLifecycleState::Ready | MobileLifecycleState::Running
                ) =>
            {
                (MobileLifecycleState::Backgrounded, None, false)
            }
            MobileLifecycleEvent::MemoryWarning
                if matches!(
                    self.state,
                    MobileLifecycleState::Ready | MobileLifecycleState::Running
                ) =>
            {
                (MobileLifecycleState::RebuildRequired, None, false)
            }
            MobileLifecycleEvent::Resume(route)
                if matches!(
                    self.state,
                    MobileLifecycleState::Backgrounded
                        | MobileLifecycleState::Interrupted
                        | MobileLifecycleState::RebuildRequired
                ) =>
            {
                (MobileLifecycleState::Ready, Some(route.validate()?), true)
            }
            MobileLifecycleEvent::Close => (MobileLifecycleState::Closed, None, false),
            _ => {
                return Err(format!(
                    "mobile lifecycle event is invalid from {:?}",
                    self.state
                ))
            }
        };
        let next_generation = if rebuild {
            self.route_generation
                .checked_add(1)
                .ok_or("mobile route generation overflow")?
        } else {
            self.route_generation
        };
        self.state = next;
        self.route = route;
        self.route_generation = next_generation;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(sample_rate: u32) -> MobileAudioRoute {
        MobileAudioRoute {
            sample_rate,
            buffer_frames: 256,
            channels: 2,
        }
    }

    #[test]
    fn route_changes_and_resume_always_advance_generation() {
        let mut lifecycle = MobileLifecycle::default();
        lifecycle
            .transition(MobileLifecycleEvent::Configure(route(48_000)))
            .unwrap_or_else(|error| panic!("configure lifecycle: {error}"));
        assert_eq!(lifecycle.route_generation(), 1);
        lifecycle
            .transition(MobileLifecycleEvent::Start)
            .unwrap_or_else(|error| panic!("start lifecycle: {error}"));
        lifecycle
            .transition(MobileLifecycleEvent::RouteChange(route(44_100)))
            .unwrap_or_else(|error| panic!("change route: {error}"));
        assert_eq!(lifecycle.state(), MobileLifecycleState::Ready);
        assert_eq!(lifecycle.route_generation(), 2);
        lifecycle
            .transition(MobileLifecycleEvent::Interrupt)
            .unwrap_or_else(|error| panic!("interrupt lifecycle: {error}"));
        assert!(lifecycle.route().is_none());
        lifecycle
            .transition(MobileLifecycleEvent::Resume(route(48_000)))
            .unwrap_or_else(|error| panic!("resume lifecycle: {error}"));
        assert_eq!(lifecycle.route_generation(), 3);
    }

    #[test]
    fn invalid_transitions_do_not_mutate_state() {
        let mut lifecycle = MobileLifecycle::default();
        assert!(lifecycle.transition(MobileLifecycleEvent::Start).is_err());
        assert_eq!(lifecycle, MobileLifecycle::default());
        assert!(lifecycle
            .transition(MobileLifecycleEvent::Configure(route(0)))
            .is_err());
        assert_eq!(lifecycle, MobileLifecycle::default());
        lifecycle
            .transition(MobileLifecycleEvent::Close)
            .unwrap_or_else(|error| panic!("close lifecycle: {error}"));
        assert!(lifecycle
            .transition(MobileLifecycleEvent::Configure(route(48_000)))
            .is_err());
    }
}
