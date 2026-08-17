//! Debug-only deterministic fault injection used by resilience tests.
//!
//! Release builds ignore the environment completely. Debug and test builds
//! accept one exact specification in `DENOIZE_INTERNAL_FAULT_V1`:
//! `v1|POINT|OCCURRENCE|error` or `v1|POINT|OCCURRENCE|exit`.

/// Environment variable understood by debug/test builds.
pub const ENVIRONMENT_VARIABLE: &str = "DENOIZE_INTERNAL_FAULT_V1";
/// Exit status used for an injected abrupt process exit.
pub const EXIT_CODE: i32 = 86;

#[cfg(debug_assertions)]
mod enabled {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use super::{ENVIRONMENT_VARIABLE, EXIT_CODE};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Action {
        Error,
        Exit,
    }

    #[derive(Debug, Eq, PartialEq)]
    struct Specification<'a> {
        point: &'a str,
        occurrence: u64,
        action: Action,
    }

    fn valid_point(point: &str) -> bool {
        !point.is_empty()
            && point.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'-')
            })
    }

    fn parse(raw: &str) -> Result<Specification<'_>, String> {
        let mut fields = raw.split('|');
        let version = fields.next();
        let point = fields.next();
        let occurrence = fields.next();
        let action = fields.next();
        if fields.next().is_some()
            || version != Some("v1")
            || point.is_none_or(|point| !valid_point(point))
        {
            return Err(format!(
                "invalid {ENVIRONMENT_VARIABLE}; expected v1|POINT|OCCURRENCE|error-or-exit"
            ));
        }
        let occurrence = occurrence
            .ok_or_else(|| {
                format!(
                    "invalid {ENVIRONMENT_VARIABLE}; expected v1|POINT|OCCURRENCE|error-or-exit"
                )
            })?
            .parse::<u64>()
            .map_err(|_| format!("invalid {ENVIRONMENT_VARIABLE} occurrence"))?;
        if occurrence == 0 {
            return Err(format!(
                "invalid {ENVIRONMENT_VARIABLE} occurrence: it must be positive"
            ));
        }
        let action = match action {
            Some("error") => Action::Error,
            Some("exit") => Action::Exit,
            _ => {
                return Err(format!(
                    "invalid {ENVIRONMENT_VARIABLE} action; expected error or exit"
                ));
            }
        };
        Ok(Specification {
            point: point.expect("validated fault point"),
            occurrence,
            action,
        })
    }

    static OCCURRENCES: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();

    pub(super) fn hit(point: &str) -> Result<(), String> {
        let Some(raw) = std::env::var_os(ENVIRONMENT_VARIABLE) else {
            return Ok(());
        };
        let raw = raw
            .to_str()
            .ok_or_else(|| format!("{ENVIRONMENT_VARIABLE} is not valid UTF-8"))?;
        let specification = parse(raw)?;
        if specification.point != point {
            return Ok(());
        }

        let mut occurrences = OCCURRENCES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|_| "fault-injection occurrence counter is poisoned".to_string())?;
        let occurrence = occurrences.entry(raw.to_owned()).or_default();
        *occurrence = occurrence
            .checked_add(1)
            .ok_or_else(|| "fault-injection occurrence counter overflowed".to_string())?;
        if *occurrence != specification.occurrence {
            return Ok(());
        }

        let message = format!(
            "injected fault at {point} occurrence {}",
            specification.occurrence
        );
        match specification.action {
            Action::Error => Err(message),
            Action::Exit => {
                eprintln!("denoize: {message}; exiting with status {EXIT_CODE}");
                std::process::exit(EXIT_CODE);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{parse, Action, Specification};

        #[test]
        fn specification_is_exact_and_versioned() {
            assert_eq!(
                parse("v1|atomic-output.after-stage-sync|2|exit").unwrap(),
                Specification {
                    point: "atomic-output.after-stage-sync",
                    occurrence: 2,
                    action: Action::Exit,
                }
            );
            for invalid in [
                "",
                "v2|point|1|exit",
                "v1||1|exit",
                "v1|UPPER|1|exit",
                "v1|point|0|exit",
                "v1|point|one|exit",
                "v1|point|1|panic",
                "v1|point|1|exit|extra",
            ] {
                assert!(parse(invalid).is_err(), "accepted {invalid:?}");
            }
        }
    }
}

/// Trigger a named deterministic fault in a debug/test build.
///
/// The function is deliberately a no-op when `debug_assertions` are disabled,
/// so production release binaries do not honor the internal test environment.
pub fn hit(point: &str) -> Result<(), String> {
    #[cfg(debug_assertions)]
    {
        return enabled::hit(point);
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = point;
        Ok(())
    }
}
