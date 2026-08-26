//! Hash-chained append-only commands, undo/redo, and crash-safe checkpoints.

use super::*;
use crate::{AtomicOutput, CommitMode};
use serde::{Deserialize, Serialize};
use std::io::Write as _;
use std::path::Path;

pub const PROJECT_V2_JOURNAL_SCHEMA: &str = "denoize-project-v2-journal-entry-v1";
pub const PROJECT_V2_JOURNAL_INSPECTION_SCHEMA: &str = "denoize-project-v2-journal-inspection-v1";
pub const PROJECT_V2_CHECKPOINT_SCHEMA: &str = "denoize-project-v2-checkpoint-v1";
const JOURNAL_COMMAND_DOMAIN: &[u8] = b"denoize-project-v2-command-digest-v1";
const JOURNAL_PREFIX_DOMAIN: &[u8] = b"denoize-project-v2-journal-prefix-v1";
const MAX_JOURNAL_BYTES: u64 = 64 * 1024 * 1024;
const MAX_JOURNAL_ENTRIES: usize = 200_000;
const MAX_JOURNAL_LINE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "owner", rename_all = "kebab-case")]
pub enum ProjectV2EffectChainOwner {
    Track { graph_id: String, track_id: String },
    Bus { graph_id: String, bus_id: String },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "command", rename_all = "kebab-case")]
pub enum ProjectV2Command {
    InsertClip {
        graph_id: String,
        clip: ProjectV2Clip,
        transitions: Vec<ProjectV2Transition>,
    },
    RemoveClip {
        graph_id: String,
        clip: ProjectV2Clip,
        transitions: Vec<ProjectV2Transition>,
    },
    MoveClip {
        graph_id: String,
        clip_id: String,
        from_track_id: String,
        from_start: ProjectV2Time,
        to_track_id: String,
        to_start: ProjectV2Time,
    },
    SplitClip {
        graph_id: String,
        original: ProjectV2Clip,
        left: ProjectV2Clip,
        right: ProjectV2Clip,
    },
    JoinClips {
        graph_id: String,
        left: ProjectV2Clip,
        right: ProjectV2Clip,
        joined: ProjectV2Clip,
    },
    ReplaceEffectChain {
        owner: ProjectV2EffectChainOwner,
        before: Vec<ProjectV2EffectReference>,
        after: Vec<ProjectV2EffectReference>,
    },
    ReplaceEffectRevision {
        before: ProjectV2EffectNode,
        after: ProjectV2EffectNode,
    },
    SelectEffectRevision {
        before: ProjectV2EffectReference,
        after: ProjectV2EffectReference,
    },
}

impl ProjectV2Command {
    pub fn digest(&self) -> Result<Digest, String> {
        digest_json(JOURNAL_COMMAND_DOMAIN, self, "project v2 journal command")
    }

    pub fn inverse(&self) -> Result<Self, String> {
        Ok(match self {
            Self::InsertClip {
                graph_id,
                clip,
                transitions,
            } => Self::RemoveClip {
                graph_id: graph_id.clone(),
                clip: clip.clone(),
                transitions: transitions.clone(),
            },
            Self::RemoveClip {
                graph_id,
                clip,
                transitions,
            } => Self::InsertClip {
                graph_id: graph_id.clone(),
                clip: clip.clone(),
                transitions: transitions.clone(),
            },
            Self::MoveClip {
                graph_id,
                clip_id,
                from_track_id,
                from_start,
                to_track_id,
                to_start,
            } => Self::MoveClip {
                graph_id: graph_id.clone(),
                clip_id: clip_id.clone(),
                from_track_id: to_track_id.clone(),
                from_start: *to_start,
                to_track_id: from_track_id.clone(),
                to_start: *from_start,
            },
            Self::SplitClip {
                graph_id,
                original,
                left,
                right,
            } => Self::JoinClips {
                graph_id: graph_id.clone(),
                left: left.clone(),
                right: right.clone(),
                joined: original.clone(),
            },
            Self::JoinClips {
                graph_id,
                left,
                right,
                joined,
            } => Self::SplitClip {
                graph_id: graph_id.clone(),
                original: joined.clone(),
                left: left.clone(),
                right: right.clone(),
            },
            Self::ReplaceEffectChain {
                owner,
                before,
                after,
            } => Self::ReplaceEffectChain {
                owner: owner.clone(),
                before: after.clone(),
                after: before.clone(),
            },
            Self::ReplaceEffectRevision { before, after } => Self::SelectEffectRevision {
                before: after.reference()?,
                after: before.reference()?,
            },
            Self::SelectEffectRevision { before, after } => Self::SelectEffectRevision {
                before: after.clone(),
                after: before.clone(),
            },
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2JournalEntry {
    pub schema: String,
    pub schema_version: u32,
    pub sequence: u64,
    pub parent_digest: Digest,
    pub command_digest: Digest,
    pub result_digest: Digest,
    pub command: ProjectV2Command,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2JournalReadReport {
    pub schema: String,
    pub schema_version: u32,
    pub entries: Vec<ProjectV2JournalEntry>,
    pub truncated_tail_bytes: u64,
    pub prefix_digest: Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectV2Checkpoint {
    pub schema: String,
    pub schema_version: u32,
    pub snapshot: ProjectV2Manifest,
    pub snapshot_digest: Digest,
    pub prior_root_digest: Digest,
    pub journal_prefix_digest: Digest,
    pub compacted_entries: u64,
}

/// Apply one closed command to an exact parent root and return its journal
/// entry. Historical effect revisions remain in the manifest; only references
/// move to a newer immutable revision.
pub fn apply_project_v2_command(
    manifest: &ProjectV2Manifest,
    sequence: u64,
    command: ProjectV2Command,
) -> Result<(ProjectV2Manifest, ProjectV2JournalEntry), String> {
    manifest.validate()?;
    if sequence == 0 || sequence > MAX_JSON_SAFE_INTEGER {
        return Err("project v2 journal sequence is unsupported".into());
    }
    validate_command(&command)?;
    let parent_digest = manifest.digest()?;
    let command_digest = command.digest()?;
    let mut next = manifest.clone();
    mutate_manifest(&mut next, &command)?;
    next.root_revision = next
        .root_revision
        .checked_add(1)
        .ok_or_else(|| "project v2 root revision overflows".to_string())?;
    next.parent_digest = Some(parent_digest);
    next.canonicalize();
    next.validate()?;
    let result_digest = next.digest()?;
    let entry = ProjectV2JournalEntry {
        schema: PROJECT_V2_JOURNAL_SCHEMA.into(),
        schema_version: 1,
        sequence,
        parent_digest,
        command_digest,
        result_digest,
        command,
    };
    validate_entry(&entry)?;
    Ok((next, entry))
}

pub fn undo_project_v2_command(
    manifest: &ProjectV2Manifest,
    sequence: u64,
    entry: &ProjectV2JournalEntry,
) -> Result<(ProjectV2Manifest, ProjectV2JournalEntry), String> {
    validate_entry(entry)?;
    if manifest.digest()? != entry.result_digest {
        return Err("project v2 undo root does not match the journal result".into());
    }
    apply_project_v2_command(manifest, sequence, inverse_for_manifest(entry, manifest)?)
}

pub fn replay_project_v2_journal(
    initial: &ProjectV2Manifest,
    entries: &[ProjectV2JournalEntry],
) -> Result<ProjectV2Manifest, String> {
    initial.validate()?;
    let mut manifest = initial.clone();
    let mut expected_sequence = entries.first().map_or(1, |entry| entry.sequence);
    for entry in entries {
        validate_entry(entry)?;
        if entry.sequence != expected_sequence || entry.parent_digest != manifest.digest()? {
            return Err(format!(
                "project v2 journal chain breaks at sequence {}",
                entry.sequence
            ));
        }
        let (next, reconstructed) =
            apply_project_v2_command(&manifest, entry.sequence, entry.command.clone())?;
        if reconstructed != *entry {
            return Err(format!(
                "project v2 journal entry {} does not reproduce its recorded root",
                entry.sequence
            ));
        }
        manifest = next;
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or_else(|| "project v2 journal sequence overflows".to_string())?;
    }
    Ok(manifest)
}

/// Read bounded NDJSON. A final unterminated fragment is reported and ignored;
/// corruption in any newline-terminated record fails closed.
pub fn read_project_v2_journal(
    path: impl AsRef<Path>,
) -> Result<ProjectV2JournalReadReport, String> {
    let path = path.as_ref();
    let (mut file, length) = crate::input::open_regular_file(path, "project v2 journal")?;
    if length > MAX_JOURNAL_BYTES {
        return Err(format!(
            "project v2 journal {} exceeds {MAX_JOURNAL_BYTES} bytes",
            path.display()
        ));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length as usize)
        .map_err(|_| "unable to reserve project v2 journal bytes".to_string())?;
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("read project v2 journal {}: {error}", path.display()))?;
    if bytes.len() as u64 != length {
        return Err("project v2 journal changed while reading".into());
    }
    let complete_length = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |index| index + 1);
    let truncated_tail_bytes = (bytes.len() - complete_length) as u64;
    let mut entries = Vec::new();
    for (index, line) in bytes[..complete_length]
        .split(|byte| *byte == b'\n')
        .enumerate()
    {
        if line.is_empty() {
            continue;
        }
        if line.len() > MAX_JOURNAL_LINE_BYTES {
            return Err(format!(
                "project v2 journal line {} exceeds its limit",
                index + 1
            ));
        }
        let entry: ProjectV2JournalEntry = serde_json::from_slice(line)
            .map_err(|error| format!("parse project v2 journal line {}: {error}", index + 1))?;
        validate_entry(&entry)?;
        entries.push(entry);
        if entries.len() > MAX_JOURNAL_ENTRIES {
            return Err(format!(
                "project v2 journal exceeds {MAX_JOURNAL_ENTRIES} entries"
            ));
        }
    }
    validate_entry_chain(&entries)?;
    Ok(ProjectV2JournalReadReport {
        schema: PROJECT_V2_JOURNAL_INSPECTION_SCHEMA.into(),
        schema_version: 1,
        prefix_digest: journal_prefix_digest(&entries)?,
        entries,
        truncated_tail_bytes,
    })
}

/// Atomically replace the journal with its previous verified prefix plus one
/// new entry. This preserves append-only logical history and avoids publishing
/// a partial record after a crash.
pub fn append_project_v2_journal(
    path: impl AsRef<Path>,
    entry: &ProjectV2JournalEntry,
) -> Result<(), String> {
    let path = path.as_ref();
    validate_entry(entry)?;
    let existing = if path.exists() {
        let report = read_project_v2_journal(path)?;
        if report.truncated_tail_bytes != 0 {
            return Err(
                "project v2 journal has a truncated tail; checkpoint or repair it before appending"
                    .into(),
            );
        }
        report.entries
    } else {
        Vec::new()
    };
    if let Some(previous) = existing.last() {
        if entry.sequence != previous.sequence + 1 || entry.parent_digest != previous.result_digest
        {
            return Err("project v2 journal append does not extend the current hash chain".into());
        }
    } else if entry.sequence != 1 {
        return Err("the first project v2 journal entry must have sequence 1".into());
    }
    if existing.len() >= MAX_JOURNAL_ENTRIES {
        return Err("project v2 journal entry limit reached".into());
    }
    let mut transaction = AtomicOutput::new(path)?;
    for old in &existing {
        write_journal_line(transaction.file_mut(), old)?;
    }
    write_journal_line(transaction.file_mut(), entry)?;
    let size = transaction
        .file_mut()
        .metadata()
        .map_err(|error| format!("inspect staged project v2 journal: {error}"))?
        .len();
    if size > MAX_JOURNAL_BYTES {
        return Err("project v2 journal byte limit reached".into());
    }
    transaction.commit(if path.exists() {
        CommitMode::Replace
    } else {
        CommitMode::NoClobber
    })
}

pub fn create_project_v2_checkpoint(
    prior_root_digest: Digest,
    snapshot: ProjectV2Manifest,
    compacted_entries: &[ProjectV2JournalEntry],
) -> Result<ProjectV2Checkpoint, String> {
    snapshot.validate()?;
    validate_entry_chain(compacted_entries)?;
    let snapshot_digest = snapshot.digest()?;
    if compacted_entries
        .last()
        .is_some_and(|entry| entry.result_digest != snapshot_digest)
    {
        return Err("project v2 checkpoint snapshot does not match the journal prefix".into());
    }
    if compacted_entries
        .first()
        .is_some_and(|entry| entry.parent_digest != prior_root_digest)
    {
        return Err("project v2 checkpoint prior root does not match the journal prefix".into());
    }
    let checkpoint = ProjectV2Checkpoint {
        schema: PROJECT_V2_CHECKPOINT_SCHEMA.into(),
        schema_version: 1,
        snapshot,
        snapshot_digest,
        prior_root_digest,
        journal_prefix_digest: journal_prefix_digest(compacted_entries)?,
        compacted_entries: compacted_entries.len() as u64,
    };
    verify_project_v2_checkpoint(&checkpoint, compacted_entries)?;
    Ok(checkpoint)
}

/// Verify that a checkpoint is bound to the exact journal prefix it claims to
/// compact. This must be performed before accepting or publishing a checkpoint;
/// validating the snapshot digest alone cannot authenticate the prefix count,
/// endpoints, or ordered entries.
pub fn verify_project_v2_checkpoint(
    checkpoint: &ProjectV2Checkpoint,
    compacted_entries: &[ProjectV2JournalEntry],
) -> Result<(), String> {
    validate_checkpoint(checkpoint)?;
    validate_entry_chain(compacted_entries)?;
    if checkpoint.compacted_entries != compacted_entries.len() as u64 {
        return Err("project v2 checkpoint compacted-entry count does not match".into());
    }
    if checkpoint.journal_prefix_digest != journal_prefix_digest(compacted_entries)? {
        return Err("project v2 checkpoint journal-prefix digest does not match".into());
    }
    if let Some(first) = compacted_entries.first() {
        if first.parent_digest != checkpoint.prior_root_digest {
            return Err(
                "project v2 checkpoint prior root does not match the journal prefix".into(),
            );
        }
        if compacted_entries
            .last()
            .is_some_and(|entry| entry.result_digest != checkpoint.snapshot_digest)
        {
            return Err("project v2 checkpoint snapshot does not match the journal prefix".into());
        }
    } else if checkpoint.prior_root_digest != checkpoint.snapshot_digest {
        return Err("an empty project v2 checkpoint must preserve its prior root".into());
    }
    Ok(())
}

pub fn write_project_v2_checkpoint(
    path: impl AsRef<Path>,
    checkpoint: &ProjectV2Checkpoint,
    compacted_entries: &[ProjectV2JournalEntry],
    mode: CommitMode,
) -> Result<(), String> {
    verify_project_v2_checkpoint(checkpoint, compacted_entries)?;
    let encoded = serde_json::to_vec_pretty(checkpoint)
        .map_err(|error| format!("serialize project v2 checkpoint: {error}"))?;
    if encoded.len() as u64 >= MAX_PROJECT_V2_JSON_BYTES {
        return Err("project v2 checkpoint exceeds its 32 MiB limit".into());
    }
    let mut transaction = AtomicOutput::new(path)?;
    transaction
        .file_mut()
        .write_all(&encoded)
        .map_err(|error| format!("write project v2 checkpoint: {error}"))?;
    transaction.commit(mode)
}

fn mutate_manifest(
    manifest: &mut ProjectV2Manifest,
    command: &ProjectV2Command,
) -> Result<(), String> {
    match command {
        ProjectV2Command::InsertClip {
            graph_id,
            clip,
            transitions,
        } => {
            let graph = graph_mut(manifest, graph_id)?;
            if graph.clips.iter().any(|old| old.id == clip.id) {
                return Err(format!("project v2 clip {} already exists", clip.id));
            }
            graph.clips.push(clip.clone());
            graph.transitions.extend(transitions.iter().cloned());
            graph.revision = checked_revision(graph.revision)?;
        }
        ProjectV2Command::RemoveClip {
            graph_id,
            clip,
            transitions,
        } => {
            let graph = graph_mut(manifest, graph_id)?;
            remove_exact(&mut graph.clips, clip, "project v2 clip")?;
            for transition in transitions {
                remove_exact(&mut graph.transitions, transition, "project v2 transition")?;
            }
            if graph
                .transitions
                .iter()
                .any(|edge| edge.from_clip_id == clip.id || edge.to_clip_id == clip.id)
            {
                return Err("project v2 remove-clip command omitted a connected transition".into());
            }
            graph.revision = checked_revision(graph.revision)?;
        }
        ProjectV2Command::MoveClip {
            graph_id,
            clip_id,
            from_track_id,
            from_start,
            to_track_id,
            to_start,
        } => {
            let graph = graph_mut(manifest, graph_id)?;
            if graph
                .transitions
                .iter()
                .any(|edge| edge.from_clip_id == *clip_id || edge.to_clip_id == *clip_id)
            {
                return Err(
                    "project v2 move-clip cannot leave transition geometry implicit".into(),
                );
            }
            let clip = graph
                .clips
                .iter_mut()
                .find(|clip| clip.id == *clip_id)
                .ok_or_else(|| format!("project v2 clip {clip_id} is missing"))?;
            if clip.track_id != *from_track_id || clip.timeline_start != *from_start {
                return Err("project v2 move-clip before-state does not match".into());
            }
            clip.track_id = to_track_id.clone();
            clip.timeline_start = *to_start;
            clip.revision = checked_revision(clip.revision)?;
            graph.revision = checked_revision(graph.revision)?;
        }
        ProjectV2Command::SplitClip {
            graph_id,
            original,
            left,
            right,
        } => {
            let graph = graph_mut(manifest, graph_id)?;
            if graph
                .transitions
                .iter()
                .any(|edge| edge.from_clip_id == original.id || edge.to_clip_id == original.id)
            {
                return Err(
                    "project v2 split-clip cannot leave transition geometry implicit".into(),
                );
            }
            validate_split(original, left, right, graph.sample_rate)?;
            remove_exact(&mut graph.clips, original, "project v2 split source clip")?;
            graph.clips.extend([left.clone(), right.clone()]);
            graph.revision = checked_revision(graph.revision)?;
        }
        ProjectV2Command::JoinClips {
            graph_id,
            left,
            right,
            joined,
        } => {
            let graph = graph_mut(manifest, graph_id)?;
            validate_split(joined, left, right, graph.sample_rate)?;
            if graph.transitions.iter().any(|edge| {
                [left.id.as_str(), right.id.as_str()].contains(&edge.from_clip_id.as_str())
                    || [left.id.as_str(), right.id.as_str()].contains(&edge.to_clip_id.as_str())
            }) {
                return Err(
                    "project v2 join-clips cannot leave transition geometry implicit".into(),
                );
            }
            remove_exact(&mut graph.clips, left, "project v2 left join clip")?;
            remove_exact(&mut graph.clips, right, "project v2 right join clip")?;
            graph.clips.push(joined.clone());
            graph.revision = checked_revision(graph.revision)?;
        }
        ProjectV2Command::ReplaceEffectChain {
            owner,
            before,
            after,
        } => {
            if let ProjectV2EffectChainOwner::Track { graph_id, track_id } = owner {
                let graph = graph_mut(manifest, graph_id)?;
                let track = graph
                    .tracks
                    .iter_mut()
                    .find(|track| track.id == *track_id)
                    .ok_or_else(|| format!("project v2 track {track_id} is missing"))?;
                replace_chain(&mut track.effect_chain, before, after)?;
                track.revision = checked_revision(track.revision)?;
                graph.revision = checked_revision(graph.revision)?;
            } else if let ProjectV2EffectChainOwner::Bus { graph_id, bus_id } = owner {
                let graph = graph_mut(manifest, graph_id)?;
                let bus = graph
                    .buses
                    .iter_mut()
                    .find(|bus| bus.id == *bus_id)
                    .ok_or_else(|| format!("project v2 bus {bus_id} is missing"))?;
                replace_chain(&mut bus.effect_chain, before, after)?;
                bus.revision = checked_revision(bus.revision)?;
                graph.revision = checked_revision(graph.revision)?;
            }
        }
        ProjectV2Command::ReplaceEffectRevision { before, after } => {
            if after.id != before.id || after.revision != before.revision + 1 {
                return Err("project v2 replacement effect must append the next revision".into());
            }
            let current = manifest
                .effects
                .iter()
                .find(|effect| effect.id == before.id && effect.revision == before.revision)
                .ok_or("project v2 replacement effect before-revision is missing")?;
            if current != before {
                return Err("project v2 replacement effect before-state does not match".into());
            }
            if manifest
                .effects
                .iter()
                .any(|effect| effect.id == after.id && effect.revision == after.revision)
            {
                return Err("project v2 replacement effect revision already exists".into());
            }
            let old = before.reference()?;
            let new = after.reference()?;
            manifest.effects.push(after.clone());
            replace_all_effect_references(manifest, &[old], &[new])?;
        }
        ProjectV2Command::SelectEffectRevision { before, after } => {
            manifest.effect(before)?;
            manifest.effect(after)?;
            if before.id != after.id {
                return Err("project v2 selected effect revisions must share an ID".into());
            }
            if count_effect_references(manifest, after) != 0 {
                return Err(
                    "project v2 selected destination revision is already referenced; use an owner-scoped effect-chain command"
                        .into(),
                );
            }
            replace_all_effect_references(
                manifest,
                std::slice::from_ref(before),
                std::slice::from_ref(after),
            )?;
        }
    }
    Ok(())
}

fn count_effect_references(
    manifest: &ProjectV2Manifest,
    needle: &ProjectV2EffectReference,
) -> usize {
    manifest
        .graphs
        .iter()
        .flat_map(|graph| {
            graph
                .tracks
                .iter()
                .flat_map(|track| &track.effect_chain)
                .chain(graph.buses.iter().flat_map(|bus| &bus.effect_chain))
        })
        .filter(|reference| *reference == needle)
        .count()
}

fn inverse_for_manifest(
    entry: &ProjectV2JournalEntry,
    manifest: &ProjectV2Manifest,
) -> Result<ProjectV2Command, String> {
    let command = entry.command.inverse()?;
    validate_command(&command)?;
    let _ = manifest;
    Ok(command)
}

fn replace_all_effect_references(
    manifest: &mut ProjectV2Manifest,
    before: &[ProjectV2EffectReference],
    after: &[ProjectV2EffectReference],
) -> Result<(), String> {
    if before.len() != 1 || after.len() != 1 {
        return Err("global project v2 effect replacement requires one exact revision".into());
    }
    let mut replacements = 0usize;
    for graph in &mut manifest.graphs {
        let mut changed = false;
        for track in &mut graph.tracks {
            for reference in &mut track.effect_chain {
                if reference == &before[0] {
                    *reference = after[0].clone();
                    replacements += 1;
                    track.revision = checked_revision(track.revision)?;
                    changed = true;
                }
            }
        }
        for bus in &mut graph.buses {
            for reference in &mut bus.effect_chain {
                if reference == &before[0] {
                    *reference = after[0].clone();
                    replacements += 1;
                    bus.revision = checked_revision(bus.revision)?;
                    changed = true;
                }
            }
        }
        if changed {
            graph.revision = checked_revision(graph.revision)?;
        }
    }
    if replacements == 0 {
        return Err("project v2 effect revision is not referenced by any chain".into());
    }
    Ok(())
}

fn replace_chain(
    target: &mut Vec<ProjectV2EffectReference>,
    before: &[ProjectV2EffectReference],
    after: &[ProjectV2EffectReference],
) -> Result<(), String> {
    if target != before {
        return Err("project v2 effect-chain before-state does not match".into());
    }
    *target = after.to_vec();
    Ok(())
}

fn validate_split(
    original: &ProjectV2Clip,
    left: &ProjectV2Clip,
    right: &ProjectV2Clip,
    sample_rate: u32,
) -> Result<(), String> {
    if original.id == left.id
        || original.id == right.id
        || left.id == right.id
        || original.track_id != left.track_id
        || original.track_id != right.track_id
        || original.source != left.source
        || original.source != right.source
        || original.channel_map != left.channel_map
        || original.channel_map != right.channel_map
        || original.gain != left.gain
        || original.gain != right.gain
    {
        return Err("project v2 split/join clip identities or immutable properties differ".into());
    }
    let original_start = original.timeline_start.frames_at(sample_rate)?;
    let left_start = left.timeline_start.frames_at(sample_rate)?;
    let right_start = right.timeline_start.frames_at(sample_rate)?;
    let left_duration = left.duration.frames_at(sample_rate)?;
    let right_duration = right.duration.frames_at(sample_rate)?;
    if left_start != original_start
        || right_start != left_start + left_duration
        || left_duration + right_duration != original.duration.frames_at(sample_rate)?
    {
        return Err("project v2 split/join timeline ranges are not contiguous and exact".into());
    }
    let source_rate = original.source_start.rate;
    let original_source_duration = original.duration.frames_at(source_rate)?;
    let left_source_duration = left.duration.frames_at(source_rate)?;
    let right_source_duration = right.duration.frames_at(source_rate)?;
    if left.source_start != original.source_start
        || right.source_start.frames_at(source_rate)?
            != original.source_start.frames_at(source_rate)? + left_source_duration
        || left_source_duration
            .checked_add(right_source_duration)
            .is_none_or(|duration| duration != original_source_duration)
    {
        return Err("project v2 split/join source ranges are not contiguous and exact".into());
    }
    Ok(())
}

fn validate_command(command: &ProjectV2Command) -> Result<(), String> {
    let encoded = serde_json::to_vec(command)
        .map_err(|error| format!("serialize project v2 command: {error}"))?;
    if encoded.len() > MAX_JOURNAL_LINE_BYTES / 2 {
        return Err("project v2 command exceeds its bounded journal allowance".into());
    }
    match command {
        ProjectV2Command::InsertClip {
            graph_id,
            clip,
            transitions,
        }
        | ProjectV2Command::RemoveClip {
            graph_id,
            clip,
            transitions,
        } => {
            validate_identifier("project v2 command graph ID", graph_id)?;
            validate_identifier("project v2 command clip ID", &clip.id)?;
            if transitions.len() > 256 {
                return Err("project v2 clip command has too many transitions".into());
            }
        }
        ProjectV2Command::MoveClip {
            graph_id,
            clip_id,
            from_track_id,
            from_start,
            to_track_id,
            to_start,
        } => {
            for (context, value) in [
                ("graph", graph_id),
                ("clip", clip_id),
                ("source track", from_track_id),
                ("destination track", to_track_id),
            ] {
                validate_identifier(&format!("project v2 {context} ID"), value)?;
            }
            from_start.validate("project v2 move source time")?;
            to_start.validate("project v2 move destination time")?;
        }
        ProjectV2Command::SplitClip { graph_id, .. }
        | ProjectV2Command::JoinClips { graph_id, .. } => {
            validate_identifier("project v2 command graph ID", graph_id)?
        }
        ProjectV2Command::ReplaceEffectChain {
            owner,
            before,
            after,
        } => {
            if before == after {
                return Err("project v2 effect-chain command must change state".into());
            }
            if before.len() > 256 || after.len() > 256 {
                return Err("project v2 effect-chain command exceeds 256 nodes".into());
            }
            match owner {
                ProjectV2EffectChainOwner::Track { graph_id, track_id } => {
                    validate_identifier("project v2 command graph ID", graph_id)?;
                    validate_identifier("project v2 command track ID", track_id)?;
                }
                ProjectV2EffectChainOwner::Bus { graph_id, bus_id } => {
                    validate_identifier("project v2 command graph ID", graph_id)?;
                    validate_identifier("project v2 command bus ID", bus_id)?;
                }
            }
        }
        ProjectV2Command::ReplaceEffectRevision { before, after } => {
            if after.id != before.id || after.revision != before.revision + 1 || before == after {
                return Err(
                    "project v2 replacement effect revision is not an immutable successor".into(),
                );
            }
            before.digest()?;
            after.digest()?;
        }
        ProjectV2Command::SelectEffectRevision { before, after } => {
            if before == after || before.id != after.id {
                return Err(
                    "project v2 selected effect revisions must be distinct revisions of one ID"
                        .into(),
                );
            }
            validate_identifier("project v2 selected effect ID", &before.id)?;
            validate_identifier("project v2 selected effect ID", &after.id)?;
        }
    }
    Ok(())
}

fn validate_entry(entry: &ProjectV2JournalEntry) -> Result<(), String> {
    if entry.schema != PROJECT_V2_JOURNAL_SCHEMA
        || entry.schema_version != 1
        || entry.sequence == 0
        || entry.sequence > MAX_JSON_SAFE_INTEGER
    {
        return Err("unsupported project v2 journal entry schema or sequence".into());
    }
    validate_command(&entry.command)?;
    if entry.command.digest()? != entry.command_digest {
        return Err("project v2 journal command digest does not match".into());
    }
    if entry.parent_digest == entry.result_digest {
        return Err("project v2 journal command did not change the root".into());
    }
    Ok(())
}

fn validate_entry_chain(entries: &[ProjectV2JournalEntry]) -> Result<(), String> {
    for entry in entries {
        validate_entry(entry)?;
    }
    for pair in entries.windows(2) {
        if pair[1].sequence != pair[0].sequence + 1
            || pair[1].parent_digest != pair[0].result_digest
        {
            return Err(format!(
                "project v2 journal chain breaks at sequence {}",
                pair[1].sequence
            ));
        }
    }
    Ok(())
}

fn journal_prefix_digest(entries: &[ProjectV2JournalEntry]) -> Result<Digest, String> {
    digest_json(JOURNAL_PREFIX_DOMAIN, &entries, "project v2 journal prefix")
}

fn validate_checkpoint(checkpoint: &ProjectV2Checkpoint) -> Result<(), String> {
    if checkpoint.schema != PROJECT_V2_CHECKPOINT_SCHEMA
        || checkpoint.schema_version != 1
        || checkpoint.compacted_entries > MAX_JOURNAL_ENTRIES as u64
    {
        return Err("unsupported project v2 checkpoint".into());
    }
    checkpoint.snapshot.validate()?;
    if checkpoint.snapshot.digest()? != checkpoint.snapshot_digest {
        return Err("project v2 checkpoint snapshot digest does not match".into());
    }
    Ok(())
}

fn graph_mut<'a>(
    manifest: &'a mut ProjectV2Manifest,
    id: &str,
) -> Result<&'a mut ProjectV2Graph, String> {
    manifest
        .graphs
        .iter_mut()
        .find(|graph| graph.id == id)
        .ok_or_else(|| format!("project v2 graph {id} is missing"))
}

fn remove_exact<T: PartialEq>(
    items: &mut Vec<T>,
    expected: &T,
    context: &str,
) -> Result<(), String> {
    let index = items
        .iter()
        .position(|item| item == expected)
        .ok_or_else(|| format!("{context} before-state does not match"))?;
    items.remove(index);
    Ok(())
}

fn checked_revision(revision: u64) -> Result<u64, String> {
    revision
        .checked_add(1)
        .filter(|value| *value <= MAX_JSON_SAFE_INTEGER)
        .ok_or_else(|| "project v2 node revision overflows".to_string())
}

fn write_journal_line(
    writer: &mut impl std::io::Write,
    entry: &ProjectV2JournalEntry,
) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, entry)
        .map_err(|error| format!("serialize project v2 journal entry: {error}"))?;
    writer
        .write_all(b"\n")
        .map_err(|error| format!("write project v2 journal entry: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project_v2::tests::fixture;

    #[test]
    fn move_undo_redo_is_an_append_only_inverse() {
        let original = fixture();
        let command = ProjectV2Command::MoveClip {
            graph_id: "main".into(),
            clip_id: "clip".into(),
            from_track_id: "track".into(),
            from_start: ProjectV2Time::zero(48_000).unwrap(),
            to_track_id: "track".into(),
            to_start: ProjectV2Time::new(24_000, 48_000).unwrap(),
        };
        let (moved, first) = apply_project_v2_command(&original, 1, command).unwrap();
        let (undone, second) = undo_project_v2_command(&moved, 2, &first).unwrap();
        assert_eq!(
            undone.graph("main").unwrap().clips[0].timeline_start,
            ProjectV2Time::zero(48_000).unwrap()
        );
        assert_eq!(undone.parent_digest, Some(moved.digest().unwrap()));
        assert_eq!(second.parent_digest, first.result_digest);
        assert_ne!(undone.digest().unwrap(), original.digest().unwrap());
    }

    #[test]
    fn truncated_final_record_is_recoverable_but_complete_corruption_is_not() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("journal.ndjson");
        let original = fixture();
        let command = ProjectV2Command::MoveClip {
            graph_id: "main".into(),
            clip_id: "clip".into(),
            from_track_id: "track".into(),
            from_start: ProjectV2Time::zero(48_000).unwrap(),
            to_track_id: "track".into(),
            to_start: ProjectV2Time::new(1, 48_000).unwrap(),
        };
        let (_, entry) = apply_project_v2_command(&original, 1, command).unwrap();
        append_project_v2_journal(&path, &entry).unwrap();
        use std::io::Write as _;
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"{\"truncated\"")
            .unwrap();
        let report = read_project_v2_journal(&path).unwrap();
        assert_eq!(report.entries, vec![entry]);
        assert!(report.truncated_tail_bytes > 0);
    }

    #[test]
    fn effect_revision_selection_is_inverse_only_for_an_unreferenced_destination() {
        let mut original = fixture();
        let before = original.effects[0].reference().unwrap();
        let mut successor = original.effects[0].clone();
        successor.revision = 2;
        successor.parameters.insert(
            "gain".into(),
            ProjectV2ParameterValue::Rational(ProjectV2Rational::new(3, 4).unwrap()),
        );
        let after = successor.reference().unwrap();
        original.effects.push(successor);
        original.canonicalize();
        original.validate().unwrap();

        let command = ProjectV2Command::SelectEffectRevision {
            before: before.clone(),
            after: after.clone(),
        };
        let (selected, entry) = apply_project_v2_command(&original, 1, command.clone()).unwrap();
        assert_eq!(
            selected.graph("main").unwrap().tracks[0].effect_chain,
            vec![after.clone()]
        );
        let (undone, _) = undo_project_v2_command(&selected, 2, &entry).unwrap();
        assert_eq!(
            undone.graph("main").unwrap().tracks[0].effect_chain,
            vec![before.clone()]
        );

        let mut mixed = original;
        mixed.graphs[0].tracks.push(ProjectV2Track {
            id: "already-new".into(),
            revision: 1,
            parent_bus_id: "master".into(),
            muted: false,
            effect_chain: vec![after],
        });
        mixed.canonicalize();
        mixed.validate().unwrap();
        assert!(apply_project_v2_command(&mixed, 1, command)
            .unwrap_err()
            .contains("already referenced"));
    }

    #[test]
    fn checkpoint_authenticates_the_exact_compacted_prefix() {
        let original = fixture();
        let command = ProjectV2Command::MoveClip {
            graph_id: "main".into(),
            clip_id: "clip".into(),
            from_track_id: "track".into(),
            from_start: ProjectV2Time::zero(48_000).unwrap(),
            to_track_id: "track".into(),
            to_start: ProjectV2Time::new(1, 48_000).unwrap(),
        };
        let prior_root = original.digest().unwrap();
        let (snapshot, entry) = apply_project_v2_command(&original, 1, command).unwrap();
        let checkpoint =
            create_project_v2_checkpoint(prior_root, snapshot, std::slice::from_ref(&entry))
                .unwrap();
        verify_project_v2_checkpoint(&checkpoint, std::slice::from_ref(&entry)).unwrap();

        let directory = tempfile::tempdir().unwrap();
        write_project_v2_checkpoint(
            directory.path().join("checkpoint.json"),
            &checkpoint,
            std::slice::from_ref(&entry),
            CommitMode::NoClobber,
        )
        .unwrap();
        assert!(verify_project_v2_checkpoint(&checkpoint, &[])
            .unwrap_err()
            .contains("count"));

        let mut tampered = entry;
        tampered.sequence = 2;
        assert!(verify_project_v2_checkpoint(&checkpoint, &[tampered]).is_err());
    }

    #[test]
    fn empty_checkpoint_cannot_rebase_the_snapshot() {
        let snapshot = fixture();
        let snapshot_digest = snapshot.digest().unwrap();
        create_project_v2_checkpoint(snapshot_digest, snapshot.clone(), &[]).unwrap();

        let unrelated = digest_json(b"test", &"unrelated", "test digest").unwrap();
        assert!(create_project_v2_checkpoint(unrelated, snapshot, &[])
            .unwrap_err()
            .contains("preserve its prior root"));
    }
}
