//! Resource limits for metadata parsing and serialization.

/// Finite limits applied while reading and writing metadata.
///
/// The defaults are intentionally generous enough for ordinary cover art and
/// chapter lists, while keeping attacker-controlled counts and allocations
/// bounded. All sizes are measured in bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct MetadataLimits {
    /// Maximum aggregate size of comment bodies and decoded/native pictures.
    pub max_total_bytes: usize,
    /// Maximum size of one text, binary, picture, or raw comment item.
    pub max_item_bytes: usize,
    /// Maximum number of metadata items, including pictures.
    pub max_items: usize,
    /// Maximum size of one FLAC metadata block.
    pub max_flac_block_bytes: usize,
    /// Maximum number of FLAC metadata blocks before the audio frames.
    pub max_flac_blocks: usize,
    /// Maximum size of one reconstructed Ogg packet while inspecting tags.
    pub max_ogg_packet_bytes: usize,
    /// Maximum number of Ogg pages inspected before the comment packet.
    pub max_ogg_pages: usize,
    /// Maximum number of logical Ogg streams encountered while finding tags.
    pub max_ogg_streams: usize,
}

impl MetadataLimits {
    /// 64 MiB aggregate metadata budget.
    pub const DEFAULT_MAX_TOTAL_BYTES: usize = 64 * 1024 * 1024;
    /// 16 MiB per-item budget, matching Lofty's finite default allocation cap.
    pub const DEFAULT_MAX_ITEM_BYTES: usize = 16 * 1024 * 1024;
    /// Maximum number of tag values and pictures retained by default.
    pub const DEFAULT_MAX_ITEMS: usize = 16_384;
    /// Maximum FLAC block size accepted by default.
    pub const DEFAULT_MAX_FLAC_BLOCK_BYTES: usize = 16 * 1024 * 1024 - 1;
    /// Maximum number of FLAC metadata blocks accepted by default.
    pub const DEFAULT_MAX_FLAC_BLOCKS: usize = 1_024;
    /// Maximum Ogg metadata packet size accepted by default.
    pub const DEFAULT_MAX_OGG_PACKET_BYTES: usize = 16 * 1024 * 1024;
    /// Maximum Ogg pages searched for codec headers and comments by default.
    pub const DEFAULT_MAX_OGG_PAGES: usize = 4_096;
    /// Maximum logical Ogg streams searched for metadata by default.
    pub const DEFAULT_MAX_OGG_STREAMS: usize = 64;
}

impl Default for MetadataLimits {
    fn default() -> Self {
        Self {
            max_total_bytes: Self::DEFAULT_MAX_TOTAL_BYTES,
            max_item_bytes: Self::DEFAULT_MAX_ITEM_BYTES,
            max_items: Self::DEFAULT_MAX_ITEMS,
            max_flac_block_bytes: Self::DEFAULT_MAX_FLAC_BLOCK_BYTES,
            max_flac_blocks: Self::DEFAULT_MAX_FLAC_BLOCKS,
            max_ogg_packet_bytes: Self::DEFAULT_MAX_OGG_PACKET_BYTES,
            max_ogg_pages: Self::DEFAULT_MAX_OGG_PAGES,
            max_ogg_streams: Self::DEFAULT_MAX_OGG_STREAMS,
        }
    }
}
