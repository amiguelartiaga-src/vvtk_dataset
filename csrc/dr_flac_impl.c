/*
 * dr_flac implementation file.
 * Compile this as C alongside vvtk_lib.cpp.
 * See dr_flac.h for license (public domain / MIT-0).
 */
#define DR_FLAC_NO_OGG          /* We only use raw FLAC bitstreams, not Ogg/FLAC */
#define DR_FLAC_NO_CRC          /* Skip CRC checks for speed — data integrity is guaranteed by our shard format */
#define DR_FLAC_NO_STDIO        /* No file I/O needed — we decode from memory */
#define DR_FLAC_IMPLEMENTATION
#include "dr_flac.h"
