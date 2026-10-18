//! Output sink abstraction so the MIL emitter can write to either an in-memory
//! buffer (the historical path) or directly to a file handle (the streaming
//! path used on memory-constrained devices like 32-bit Apple Watches).
//!
//! The streaming mode exists because a 1.6 MB `model.mil` text exceeds the
//! arm64_32 watchOS jetsam limit if accumulated naively in a `Vec<u8>` —
//! `Vec`'s exponential-doubling growth peaks at ~3× the final size during the
//! last realloc. Flushing ~256 KB chunks keeps the transient memory bounded.

use std::fs::File;
use std::io::{self, BufWriter, Write};

/// A text sink that either accumulates into an in-memory buffer or streams
/// UTF-8 bytes to a file. Errors during streaming are surfaced through
/// [`MILOutputSink::finalize`].
pub struct MILOutputSink {
    bytes_written: usize,
    backing: Backing,
    error: Option<io::Error>,
}

enum Backing {
    Memory(Vec<u8>),
    File { writer: BufWriter<File> },
}

impl MILOutputSink {
    /// Create an in-memory sink with optional initial capacity.
    pub fn in_memory(capacity: usize) -> Self {
        Self {
            bytes_written: 0,
            backing: Backing::Memory(Vec::with_capacity(capacity)),
            error: None,
        }
    }

    /// Wrap an open `File` handle for streaming output. The handle is wrapped
    /// in a 256 KB `BufWriter` so callers don't pay per-write syscall cost.
    pub fn streaming(file: File) -> Self {
        const FLUSH_BUFFER: usize = 256 * 1024;
        Self {
            bytes_written: 0,
            backing: Backing::File {
                writer: BufWriter::with_capacity(FLUSH_BUFFER, file),
            },
            error: None,
        }
    }

    /// Number of UTF-8 bytes written to this sink so far.
    pub fn bytes_written(&self) -> usize {
        self.bytes_written
    }

    /// Append a UTF-8 string. Errors during file I/O are captured and
    /// reported by [`MILOutputSink::finalize`].
    #[inline]
    pub fn write_str(&mut self, s: &str) {
        self.write_bytes(s.as_bytes());
    }

    /// Append raw UTF-8 bytes — the hot path used by `hex_float32_bytes`.
    /// Skips the `&str` validation that would otherwise serve no purpose
    /// for ASCII MIL output.
    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        if self.error.is_some() {
            return;
        }
        self.bytes_written += bytes.len();
        match &mut self.backing {
            Backing::Memory(buf) => buf.extend_from_slice(bytes),
            Backing::File { writer } => {
                if let Err(e) = writer.write_all(bytes) {
                    self.error = Some(e);
                }
            }
        }
    }

    /// Flush pending bytes and consume the sink. For an in-memory sink,
    /// returns the accumulated bytes; for a streaming sink, returns an empty
    /// vec after flushing the underlying writer.
    pub fn finalize(mut self) -> io::Result<Vec<u8>> {
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        match self.backing {
            Backing::Memory(buf) => Ok(buf),
            Backing::File { mut writer } => {
                writer.flush()?;
                Ok(Vec::new())
            }
        }
    }
}
