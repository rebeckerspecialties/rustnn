//! Private scratch storage for source compilation. No inference tensors or
//! source files map this storage; each file belongs to one compiler invocation.
use memmap2::{MmapMut, MmapOptions};
use std::{
    fs::File,
    io,
    ops::{Deref, DerefMut},
};

/// Temporary owned weights, optionally backed by a private local file.
/// Flushing makes modified pages reclaimable without a second heap-sized copy.
pub struct CoremlWeightStorage {
    memory: Vec<u8>,
    mapped: Option<MmapMut>,
    file: Option<File>,
    length: usize,
}
impl CoremlWeightStorage {
    pub(crate) fn memory(data: Vec<u8>) -> Self {
        Self {
            memory: data,
            mapped: None,
            file: None,
            length: 0,
        }
    }
    pub(crate) fn file_backed() -> io::Result<Self> {
        Ok(Self {
            memory: Vec::new(),
            mapped: None,
            file: Some(tempfile::tempfile()?),
            length: 0,
        })
    }
    pub(crate) fn into_memory(self) -> Vec<u8> {
        assert!(self.file.is_none());
        self.memory
    }
    /// Flush compiler writes to the private scratch file. This does not access
    /// runtime tensors and is never called during prediction.
    pub fn flush(&self) -> io::Result<()> {
        self.mapped.as_ref().map_or(Ok(()), |m| m.flush())
    }
    pub(crate) fn flush_range(&self, start: usize, len: usize) -> io::Result<()> {
        self.mapped
            .as_ref()
            .map_or(Ok(()), |m| m.flush_range(start, len))
    }
    pub(crate) fn try_reserve(&mut self, additional: usize) -> io::Result<()> {
        let Some(file) = &self.file else {
            return self
                .memory
                .try_reserve(additional)
                .map_err(io::Error::other);
        };
        let required = self
            .length
            .checked_add(additional)
            .ok_or_else(|| io::Error::other("weight extent overflow"))?;
        if required <= self.mapped.as_ref().map_or(0, |m| m.len()) {
            return Ok(());
        }
        self.flush()?;
        let capacity = required
            .checked_add(4095)
            .map(|n| n & !4095)
            .ok_or_else(|| io::Error::other("weight capacity overflow"))?;
        file.set_len(capacity as u64)?;
        // The anonymous temporary file never escapes this owner. There are no
        // other writers or truncations, and remapping requires exclusive self.
        self.mapped = Some(unsafe { MmapOptions::new().len(capacity).map_mut(file)? });
        Ok(())
    }
    pub(crate) fn resize(&mut self, len: usize, value: u8) {
        if self.file.is_none() {
            self.memory.resize(len, value);
            return;
        }
        let map = self
            .mapped
            .as_mut()
            .expect("reserve before writing weights");
        assert!(
            len <= map.len(),
            "weight writer exceeded its reserved extent"
        );
        if len > self.length {
            map[self.length..len].fill(value);
        }
        self.length = len;
    }
    pub(crate) fn extend_from_slice(&mut self, bytes: &[u8]) {
        if self.file.is_none() {
            self.memory.extend_from_slice(bytes);
            return;
        }
        let start = self.length;
        self.resize(start.checked_add(bytes.len()).expect("reserved extent"), 0);
        self[start..].copy_from_slice(bytes);
    }
    pub(crate) fn truncate(&mut self, len: usize) {
        if self.file.is_none() {
            self.memory.truncate(len);
        } else {
            self.length = self.length.min(len);
        }
    }
}
impl Deref for CoremlWeightStorage {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        if self.file.is_some() {
            self.mapped.as_ref().map_or(&[], |m| &m[..self.length])
        } else {
            &self.memory
        }
    }
}
impl DerefMut for CoremlWeightStorage {
    fn deref_mut(&mut self) -> &mut [u8] {
        if self.file.is_some() {
            self.mapped
                .as_mut()
                .map_or(&mut [], |m| &mut m[..self.length])
        } else {
            &mut self.memory
        }
    }
}
impl AsRef<[u8]> for CoremlWeightStorage {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn growth_flush_and_shrink_preserve_all_live_bytes() {
        let mut b = CoremlWeightStorage::file_backed().unwrap();
        b.try_reserve(7).unwrap();
        b.extend_from_slice(b"abcdefg");
        b.try_reserve(8192).unwrap();
        assert_eq!(&*b, b"abcdefg");
        b.resize(8200, 19);
        b[0] = 42;
        b.flush().unwrap();
        b.truncate(4);
        b.try_reserve(16384).unwrap();
        assert_eq!(&*b, b"*bcd");
        b.resize(8, 0);
        assert_eq!(&b[4..], &[0; 4]);
    }
    #[test]
    fn impossible_growth_is_an_error_without_changing_bytes() {
        let mut b = CoremlWeightStorage::file_backed().unwrap();
        b.try_reserve(1).unwrap();
        b.extend_from_slice(&[7]);
        assert!(b.try_reserve(usize::MAX).is_err());
        assert_eq!(&*b, &[7]);
    }
}
