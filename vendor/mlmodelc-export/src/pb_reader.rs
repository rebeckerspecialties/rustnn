//! Minimal protobuf wire-format reader.
//!
//! No external dependencies; the surface area is just enough to walk the
//! handful of messages we need from `MIL.proto` and `Model.proto`.

#[derive(Clone)]
pub(crate) struct PBReader<'a> {
    pub data: &'a [u8],
    pub offset: usize,
}

impl<'a> PBReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    pub fn is_at_end(&self) -> bool {
        self.offset >= self.data.len()
    }

    #[allow(dead_code)]
    pub fn read_byte(&mut self) -> u8 {
        let b = self.data[self.offset];
        self.offset += 1;
        b
    }

    /// Read a base-128 varint (up to 64 bits).
    pub fn read_varint(&mut self) -> u64 {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        while self.offset < self.data.len() {
            let b = self.data[self.offset] as u64;
            self.offset += 1;
            result |= (b & 0x7F) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        result
    }

    pub fn read_fixed32(&mut self) -> u32 {
        let mut v = 0u32.to_le_bytes();
        v.copy_from_slice(&self.data[self.offset..self.offset + 4]);
        self.offset += 4;
        u32::from_le_bytes(v)
    }

    pub fn read_float32(&mut self) -> f32 {
        f32::from_bits(self.read_fixed32())
    }

    /// Read a length-delimited field and return a borrowed slice of its bytes.
    pub fn read_length_delimited(&mut self) -> &'a [u8] {
        let len = self.read_varint() as usize;
        let start = self.offset;
        self.offset += len;
        &self.data[start..start + len]
    }

    /// Read a UTF-8 string (length-delimited).
    pub fn read_string(&mut self) -> String {
        let bytes = self.read_length_delimited();
        String::from_utf8_lossy(bytes).into_owned()
    }

    /// Read a protobuf tag. Returns `(field_number, wire_type)` or `None` at end.
    pub fn read_tag(&mut self) -> Option<(u32, u32)> {
        if self.is_at_end() {
            return None;
        }
        let tag = self.read_varint();
        Some(((tag >> 3) as u32, (tag & 0x07) as u32))
    }

    /// Skip an unknown field based on its wire type. Best-effort for unknown
    /// wire types — we walk past the bytes rather than failing.
    pub fn skip(&mut self, wire_type: u32) {
        match wire_type {
            0 => {
                let _ = self.read_varint();
            }
            1 => self.offset += 8,
            2 => {
                let len = self.read_varint() as usize;
                self.offset += len;
            }
            5 => self.offset += 4,
            _ => {}
        }
    }
}

/// Read packed varints from a length-delimited field's body.
#[allow(dead_code)]
pub(crate) fn read_packed_varints(data: &[u8]) -> Vec<u64> {
    let mut reader = PBReader::new(data);
    let mut result = Vec::new();
    while !reader.is_at_end() {
        result.push(reader.read_varint());
    }
    result
}

/// Read packed signed-varint values, truncated to `i32`.
pub(crate) fn read_packed_signed_varints(data: &[u8]) -> Vec<i32> {
    let mut reader = PBReader::new(data);
    let mut result = Vec::new();
    while !reader.is_at_end() {
        result.push(reader.read_varint() as i64 as i32);
    }
    result
}

/// Read packed `float32` values.
pub(crate) fn read_packed_floats(data: &[u8]) -> Vec<f32> {
    let mut reader = PBReader::new(data);
    let mut result = Vec::new();
    while !reader.is_at_end() {
        result.push(reader.read_float32());
    }
    result
}
