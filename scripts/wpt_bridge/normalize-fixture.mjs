import {readFileSync} from 'node:fs';

// Shared with Rust option parsing; data and MLNumber preserve the sign of zero,
// while integer WebIDL fields keep their usual conversion to positive zero.
const fields = JSON.parse(readFileSync(new URL('./numeric-fields.json', import.meta.url), 'utf8'));
const signedZeroFields = new Set([...fields.tensorData, ...fields.floatOptions]);

export function normalizeValue(value, key = '') {
  if (typeof value === 'number' && Object.is(value, -0)) {
    return signedZeroFields.has(key) ? '-0' : 0;
  }
  if (typeof value === 'number' && !Number.isFinite(value)) {
    if (Number.isNaN(value)) return 'NaN';
    return value > 0 ? 'Infinity' : '-Infinity';
  }
  if (typeof value === 'bigint') return value.toString();
  if (Array.isArray(value)) return value.map(item => normalizeValue(item, key));
  if (value && typeof value === 'object') {
    const normalized = {};
    for (const [field, item] of Object.entries(value)) {
      normalized[field] = normalizeValue(item, field);
    }
    return normalized;
  }
  return value;
}
