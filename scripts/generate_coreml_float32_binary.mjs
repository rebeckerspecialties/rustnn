import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

// Independent exact-rational binary32 reference. No floating-point arithmetic
// participates in decoding, multiplying/dividing, or nearest-even rounding.
const root = path.resolve(process.argv[2] ?? path.join(import.meta.dirname, 'float32-binary-oracle'));
const packed = process.argv.includes('--packed');
function decode(bits) {
  const sign=bits>>>31, exponent=(bits>>>23)&255, fraction=bits&0x7fffff;
  if(exponent===255) return {sign,kind:fraction?'nan':'infinity'};
  if(exponent===0 && fraction===0) return {sign,kind:'zero'};
  return {sign,kind:'finite',significand:BigInt(exponent?fraction+0x800000:fraction),power:exponent?exponent-150:-149};
}
const width = n => n.toString(2).length;
function roundRational(n,d,power,sign) {
  assert(n>0n && d>0n);
  let exponent=width(n)-width(d);
  if(exponent>=0 ? n<(d<<BigInt(exponent)) : (n<<BigInt(-exponent))<d) exponent--;
  exponent+=power;
  if(exponent>127) return ((sign<<31)|0x7f800000)>>>0;
  const unit=exponent < -126 ? -149 : exponent-23;
  const shift=power-unit;
  const numerator=shift>=0 ? n<<BigInt(shift) : n;
  const denominator=shift>=0 ? d : d<<BigInt(-shift);
  let q=numerator/denominator;
  const twiceRemainder=2n*(numerator%denominator);
  if(twiceRemainder>denominator || (twiceRemainder===denominator && (q&1n))) q++;
  if(q===0n) return (sign<<31)>>>0;
  if(exponent < -126) {
    assert(q<=0x800000n);
    return ((sign<<31)|Number(q))>>>0;
  }
  if(q===0x1000000n) {q>>=1n; exponent++;}
  if(exponent>127) return ((sign<<31)|0x7f800000)>>>0;
  assert(q>=0x800000n && q<0x1000000n);
  return ((sign<<31)|((exponent+127)<<23)|Number(q-0x800000n))>>>0;
}
function reference(op,aBits,bBits) {
  const a=decode(aBits),b=decode(bBits),sign=a.sign^b.sign;
  const zero=(sign<<31)>>>0, infinity=(zero|0x7f800000)>>>0;
  if(a.kind==='nan'||b.kind==='nan') return 0x7fc00000;
  if(op==='mul') {
    if((a.kind==='zero'&&b.kind==='infinity')||(b.kind==='zero'&&a.kind==='infinity')) return 0x7fc00000;
    if(a.kind==='infinity'||b.kind==='infinity') return infinity;
    if(a.kind==='zero'||b.kind==='zero') return zero;
    return roundRational(a.significand*b.significand,1n,a.power+b.power,sign);
  }
  assert(op==='div');
  if((a.kind==='zero'&&b.kind==='zero')||(a.kind==='infinity'&&b.kind==='infinity')) return 0x7fc00000;
  if(a.kind==='infinity'||b.kind==='zero') return infinity;
  if(a.kind==='zero'||b.kind==='infinity') return zero;
  return roundRational(a.significand,b.significand,a.power-b.power,sign);
}
const view=new DataView(new ArrayBuffer(4));
function float(bits) {view.setUint32(0,bits,true);return view.getFloat32(0,true);}
function bits(value) {view.setFloat32(0,value,true);return view.getUint32(0,true);}
const isNaNBits=b=>((b>>>23)&255)===255 && (b&0x7fffff)!==0;
const hex=n=>n.toString(16).padStart(8,'0');
const boundary=[0,1,2,3,0x3fffff,0x400000,0x7ffffe,0x7fffff,0x800000,0x800001,
  0x00800002,0x01000000,0x33000000,0x33800000,0x34000000,0x3effffff,0x3f000000,
  0x3f000001,0x3f7fffff,0x3f800000,0x3f800001,0x3fffffff,0x40000000,0x40000001,
  0x4b000000,0x4b800000,0x4c000000,0x7effffff,0x7f000000,0x7f7ffffe,0x7f7fffff,
  0x7f800000,0x7f800001,0x7fc00000,0x7fffffff];
const domain=boundary.flatMap(v=>[v,(v|0x80000000)>>>0]);
const pairs=[];
for(const a of domain)for(const b of domain)pairs.push([a,b]);
let seed=0x53a14927;
function next(){seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return seed>>>0;}
for(let i=0;i<10000;i++)pairs.push([next(),next()]);
// Validate the rounding implementation on exact representations before using
// it as an oracle for arithmetic, including both signs and all exponent bins.
let interchange=0;
for(let e=0;e<255;e++)for(const f of [0,1,0x3fffff,0x400000,0x7ffffe,0x7fffff])for(const s of [0,1]){
  const original=((s<<31)|(e<<23)|f)>>>0,v=decode(original);
  if(v.kind==='finite')assert.equal(roundRational(v.significand,1n,v.power,v.sign),original);
  else {assert.equal(v.kind,'zero');assert.equal((v.sign<<31)>>>0,original);}
  interchange++;
}
const cases=[],records=[],stats={pairs:pairs.length,cases:0,interchange,finite:0,zero:0,infinity:0,nan:0,subnormalInputsNormalOutput:0};
for(const op of ['mul','div'])for(const [a,b] of pairs){
  const expected=reference(op,a,b), native=bits(op==='mul'?float(a)*float(b):float(a)/float(b));
  assert(isNaNBits(expected)?isNaNBits(native):expected===native, `${op} ${hex(a)} ${hex(b)} ${hex(expected)} != ${hex(native)}`);
  const kind=decode(expected).kind;stats[kind]++;
  if(([a,b].some(v=>(v&0x7f800000)===0&&(v&0x7fffff)!==0)) && (expected&0x7f800000)!==0 && (expected&0x7f800000)!==0x7f800000)stats.subnormalInputsNormalOutput++;
  cases.push(`${op} ${hex(a)} ${hex(b)} ${hex(expected)}`);
  if(packed) {
    const record=Buffer.alloc(16);
    [op==='mul'?0:1,a,b,expected].forEach((v,i)=>record.writeUInt32LE(v,i*4));
    records.push(record);
  }
}
stats.cases=cases.length;
fs.mkdirSync(root,{recursive:true});
const output=cases.join('\n')+'\n';
const receipt={date:new Date().toISOString(),method:'Exact BigInt rational operations and direct nearest-even binary32 rounding; special values checked by class, zeros by sign.',seed:'53a14927',stats,generatorSHA256:createHash('sha256').update(fs.readFileSync(new URL(import.meta.url))).digest('hex'),corpusSHA256:createHash('sha256').update(output).digest('hex'),hostCrossCheck:'Every generated case matched independent JavaScript binary64 operation then DataView Float32 conversion; NaN payload/sign not compared.'};
fs.writeFileSync(path.join(root,'cases.txt'),output);
if(packed) {
  const data=Buffer.concat(records);
  fs.writeFileSync(path.join(root,'cases.bin'),data);
  receipt.packed={layout:'Four little-endian uint32 words: operation (0 mul / 1 div), left bits, right bits, expected bits; NaNs checked by class.',bytes:data.length,sha256:createHash('sha256').update(data).digest('hex')};
}
fs.writeFileSync(path.join(root,'receipt.json'),JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify(receipt));
