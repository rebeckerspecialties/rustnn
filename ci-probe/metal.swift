// Metal compute smoke test: a 1M-element float16 add, checked on the host.
import Foundation
import Metal
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    print("metal: no device"); exit(2)
}
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void add(device const half *a [[buffer(0)]], device const half *b [[buffer(1)]],
                device half *c [[buffer(2)]], uint i [[thread_position_in_grid]]) { c[i] = a[i] + b[i]; }
"""
let n = 1 << 20
let library = try device.makeLibrary(source: source, options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "add")!)
let a = [Float16](repeating: 1.5, count: n), b = [Float16](repeating: 0.25, count: n)
let ba = device.makeBuffer(bytes: a, length: n * 2)!, bb = device.makeBuffer(bytes: b, length: n * 2)!
let bc = device.makeBuffer(length: n * 2)!
let start = Date()
let command = queue.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline); encoder.setBuffer(ba, offset: 0, index: 0)
encoder.setBuffer(bb, offset: 0, index: 1); encoder.setBuffer(bc, offset: 0, index: 2)
encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: pipeline.maxTotalThreadsPerThreadgroup, height: 1, depth: 1))
encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
let c = bc.contents().bindMemory(to: Float16.self, capacity: n)
let wrong = (0..<n).filter { c[$0] != 1.75 }.count
print("metal: device=\(device.name) status=\(command.status.rawValue) error=\(String(describing: command.error)) wrong=\(wrong) ms=\(Int(Date().timeIntervalSince(start) * 1000))")
exit(wrong == 0 && command.error == nil ? 0 : 1)
