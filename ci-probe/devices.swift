// Prints what Core ML and Metal can see on this machine (host or virtual machine).
import CoreML
import Foundation
import Metal
print("os:", ProcessInfo.processInfo.operatingSystemVersionString)
if #available(macOS 14.0, *) {
    for device in MLComputeDevice.allComputeDevices {
        switch device {
        case .cpu: print("coreml compute device: cpu")
        case .gpu(let gpu): print("coreml compute device: gpu", gpu.metalDevice.name)
        case .neuralEngine(let engine): print("coreml compute device: neuralEngine cores=\(engine.totalCoreCount)")
        @unknown default: print("coreml compute device: unknown")
        }
    }
}
for device in MTLCopyAllDevices() {
    print("metal device:", device.name, "unifiedMemory:", device.hasUnifiedMemory,
          "recommendedMaxWorkingSet MiB:", device.recommendedMaxWorkingSetSize >> 20)
}
