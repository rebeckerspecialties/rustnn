// usage: plan <container.mlmodelc or modelN dir> <policy 0 cpu|1 cpuAndGPU|2 all|3 cpuAndNeuralEngine> [maxChildren]
// Prints, per compiled child (directory holding model.mil), the operator kinds the plan marks supported and
// preferred on each device, in JSON. Same MLComputePlan API as the app's ComputePlanReport.
import CoreML
import Foundation
let args = CommandLine.arguments
let root = URL(fileURLWithPath: args[1]); let policy = Int32(args[2]) ?? 3; let maxChildren = args.count > 3 ? Int(args[3]) ?? 0 : 0
let configuration = MLModelConfiguration()
configuration.computeUnits = [MLComputeUnits.cpuOnly, .cpuAndGPU, .all, .cpuAndNeuralEngine][Int(policy)]
var bundles: [URL] = []
if FileManager.default.fileExists(atPath: root.appendingPathComponent("model.mil").path) { bundles = [root] }
else if let e = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) {
    for case let u as URL in e where u.lastPathComponent == "model.mil" { bundles.append(u.deletingLastPathComponent()) }
}
func idx(_ u: URL) -> Int { Int(u.lastPathComponent.dropFirst(5)) ?? 0 }
bundles.sort { idx($0) < idx($1) }
if maxChildren > 0 && bundles.count > maxChildren { bundles = Array(bundles.prefix(maxChildren)) }
func deviceName(_ d: MLComputeDevice) -> String { switch d { case .cpu: return "cpu"; case .gpu: return "gpu"; case .neuralEngine: return "neuralEngine"; @unknown default: return "unknown" } }
var out: [[String: Any]] = []
let sem = DispatchSemaphore(value: 0)
Task {
    for url in bundles {
        var entry: [String: Any] = ["child": url.lastPathComponent]
        do {
            let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
            guard case let .program(program) = plan.modelStructure else { entry["error"] = "not a program"; out.append(entry); continue }
            var supported: [String: [String: Int]] = [:], preferred: [String: [String: Int]] = [:], ops = 0, unsupportedNE: [String: Int] = [:]
            for (_, function) in program.functions {
                for op in function.block.operations {
                    ops += 1
                    let kind = op.operatorName
                    if let usage = plan.deviceUsage(for: op) {
                        let p = deviceName(usage.preferred); preferred[p, default: [:]][kind, default: 0] += 1
                        var onNE = false
                        for d in usage.supported { let n = deviceName(d); supported[n, default: [:]][kind, default: 0] += 1; if n == "neuralEngine" { onNE = true } }
                        if !onNE { unsupportedNE[kind, default: 0] += 1 }
                    }
                }
            }
            entry["operations"] = ops; entry["supported"] = supported; entry["preferred"] = preferred; entry["unsupportedOnNeuralEngine"] = unsupportedNE
        } catch { entry["error"] = "\(error)" }
        out.append(entry)
    }
    sem.signal()
}
sem.wait()
let data = try! JSONSerialization.data(withJSONObject: ["policy": policy, "children": out], options: [.prettyPrinted, .sortedKeys])
print(String(data: data, encoding: .utf8)!)
