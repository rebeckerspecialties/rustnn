import CoreML
import Foundation

struct Fixture: Decodable { let name: String; let input: [Int32]; let expected: [Int32] }
struct Failure: Error { let reason: String }
let root = URL(fileURLWithPath: CommandLine.arguments[1])
let fixtures = try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: root.appendingPathComponent("cases.json")))
for fixture in fixtures {
    for fast in [false, true] {
      for backed in [false, true] {
        do {
            let compiled = try MLModel.compileModel(at: root.appendingPathComponent(fixture.name + ".mlmodel"))
            defer { try? FileManager.default.removeItem(at: compiled) }
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuOnly
            if fast { configuration.optimizationHints.specializationStrategy = .fastPrediction }
            let model = try MLModel(contentsOf: compiled, configuration: configuration)
            let input = try MLMultiArray(shape: [3, 3], dataType: .int32)
            let p = input.dataPointer.assumingMemoryBound(to: Int32.self)
            let strides = input.strides.map(\.intValue)
            for i in 0..<9 { p[(i / 3) * strides[0] + (i % 3) * strides[1]] = fixture.input[i] }
            let readback = (0..<9).map { p[($0 / 3) * strides[0] + ($0 % 3) * strides[1]] }
            guard readback == fixture.input else { throw Failure(reason: "input readback mismatch") }
            let options = MLPredictionOptions()
            let backing = try MLMultiArray(shape: [3,3], dataType: .int32)
            if backed { options.outputBackings = ["result": backing] }
            let prediction = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input": input]), options: options)
            guard let output = prediction.featureValue(for: "result")?.multiArrayValue,
                  output.dataType == .int32, output.shape.map(\.intValue) == [3, 3] else {
                throw Failure(reason: "invalid output dtype or shape")
            }
            let q = output.dataPointer.assumingMemoryBound(to: Int32.self)
            let s = output.strides.map(\.intValue)
            let actual = (0..<9).map { q[($0 / 3) * s[0] + ($0 % 3) * s[1]] }
            let row: [String: Any] = ["case": fixture.name, "fast_prediction": fast, "backed": backed, "same_backing": output.dataPointer == backing.dataPointer, "input_readback": readback,
                "actual": actual, "expected": fixture.expected, "exact": actual == fixture.expected,
                "output_shape": output.shape, "output_strides": output.strides, "dtype": output.dataType.rawValue]
            print(String(data: try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]), encoding: .utf8)!)
        } catch { print("ERROR \(fixture.name) fast=\(fast) backed=\(backed): \(error)") }
      }
    }
}
