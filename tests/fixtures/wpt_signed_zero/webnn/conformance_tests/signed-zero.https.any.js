// Exercise the actual fixture dump paths, not a hand-written Rust JSON value.
if ('Float16Array' in globalThis && typeof Float16Array !== 'function') {
  throw new Error('Fixture harness must not advertise an unavailable typed array');
}
const bridgeTests = [];
for (const dataType of ['float16', 'float32']) {
  for (const constant of [false, true]) {
    for (const [kind, shape, data, expected] of [
      ['scalar', [], [-0], [-0]],
      ['vector', [4], [0, -0, 2 ** -24, -(2 ** -24)],
       [0, -0, 2 ** -24, -(2 ** -24)]],
      ['negative-fill', [3], -0, [-0, -0, -0]],
      ['positive-fill', [3], 0, [0, 0, 0]],
      ['nonfinite', [5], [0, -0, Infinity, -Infinity, NaN],
       [0, -0, Infinity, -Infinity, NaN]],
    ]) {
      bridgeTests.push({
        name: `${dataType} ${constant ? 'constant' : 'input'} ${kind}`,
        graph: {
          inputs: {
            input: {data, descriptor: {shape, dataType}, constant},
          },
          operators: [{name: 'identity', arguments: [{input: 'input'}],
                       outputs: 'output'}],
          expectedOutputs: {
            output: {data: expected, descriptor: {shape, dataType}},
          },
        },
      });
    }
  }
}
bridgeTests.push({
  name: 'float options retain negative zero without changing string labels',
  graph: {
    inputs: {
      input: {data: [1], descriptor: {shape: [1], dataType: 'float32'}},
    },
    operators: [{name: 'hardSigmoid', arguments: [
      {input: 'input'}, {options: {alpha: -0, beta: 0, label: '-0'}},
    ], outputs: 'output'}],
    expectedOutputs: {
      output: {data: [0], descriptor: {shape: [1], dataType: 'float32'}},
    },
  },
});
bridgeTests.push({
  name: 'integer axis accepts negative zero without changing string labels',
  graph: {
    inputs: {
      input: {data: [1], descriptor: {shape: [1], dataType: 'float32'}},
    },
    operators: [{name: 'softmax', arguments: [
      {input: 'input'}, {axis: -0}, {options: {label: '-0'}},
    ], outputs: 'output'}],
    expectedOutputs: {
      output: {data: [1], descriptor: {shape: [1], dataType: 'float32'}},
    },
  },
});
bridgeTests.push({
  name: 'integer axes accept negative zero without changing string labels',
  graph: {
    inputs: {
      input: {data: [1], descriptor: {shape: [1], dataType: 'float32'}},
    },
    operators: [{name: 'reduceSum', arguments: [
      {input: 'input'}, {options: {axes: [-0], label: '-0'}},
    ], outputs: 'output'}],
    expectedOutputs: {
      output: {data: [1], descriptor: {shape: [], dataType: 'float32'}},
    },
  },
});
webnn_conformance_test(bridgeTests, () => {}, () => ({metricType: 'ULP', value: 0}));
