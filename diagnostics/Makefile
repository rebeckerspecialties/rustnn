.PHONY: probe
probe:
	xcrun swiftc diagnostics/Int32Native.swift -framework CoreML -o /tmp/triangular-int32-native
	/tmp/triangular-int32-native diagnostics/models
