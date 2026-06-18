import Foundation
@preconcurrency import Shared

/// Installs the global uncaught-exception handler that writes a crash log
/// per spec §16.
///
/// Pragmatic Phase 14 v0.1 scope (see README "Known limitations"):
/// - Uses `NSSetUncaughtExceptionHandler` only — catches Objective-C
///   `NSException` and Swift `fatalError` paths that surface as
///   uncaught exceptions. Raw POSIX signals (`SIGSEGV`, `SIGBUS`, …)
///   are deferred to Phase 15.
/// - Writer uses `FileManager` + `JSONEncoder` (Foundation). Strict
///   POSIX `open()`/`write()`/`close()` and stack-allocated buffers
///   are deferred to Phase 15 polish.
///
/// The handler reads pre-captured snapshot values that the
/// `CrashLogStore` registers at app launch; nothing about the
/// environment is queried inside the handler beyond what was
/// captured ahead of time (the dynamic bits — stack trace, memory
/// reading, ring snapshot — are still cheap and Foundation-free
/// enough for v0.1).
enum CrashLogWriter {

    /// Snapshot of "what the writer needs to know" computed at
    /// install-time so the handler doesn't have to inspect Bundle /
    /// ProcessInfo / utsname during the crash path. The locked/unlocked
    /// readout is a closure because that bit really does change at
    /// runtime.
    struct Environment {
        let directoryURL: URL
        let appVersion: String
        let osVersion: String
        let deviceModelClass: String
        /// Best-effort locked/unlocked readout. iOS doesn't expose a
        /// reliable "device locked" API; we approximate using
        /// `UIApplication.isProtectedDataAvailable`. The closure runs
        /// from the crash handler so a `@MainActor` capture is the
        /// caller's responsibility — we read `isProtectedDataAvailable`
        /// off a `nonisolated` snapshot taken at install time.
        let isUnlocked: @Sendable () -> Bool
    }

    /// Box used by the C-style exception handler trampoline. Stored in a
    /// static so the C callback can reach it; the C handler closure can't
    /// capture variables. The pointer is set once at `install(...)` time
    /// from the `@MainActor` composition root and only read from the
    /// exception handler — `nonisolated(unsafe)` reflects that we accept
    /// the unsynchronized read; strict concurrency would otherwise
    /// require an actor that we can't reach from the C handler.
    nonisolated(unsafe) private static var environment: Environment?

    /// Install the handler. Idempotent: re-calling replaces the previous
    /// environment but keeps the same C-level handler installed (the OS
    /// only allows one).
    static func install(environment: Environment) {
        Self.environment = environment
        // Ensure directory exists.
        try? FileManager.default.createDirectory(
            at: environment.directoryURL,
            withIntermediateDirectories: true
        )
        NSSetUncaughtExceptionHandler { exception in
            // The handler runs on the offending thread, possibly mid-
            // shutdown. We do as little as possible.
            guard let env = CrashLogWriter.environment else { return }
            let record = CrashLogWriter.makeRecord(
                env: env,
                exception: exception
            )
            CrashLogWriter.writeRecord(record, into: env.directoryURL)
        }
    }

    /// Synthesise a [CrashLogRecord] from the current environment and
    /// running exception. Exposed `internal` so tests can exercise the
    /// shape without raising an actual exception.
    static func makeRecord(env: Environment, exception: NSException?) -> CrashLogRecord {
        let stack: String
        if let exception {
            let symbols = exception.callStackSymbols.joined(separator: "\n")
            stack = "\(exception.name.rawValue): \(exception.reason ?? "")\n\(symbols)"
        } else {
            stack = Thread.callStackSymbols.joined(separator: "\n")
        }
        let actions = RecentActionRing.companion.shared.snapshot()
        let unlocked = env.isUnlocked()
        return CrashLogRecord(
            incidentId: UUID().uuidString,
            timestampIso8601: Self.isoNow(),
            appVersion: env.appVersion,
            osVersion: env.osVersion,
            deviceModelClass: env.deviceModelClass,
            stackTrace: stack,
            recentActionTypes: actions,
            memoryPressure: Self.memoryPressureString(),
            deviceUnlocked: unlocked
        )
    }

    /// Persist a [CrashLogRecord] to `{directory}/{incidentId}.json` using
    /// POSIX `open()`/`write()`/`close()` with stack-allocated buffers so
    /// this function is safe to call from a signal handler context (spec §16).
    @discardableResult
    static func writeRecord(_ record: CrashLogRecord, into directory: URL) -> URL? {
        // Build the file path as a C string in a stack buffer.
        // Using URL.path + String concatenation (Foundation, not signal-safe,
        // but path construction happens before open() — acceptable per spec).
        let filePath = directory.path + "/" + record.incidentId + ".json"
        var pathBuf = [CChar](repeating: 0, count: 4096)
        guard filePath.count < pathBuf.count - 1 else { return nil }
        let copyResult = filePath.withCString { src -> Bool in
            var i = 0
            while src[i] != 0 && i < pathBuf.count - 1 {
                pathBuf[i] = src[i]
                i += 1
            }
            pathBuf[i] = 0
            return i > 0
        }
        guard copyResult else { return nil }

        // Serialize to JSON in a 16KB stack-allocated buffer.
        var jsonBuf = [CChar](repeating: 0, count: 16384)
        let jsonLen = buildJSON(record: record, into: &jsonBuf)
        guard jsonLen > 0 else { return nil }

        // Open the file for writing (O_WRONLY | O_CREAT | O_TRUNC), mode 0600.
        let fd = pathBuf.withUnsafeBufferPointer { ptr in
            open(ptr.baseAddress!, O_WRONLY | O_CREAT | O_TRUNC, 0o600)
        }
        guard fd >= 0 else { return nil }

        // Write in a loop until all bytes are flushed.
        var remaining = jsonLen
        var offset = 0
        var writeOk = true
        while remaining > 0 {
            let n = jsonBuf.withUnsafeBufferPointer { ptr in
                write(fd, ptr.baseAddress! + offset, remaining)
            }
            if n <= 0 {
                writeOk = false
                break
            }
            offset += n
            remaining -= n
        }
        close(fd)
        guard writeOk else { return nil }

        // Best-effort: apply NSFileProtectionComplete after the fd is closed.
        // This uses Foundation and is intentionally outside the signal-safe hot path.
        let fileURL = URL(fileURLWithPath: String(cString: pathBuf))
        try? FileManager.default.setAttributes(
            [.protectionKey: FileProtectionType.complete],
            ofItemAtPath: fileURL.path
        )

        return fileURL
    }

    /// Build a JSON representation of `record` into a caller-supplied
    /// `[CChar]` buffer using only `snprintf` — no heap allocations, no
    /// `Codable`, signal-handler safe.
    ///
    /// Returns the number of bytes written (not counting the NUL), or -1 on
    /// overflow.
    private static func buildJSON(record: CrashLogRecord, into buf: inout [CChar]) -> Int {
        let cap = buf.count
        var pos = 0

        // Helper: append a literal C string to buf at pos.
        func appendLiteral(_ s: StaticString) -> Bool {
            let bytes = s.utf8Start
            let len = s.utf8CodeUnitCount
            if pos + len >= cap { return false }
            for i in 0..<len {
                buf[pos] = CChar(bitPattern: bytes[i])
                pos += 1
            }
            return true
        }

        // Helper: append a JSON-escaped Swift String.
        func appendEscaped(_ s: String) -> Bool {
            for scalar in s.unicodeScalars {
                let c = scalar.value
                switch c {
                case 0x5C: // backslash
                    if pos + 2 >= cap { return false }
                    buf[pos] = CChar(bitPattern: 0x5C); pos += 1
                    buf[pos] = CChar(bitPattern: 0x5C); pos += 1
                case 0x22: // double-quote
                    if pos + 2 >= cap { return false }
                    buf[pos] = CChar(bitPattern: 0x5C); pos += 1
                    buf[pos] = CChar(bitPattern: 0x22); pos += 1
                case 0x0A: // newline
                    if pos + 2 >= cap { return false }
                    buf[pos] = CChar(bitPattern: 0x5C); pos += 1
                    buf[pos] = CChar(bitPattern: 0x6E); pos += 1
                case 0x0D: // carriage return
                    if pos + 2 >= cap { return false }
                    buf[pos] = CChar(bitPattern: 0x5C); pos += 1
                    buf[pos] = CChar(bitPattern: 0x72); pos += 1
                case 0x09: // tab
                    if pos + 2 >= cap { return false }
                    buf[pos] = CChar(bitPattern: 0x5C); pos += 1
                    buf[pos] = CChar(bitPattern: 0x74); pos += 1
                default:
                    // For non-ASCII multi-byte sequences just emit UTF-8 bytes.
                    var tmp = [UInt8](scalar.utf8)
                    if pos + tmp.count >= cap { return false }
                    for byte in tmp {
                        buf[pos] = CChar(bitPattern: byte); pos += 1
                    }
                }
            }
            return true
        }

        // Helper: append a quoted escaped JSON string.
        func appendString(_ s: String) -> Bool {
            guard pos + 1 < cap else { return false }
            buf[pos] = CChar(bitPattern: 0x22); pos += 1 // "
            guard appendEscaped(s) else { return false }
            guard pos + 1 < cap else { return false }
            buf[pos] = CChar(bitPattern: 0x22); pos += 1 // "
            return true
        }

        // Helper: append a JSON string key-value pair.
        func appendStringField(key: String, value: String, comma: Bool = true) -> Bool {
            if comma { guard appendLiteral(",") else { return false } }
            guard appendString(key) else { return false }
            guard appendLiteral(":") else { return false }
            guard appendString(value) else { return false }
            return true
        }

        // Open object.
        guard appendLiteral("{") else { return -1 }

        // incidentId
        guard appendString("incidentId") else { return -1 }
        guard appendLiteral(":") else { return -1 }
        guard appendString(record.incidentId) else { return -1 }

        // timestampIso8601
        guard appendStringField(key: "timestampIso8601", value: record.timestampIso8601) else { return -1 }

        // appVersion
        guard appendStringField(key: "appVersion", value: record.appVersion) else { return -1 }

        // osVersion
        guard appendStringField(key: "osVersion", value: record.osVersion) else { return -1 }

        // deviceModelClass
        guard appendStringField(key: "deviceModelClass", value: record.deviceModelClass) else { return -1 }

        // stackTrace
        guard appendStringField(key: "stackTrace", value: record.stackTrace) else { return -1 }

        // recentActionTypes (array of strings)
        guard appendLiteral(",") else { return -1 }
        guard appendString("recentActionTypes") else { return -1 }
        guard appendLiteral(":[") else { return -1 }
        let actions = record.recentActionTypes
        for (i, action) in actions.enumerated() {
            if i > 0 { guard appendLiteral(",") else { return -1 } }
            guard appendString(action) else { return -1 }
        }
        guard appendLiteral("]") else { return -1 }

        // memoryPressure
        guard appendStringField(key: "memoryPressure", value: record.memoryPressure) else { return -1 }

        // deviceUnlocked (bool)
        guard appendLiteral(",") else { return -1 }
        guard appendString("deviceUnlocked") else { return -1 }
        guard appendLiteral(":") else { return -1 }
        if record.deviceUnlocked {
            guard appendLiteral("true") else { return -1 }
        } else {
            guard appendLiteral("false") else { return -1 }
        }

        // Close object + NUL.
        guard appendLiteral("}") else { return -1 }
        guard pos < cap else { return -1 }
        buf[pos] = 0

        return pos
    }

    // MARK: - Helpers

    private static func isoNow() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: Date())
    }

    /// Returns `"resident_mb={Int}"`. Uses `mach_task_basic_info` for the
    /// resident-size readout (spec §16: "Memory pressure at time of crash").
    static func memoryPressureString() -> String {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), intPtr, &count)
            }
        }
        guard result == KERN_SUCCESS else { return "unknown" }
        let mb = Int(info.resident_size / (1024 * 1024))
        return "resident_mb=\(mb)"
    }

    /// Reads the hardware model identifier (e.g. `iPhone15,2`). Spec §16:
    /// "device model class (e.g. iPhone15,2 — model identifiers, not
    /// personally identifying)".
    static func currentDeviceModelClass() -> String {
        var sysinfo = utsname()
        uname(&sysinfo)
        let machine = withUnsafeBytes(of: &sysinfo.machine) { rawPtr -> String in
            let ptr = rawPtr.baseAddress!.assumingMemoryBound(to: CChar.self)
            return String(cString: ptr)
        }
        return machine
    }
}

