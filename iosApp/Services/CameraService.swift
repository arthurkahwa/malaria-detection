@preconcurrency import AVFoundation
import CoreVideo
import Foundation
import Observation
@preconcurrency import Shared
import UIKit

/// AVFoundation-backed capture service (spec §11 active screening).
///
/// Owns a single `AVCaptureSession` with a back-camera input and a video
/// data output that streams `CMSampleBuffer`s into a serial queue. The
/// view drives the lifecycle: `start()` on appear, `stop()` on disappear,
/// background, or auto-logout. Per spec §6 ownership, the per-prediction
/// classify task runs from the view's `.task`/async closure — the
/// service itself only owns the AVFoundation lifecycle and the
/// most-recent frame.
///
/// Capture is one-shot per tap (`captureOneFrame()`), not continuous:
/// the running session keeps the preview live, and Capture grabs the
/// most-recent `CVPixelBuffer` from the delegate's serial queue,
/// converts BGRA → RGB, and returns a `Shared.ImageInput`. The shared
/// `Preprocessor.resizeRGB` then downsamples to 128 at the Core ML
/// boundary; no view-side resize is required.
///
/// **Simulator caveat:** `AVCaptureSession` configuration succeeds on
/// the iOS Simulator, but Apple's simulator does not produce real
/// frames. `captureOneFrame()` will time out / never receive a
/// sample on simulator. The Capture path is exercised on real iPhone
/// hardware only. Unit tests exercise the surrounding state machine
/// and the override path without touching `AVCaptureSession`.
@Observable
@MainActor
final class CameraService {

    enum State: Sendable, Equatable {
        case idle
        case starting
        case running
        case stopped
        case failed(message: String)
    }

    enum CameraError: LocalizedError {
        case permissionDenied
        case noBackCamera
        case sessionFailed(reason: String)
        case captureTimeout

        var errorDescription: String? {
            switch self {
            case .permissionDenied:
                return "Camera access is required. Open Settings → Malaria Detector → Camera to grant access."
            case .noBackCamera:
                return "No back camera available on this device."
            case .sessionFailed(let reason):
                return "Camera session failed: \(reason)"
            case .captureTimeout:
                return "Capture timed out before a frame was available. Make sure the camera preview is running."
            }
        }
    }

    private(set) var state: State = .idle

    /// Exposed for `CameraPreviewView` to attach the preview layer.
    /// The session reference is stable across the service's lifetime;
    /// `start()` / `stop()` toggle `isRunning`, not the identity.
    nonisolated(unsafe) let session: AVCaptureSession = AVCaptureSession()

    // Background queue for blocking AVCaptureSession lifecycle calls
    // and the video-data output delegate. `startRunning()` blocks for
    // hundreds of ms — never call it on the main actor.
    private let sessionQueue = DispatchQueue(label: "camera.session", qos: .userInitiated)
    private let outputQueue = DispatchQueue(label: "camera.output", qos: .userInitiated)

    private nonisolated(unsafe) let output: AVCaptureVideoDataOutput = AVCaptureVideoDataOutput()
    private let frameStore = LatestFrameStore()
    private var sampleDelegate: SampleBufferDelegate?
    private var didConfigure = false

    init() {}

    // MARK: - Lifecycle

    func start() async throws {
        guard state != .running, state != .starting else { return }
        state = .starting

        let granted = await requestPermission()
        guard granted else {
            state = .failed(message: CameraError.permissionDenied.errorDescription ?? "denied")
            throw CameraError.permissionDenied
        }

        do {
            try await configureIfNeeded()
        } catch {
            state = .failed(message: error.localizedDescription)
            throw error
        }

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            sessionQueue.async { [session] in
                if !session.isRunning {
                    session.startRunning()
                }
                cont.resume()
            }
        }
        state = .running
    }

    func stop() {
        sessionQueue.async { [session] in
            if session.isRunning {
                session.stopRunning()
            }
        }
        state = .stopped
    }

    /// Capture the next available frame from the running session and
    /// return it as a `Shared.ImageInput` carrying the buffer's
    /// native dimensions (BGRA → RGB on the way out).
    ///
    /// Polls `LatestFrameStore` for up to ~2 seconds. On the simulator
    /// the delegate never fires, so this throws `captureTimeout`. On
    /// real hardware the preview producing ~30fps means the first
    /// poll usually returns immediately.
    func captureOneFrame() async throws -> ImageInput {
        guard state == .running else {
            throw CameraError.sessionFailed(reason: "Camera is not running")
        }

        let deadline = Date().addingTimeInterval(2.0)
        while Date() < deadline {
            if let image = frameStore.takeAndBuild() {
                return image
            }
            try await Task.sleep(nanoseconds: 30_000_000) // ~30ms ≈ 1 frame at 30fps
        }
        throw CameraError.captureTimeout
    }

    // MARK: - Permission

    private func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    // MARK: - Session configuration

    /// Builds the session graph once. Subsequent `start()` calls just
    /// flip `isRunning`. Throws `noBackCamera` / `sessionFailed` if
    /// configuration cannot complete.
    private func configureIfNeeded() async throws {
        guard !didConfigure else { return }

        let delegate = SampleBufferDelegate(store: frameStore)
        sampleDelegate = delegate

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            sessionQueue.async { [session, output, outputQueue] in
                session.beginConfiguration()
                session.sessionPreset = .photo

                guard let device = AVCaptureDevice.default(
                    .builtInWideAngleCamera,
                    for: .video,
                    position: .back
                ) else {
                    session.commitConfiguration()
                    cont.resume(throwing: CameraError.noBackCamera)
                    return
                }

                do {
                    let input = try AVCaptureDeviceInput(device: device)
                    guard session.canAddInput(input) else {
                        session.commitConfiguration()
                        cont.resume(throwing: CameraError.sessionFailed(reason: "Cannot add camera input."))
                        return
                    }
                    session.addInput(input)
                } catch {
                    session.commitConfiguration()
                    cont.resume(throwing: CameraError.sessionFailed(reason: error.localizedDescription))
                    return
                }

                output.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String:
                        NSNumber(value: kCVPixelFormatType_32BGRA)
                ]
                output.alwaysDiscardsLateVideoFrames = true
                output.setSampleBufferDelegate(delegate, queue: outputQueue)
                guard session.canAddOutput(output) else {
                    session.commitConfiguration()
                    cont.resume(throwing: CameraError.sessionFailed(reason: "Cannot add video data output."))
                    return
                }
                session.addOutput(output)

                if let connection = output.connection(with: .video) {
                    if connection.isVideoRotationAngleSupported(90) {
                        // 90° = portrait when the back camera is upright.
                        connection.videoRotationAngle = 90
                    }
                }

                session.commitConfiguration()
                cont.resume(returning: ())
            }
        }

        didConfigure = true
    }
}

// MARK: - Latest frame store

/// Lock-protected box that owns the most-recent CVPixelBuffer the
/// delegate has produced. Single-shot semantics: `takeAndBuild()`
/// consumes the stored buffer, converts it to a `Shared.ImageInput`,
/// and clears the slot atomically. The next Capture tap waits for a
/// fresh frame rather than re-reading an already-classified one.
///
/// CVPixelBuffer is a CoreFoundation type and is not `Sendable` in
/// strict-concurrency mode. We keep it confined behind the lock and
/// never let the raw buffer escape — only the freshly-built
/// `ImageInput` (which is plain bytes + ints) crosses the isolation
/// boundary.
final class LatestFrameStore: @unchecked Sendable {
    private let lock = NSLock()
    private var latest: CVPixelBuffer?

    func set(_ buffer: CVPixelBuffer) {
        lock.lock()
        defer { lock.unlock() }
        latest = buffer
    }

    /// Atomically take the most-recent buffer and build an
    /// `ImageInput` from it. Returns `nil` if no buffer is queued.
    func takeAndBuild() -> ImageInput? {
        lock.lock()
        let buffer = latest
        latest = nil
        lock.unlock()
        guard let buffer else { return nil }
        return try? ImageInputBuilder.makeImageInput(from: buffer)
    }
}

// MARK: - Sample buffer delegate

/// Concrete `AVCaptureVideoDataOutputSampleBufferDelegate`. The
/// delegate is invoked on the camera's outputQueue (background); it
/// just hands the pixel buffer to the lock-protected store.
final class SampleBufferDelegate: NSObject,
    AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {

    private let store: LatestFrameStore

    init(store: LatestFrameStore) {
        self.store = store
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        store.set(pixelBuffer)
    }
}

// MARK: - Pixel buffer → ImageInput

/// Pulled out as its own type so `CameraServiceTests` (when they
/// land) can exercise the BGRA → RGB packing without spinning up
/// AVCaptureSession. For Phase 8 this stays internal.
enum ImageInputBuilder {

    static func makeImageInput(from pixelBuffer: CVPixelBuffer) throws -> ImageInput {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw CameraService.CameraError.sessionFailed(reason: "Locked pixel buffer has no base address.")
        }

        // Source: BGRA, 4 bytes per pixel.
        // Target: RGB, 3 bytes per pixel, row-packed.
        let pixelCount = width * height
        let byteArray = KotlinByteArray(size: Int32(pixelCount * 3))
        let src = baseAddress.assumingMemoryBound(to: UInt8.self)

        var outIdx: Int32 = 0
        for row in 0..<height {
            let rowStart = row * bytesPerRow
            var col = 0
            while col < width {
                let i = rowStart + col * 4
                let b = src[i]
                let g = src[i + 1]
                let r = src[i + 2]
                // Kotlin Byte is signed (Int8); reinterpret the raw bits.
                byteArray.set(index: outIdx, value: Int8(bitPattern: r))
                byteArray.set(index: outIdx + 1, value: Int8(bitPattern: g))
                byteArray.set(index: outIdx + 2, value: Int8(bitPattern: b))
                outIdx += 3
                col += 1
            }
        }

        return ImageInput(
            rgbBytes: byteArray,
            width: Int32(width),
            height: Int32(height)
        )
    }
}
