package com.malaria.android.services

import android.content.Context
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.malaria.domain.ImageInput
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * CameraX-backed frame capture (spec §11 active screening).
 *
 * Mirror of `iosApp/Services/CameraService.swift`. Owns a `Preview` use
 * case + an `ImageAnalysis` use case bound to the host activity's
 * lifecycle. Per spec §6 ownership, the per-prediction classify task
 * runs from the composable's `rememberCoroutineScope()` — the service
 * itself only owns CameraX lifecycle and the most-recent frame.
 *
 * Capture is one-shot per tap (`captureOneFrame()`), not continuous:
 * the bound analyzer continuously updates [latestFrame] under
 * `STRATEGY_KEEP_ONLY_LATEST`, and Capture takes the buffer atomically,
 * builds a `Shared.ImageInput`, and clears the slot so the next tap
 * waits for a fresh frame rather than re-classifying the same one.
 *
 * Portrait-only per spec §11. `AndroidManifest.xml` pins the activity
 * to `android:screenOrientation="portrait"`, so the analyzer's natural
 * orientation matches.
 *
 * **YUV vs RGBA path:** the analyzer is configured with
 * [ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888] so frames arrive as a
 * single planar RGBA buffer — drop the alpha channel and we have the
 * tight-packed RGB the shared `ImageInput` expects. This avoids the
 * manual YUV→RGB matrix math that the YUV_420_888 path requires. On
 * devices that report no support for the RGBA format the builder falls
 * back to YUV; that legacy path is not exercised in v1 and a fallback
 * conversion is deliberately out of scope (would land in Phase 15
 * polish — devices that ship without RGBA output are uncommon on the
 * spec's minSdk = 36 target).
 *
 * **Emulator caveat:** the Android emulator can vend synthetic camera
 * frames, but the spec's local development setup does not have one
 * configured. End-to-end Capture is exercised on real hardware (see
 * `docs/MANUAL_TEST_PLAN.md` flow 4). Unit tests stay off the camera
 * path and only exercise the surrounding state machine
 * (`LiveOverrideStateTest`).
 */
class CameraService(
    private val context: Context,
    @Suppress("unused") private val scope: CoroutineScope,
) {

    sealed interface State {
        data object Idle : State
        data object Starting : State
        data object Running : State
        data object Stopped : State
        data class Failed(val message: String) : State
    }

    sealed class CameraError(message: String) : RuntimeException(message) {
        data object SessionNotRunning : CameraError("Camera session is not running.")
        data object CaptureTimeout : CameraError(
            "Capture timed out before a frame was available. Make sure the camera preview is running."
        )
        data class ProviderFailed(val reason: String) :
            CameraError("Camera provider failed: $reason")
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    /**
     * Preview use case. Lazy so JVM unit tests can instantiate the
     * service without CameraX classes being initialised — the unit
     * test for `captureOneFrame()` before `start()` exercises only the
     * state-machine guard, never the camera graph. Constructed once
     * and re-bound on each `start`.
     */
    private val preview: Preview by lazy { Preview.Builder().build() }

    private val imageAnalysis: ImageAnalysis by lazy {
        ImageAnalysis.Builder()
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
    }

    private var cameraProvider: ProcessCameraProvider? = null
    private val frameStore = LatestFrameStore()
    private val configLock = Mutex()
    private var analyzerInstalled = false

    /**
     * Wire the analyzer that forwards each frame to the lock-protected
     * store. Pulled out of the constructor so JVM unit tests can
     * instantiate the service without touching `Context`-bound
     * executors — the unit test harness's stub `android.jar` doesn't
     * vend a real `MainLooper`. Called once from [start].
     */
    private fun installAnalyzerIfNeeded() {
        if (analyzerInstalled) return
        imageAnalysis.setAnalyzer(
            ContextCompat.getMainExecutor(context),
        ) { proxy ->
            frameStore.set(proxy)
        }
        analyzerInstalled = true
    }

    // MARK: - Lifecycle

    /**
     * Bind the Preview + ImageAnalysis use cases to [lifecycleOwner].
     * Idempotent — repeated calls while already running are no-ops.
     * Throws [CameraError.ProviderFailed] if [ProcessCameraProvider]
     * initialisation fails.
     */
    suspend fun start(lifecycleOwner: LifecycleOwner) {
        val current = _state.value
        if (current is State.Running || current is State.Starting) return
        _state.value = State.Starting

        try {
            installAnalyzerIfNeeded()
            val provider = awaitCameraProvider()
            cameraProvider = provider
            withContext(Dispatchers.Main) {
                provider.unbindAll()
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis,
                )
            }
            _state.value = State.Running
        } catch (e: Throwable) {
            _state.value = State.Failed(e.message ?: e::class.simpleName ?: "unknown")
            throw if (e is CameraError) e else CameraError.ProviderFailed(e.message ?: "init")
        }
    }

    /**
     * Unbind every use case from the camera provider. Safe to call
     * from any thread — dispatches the actual unbind to Main. The
     * provider itself is retained for the next `start()`.
     */
    fun stop() {
        val provider = cameraProvider
        if (provider == null) {
            _state.value = State.Stopped
            return
        }
        // bindToLifecycle / unbindAll must run on Main per CameraX docs.
        ContextCompat.getMainExecutor(context).execute {
            try {
                provider.unbindAll()
            } catch (_: Throwable) {
                // CameraX raises if called pre-init; we've already
                // surfaced Failed in start(), nothing more to do.
            }
            frameStore.clear()
            _state.value = State.Stopped
        }
    }

    /**
     * Capture the next available frame from the running analyzer.
     * Polls the lock-protected store for up to ~2 seconds. The
     * STRATEGY_KEEP_ONLY_LATEST analyzer typically refreshes the slot
     * every 33 ms (30 fps), so the first poll usually succeeds.
     *
     * @throws CameraError.SessionNotRunning if `start()` hasn't run.
     * @throws CameraError.CaptureTimeout if no frame arrives in 2 s.
     */
    suspend fun captureOneFrame(): ImageInput {
        if (_state.value !is State.Running) {
            throw CameraError.SessionNotRunning
        }

        val deadlineMs = System.currentTimeMillis() + 2_000L
        while (System.currentTimeMillis() < deadlineMs) {
            val image = frameStore.takeAndBuild()
            if (image != null) return image
            // ~30 ms ≈ one frame at 30 fps.
            kotlinx.coroutines.delay(30L)
        }
        throw CameraError.CaptureTimeout
    }

    /**
     * Wire a `PreviewView`'s `SurfaceProvider` into the Preview use
     * case. Called from the [com.malaria.android.ui.home.CameraPreview]
     * composable's `AndroidView` update block on each recomposition;
     * the underlying `setSurfaceProvider` call is idempotent.
     */
    fun attachPreview(provider: Preview.SurfaceProvider) {
        preview.setSurfaceProvider(provider)
    }

    // MARK: - Provider plumbing

    /**
     * Suspend wrapper around [ProcessCameraProvider.getInstance],
     * which returns a `ListenableFuture`. The project does not depend
     * on `kotlinx-coroutines-guava` (which would let us call
     * `.await()`), so we hand-roll the listener with
     * [suspendCancellableCoroutine].
     */
    private suspend fun awaitCameraProvider(): ProcessCameraProvider = configLock.withLock {
        suspendCancellableCoroutine { cont ->
            val future = ProcessCameraProvider.getInstance(context)
            future.addListener(
                {
                    try {
                        cont.resume(future.get())
                    } catch (e: Throwable) {
                        cont.resumeWithException(
                            CameraError.ProviderFailed(e.message ?: "future"),
                        )
                    }
                },
                ContextCompat.getMainExecutor(context),
            )
            cont.invokeOnCancellation {
                // ListenableFuture cancellation is best-effort; the
                // listener above will simply see a thrown exception.
                future.cancel(false)
            }
        }
    }
}

// MARK: - Latest frame store

/**
 * Lock-protected box that owns the most-recent [ImageProxy] the
 * analyzer has produced. Single-shot semantics: [takeAndBuild]
 * consumes the stored frame, builds an [ImageInput], closes the
 * proxy, and clears the slot atomically.
 *
 * Closing the proxy is required: CameraX uses a small fixed pool of
 * buffers; leaving an `ImageProxy` un-closed back-pressures the
 * analyzer and frames stop arriving.
 */
internal class LatestFrameStore {

    private val lock = Any()
    private var latest: ImageProxy? = null

    fun set(proxy: ImageProxy) {
        val previous: ImageProxy?
        synchronized(lock) {
            previous = latest
            latest = proxy
        }
        // Close the displaced frame so the analyzer's buffer pool
        // recycles it. KEEP_ONLY_LATEST means we drop intermediates.
        previous?.close()
    }

    fun clear() {
        val previous: ImageProxy?
        synchronized(lock) {
            previous = latest
            latest = null
        }
        previous?.close()
    }

    /**
     * Atomically take the most-recent frame, build an [ImageInput]
     * from it, close the proxy, and return the input. Returns `null`
     * if no frame is queued.
     *
     * The RGBA_8888 path reads directly from `ImageProxy.planes[0]`
     * (no `proxy.image` access), so the `@ExperimentalGetImage`
     * opt-in is not required here.
     */
    fun takeAndBuild(): ImageInput? {
        val proxy: ImageProxy?
        synchronized(lock) {
            proxy = latest
            latest = null
        }
        if (proxy == null) return null
        return try {
            ImageInputBuilder.makeImageInput(proxy)
        } catch (e: Throwable) {
            Log.w("CameraService", "Failed to build ImageInput from proxy", e)
            null
        } finally {
            proxy.close()
        }
    }
}

// MARK: - Pixel buffer -> ImageInput

/**
 * Pulled out as its own type so [LiveOverrideStateTest] (and future
 * Phase 15 hardening tests) can exercise the RGBA → RGB packing
 * without spinning up CameraX. The path drops the alpha channel from
 * the RGBA_8888 buffer the analyzer is configured to produce.
 */
internal object ImageInputBuilder {

    fun makeImageInput(proxy: ImageProxy): ImageInput {
        // RGBA_8888 → single-plane buffer; rowStride may be padded
        // beyond width*4 on some devices, so honour it.
        val plane = proxy.planes[0]
        val buffer: ByteBuffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val width = proxy.width
        val height = proxy.height

        val out = ByteArray(width * height * 3)
        val rowBytes = ByteArray(rowStride)
        var outIdx = 0
        for (row in 0 until height) {
            buffer.position(row * rowStride)
            // Read one row at a time to avoid touching the padding bytes
            // beyond `width * pixelStride`. Some buffers are direct so a
            // bulk get is cheaper than per-byte reads.
            val toRead = (width * pixelStride).coerceAtMost(rowStride)
            buffer.get(rowBytes, 0, toRead)
            var col = 0
            while (col < width) {
                val i = col * pixelStride
                out[outIdx] = rowBytes[i]         // R
                out[outIdx + 1] = rowBytes[i + 1] // G
                out[outIdx + 2] = rowBytes[i + 2] // B
                outIdx += 3
                col += 1
            }
        }

        return ImageInput(
            rgbBytes = out,
            width = width,
            height = height,
        )
    }
}
