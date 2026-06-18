package com.malaria.android.ui.screens

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.data.entities.Prediction
import com.malaria.android.services.AuthGate
import com.malaria.android.services.CameraService
import com.malaria.android.ui.history.components.RiskBandIndicator
import com.malaria.android.ui.home.CameraPreview
import com.malaria.android.ui.locals.LocalAuthGate
import com.malaria.android.ui.locals.LocalCameraService
import com.malaria.android.ui.locals.LocalClassifier
import com.malaria.android.ui.locals.LocalPredictionStore
import com.malaria.android.ui.override.LiveOverrideSheet
import com.malaria.util.Threshold
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

/**
 * Active-screening surface (spec §11 — Home tab). Phase 8 Android.
 * Mirror of `iosApp/Views/ActiveScreeningView.swift` field-for-field.
 *
 * Owns the per-tap capture → classify → persist sequence per spec §6:
 *   1. Tap Capture launches a coroutine on [rememberCoroutineScope].
 *   2. [CameraService.captureOneFrame] consumes the latest
 *      [androidx.camera.core.ImageProxy] from the bound analyzer and
 *      returns a [com.malaria.domain.ImageInput] at native dims.
 *   3. [com.malaria.android.services.ClassifierService.classify] runs
 *      TFLite inference against the bundled BNLeaky Keras model.
 *   4. [com.malaria.android.services.PredictionStore.record] persists
 *      the Room entity + writes the `prediction_created` audit entry.
 *
 * Lifecycle:
 *   - [DisposableEffect] stops the camera on dispose.
 *   - [LifecycleEventObserver] stops on `ON_PAUSE` (spec §11
 *     backgrounding ends active screening).
 *   - [AuthGate.State.Locked] also stops the camera and resets to idle.
 *
 * Permissions:
 *   - On first composition the CAMERA permission is requested via
 *     [rememberLauncherForActivityResult].
 *   - If the user denies the permission the [PermissionDeniedView]
 *     fallback opens app settings via
 *     [Settings.ACTION_APPLICATION_DETAILS_SETTINGS].
 *
 * **Emulator caveat (spec / known limitations):** the project doesn't
 * have an Android emulator configured locally, so the Capture path is
 * verified by build only on this machine. Real-device verification is
 * the Phase 15 manual-test-plan step.
 */
@Composable
fun ActiveScreeningView(
    @Suppress("UNUSED_PARAMETER") recentPrediction: Any?,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraService = LocalCameraService.current
    val classifierService = LocalClassifier.current
    val predictionStore = LocalPredictionStore.current
    val authGate = LocalAuthGate.current
    val authState by authGate.state.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()

    var permissionGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED,
        )
    }
    var permissionRequested by remember { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
    ) { granted ->
        permissionGranted = granted
        permissionRequested = true
    }

    LaunchedEffect(Unit) {
        if (!permissionGranted && !permissionRequested) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (!permissionGranted) {
        PermissionDeniedView(
            requested = permissionRequested,
            onRequest = { permissionLauncher.launch(Manifest.permission.CAMERA) },
        )
        return
    }

    var localState by remember { mutableStateOf<LocalState>(LocalState.Idle) }
    var lastPersisted by remember { mutableStateOf<Prediction?>(null) }
    var showOverride by remember { mutableStateOf(false) }
    var started by remember { mutableStateOf(false) }

    // Start the camera once permission is granted; stop on dispose.
    LaunchedEffect(permissionGranted) {
        if (permissionGranted && !started) {
            started = true
            runCatching { cameraService.start(lifecycleOwner) }
                .onFailure { error ->
                    localState = LocalState.Error(
                        error.localizedMessage ?: error::class.simpleName ?: "Camera failed.",
                    )
                }
        }
    }

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_PAUSE) {
                // Spec §11 — backgrounding ends active screening.
                cameraService.stop()
                localState = LocalState.Idle
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            cameraService.stop()
        }
    }

    LaunchedEffect(authState) {
        if (authState is AuthGate.State.Locked) {
            cameraService.stop()
            localState = LocalState.Idle
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 16.dp, end = 16.dp),
    ) {
        ModelBadge(modifier = Modifier.padding(top = 12.dp))

        Spacer(modifier = Modifier.height(8.dp))

        // Preview area — fills the centre with a clipped, black-backed
        // rounded box so the camera image inherits the same affordance
        // as iOS's UIViewRepresentable inside a RoundedRectangle.
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .clip(RoundedCornerShape(16.dp))
                .background(Color.Black),
            contentAlignment = Alignment.Center,
        ) {
            when (val s = cameraService.state.collectAsStateWithLifecycle().value) {
                is CameraService.State.Running, is CameraService.State.Starting -> {
                    CameraPreview(
                        cameraService = cameraService,
                        modifier = Modifier.fillMaxSize(),
                    )
                }
                is CameraService.State.Failed -> {
                    Text(
                        text = s.message,
                        color = Color.White,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(start = 24.dp, end = 24.dp),
                    )
                }
                else -> {
                    Text(
                        text = "Camera idle",
                        color = Color.White.copy(alpha = 0.7f),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Controls(
            localState = localState,
            cameraRunning = cameraService.state.collectAsStateWithLifecycle().value
                is CameraService.State.Running,
            onCapture = {
                scope.launch {
                    localState = LocalState.Capturing
                    val outcome = runCatching {
                        val image = cameraService.captureOneFrame()
                        val classified = classifierService.classify(image)
                        val raw = classified.getOrThrow()
                        predictionStore.record(
                            raw = raw,
                            threshold = Threshold.DEFAULT.toDouble(),
                        )
                    }
                    outcome.onSuccess { persisted ->
                        lastPersisted = persisted
                        localState = LocalState.Showing(
                            predictionId = persisted.id,
                            label = persisted.label,
                            parasitizedProb = persisted.parasitizedProb,
                        )
                    }.onFailure { e ->
                        localState = LocalState.Error(
                            e.localizedMessage ?: e::class.simpleName ?: "unknown",
                        )
                    }
                }
            },
            onOverride = { showOverride = true },
            onEndSession = {
                cameraService.stop()
                lastPersisted = null
                localState = LocalState.Idle
            },
        )

        Spacer(modifier = Modifier.height(24.dp))
    }

    if (showOverride) {
        lastPersisted?.let { snapshot ->
            LiveOverrideSheet(
                prediction = snapshot,
                onDismiss = { showOverride = false },
                predictionStore = predictionStore,
            )
        }
    }
}

@Composable
private fun ModelBadge(modifier: Modifier = Modifier) {
    // Phase 11 Settings → Models adds the live picker. Until then the
    // bundled BNLeaky Keras model is the only choice — match the
    // affordance label spec §11 mocks describe.
    Row(
        modifier = modifier,
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Icon(
            imageVector = Icons.Filled.Star,
            contentDescription = null,
            tint = Color(0xFFE6B800),
            modifier = Modifier.height(16.dp),
        )
        Text(
            text = "BN + LeakyReLU",
            style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Medium),
        )
    }
}

@Composable
private fun Controls(
    localState: LocalState,
    cameraRunning: Boolean,
    onCapture: () -> Unit,
    onOverride: () -> Unit,
    onEndSession: () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        if (localState is LocalState.Showing) {
            PredictionOverlay(label = localState.label, parasitizedProb = localState.parasitizedProb)
        }
        if (localState is LocalState.Error) {
            Text(
                text = localState.message,
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.error,
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            if (localState is LocalState.Showing) {
                OutlinedButton(
                    onClick = onOverride,
                    modifier = Modifier.weight(1f),
                ) {
                    Text("Override")
                }
            }
            Button(
                onClick = onCapture,
                enabled = cameraRunning && localState !is LocalState.Capturing,
                modifier = Modifier.weight(1f),
            ) {
                Text(
                    text = if (localState is LocalState.Capturing) "Capturing…" else "Capture",
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                )
            }
        }
        if (localState is LocalState.Showing || localState is LocalState.Error) {
            TextButton(
                onClick = onEndSession,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("End session")
            }
        }
    }
}

@Composable
private fun PredictionOverlay(label: String, parasitizedProb: Double) {
    val percent = (parasitizedProb * 100).roundToInt()
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(start = 14.dp, end = 14.dp, top = 10.dp, bottom = 10.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
            )
            Text(
                text = "$percent%",
                style = MaterialTheme.typography.titleMedium,
            )
        }
        RiskBandIndicator(parasitizedProb = parasitizedProb)
    }
}

@Composable
private fun PermissionDeniedView(requested: Boolean, onRequest: () -> Unit) {
    val context = LocalContext.current
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 24.dp, end = 24.dp, top = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = "Camera permission required",
            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
        )
        Text(
            text = "Open Settings → Malaria Detector → Permissions to grant camera access.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        if (!requested) {
            Button(onClick = onRequest, modifier = Modifier.fillMaxWidth()) {
                Text("Grant camera permission")
            }
        }
        OutlinedButton(
            onClick = {
                val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                    data = Uri.fromParts("package", context.packageName, null)
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                context.startActivity(intent)
            },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Open Settings")
        }
    }
}

// MARK: - Local state

/**
 * Per-screen state machine. Mirrors the iOS
 * `ActiveScreeningView.LocalState` cases. Pulled out as a sealed
 * interface so JVM tests can target the transitions without touching
 * Compose.
 */
sealed interface LocalState {
    data object Idle : LocalState
    data object Capturing : LocalState
    data class Showing(
        val predictionId: String,
        val label: String,
        val parasitizedProb: Double,
    ) : LocalState
    data class Error(val message: String) : LocalState
}
