package com.malaria.android.ui.home

import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.malaria.android.services.CameraService

/**
 * Compose wrapper for CameraX's [PreviewView].
 *
 * Mirror of `iosApp/Views/Home/CameraPreviewView.swift`. The host
 * [CameraService] owns the `Preview` use case and the underlying
 * `SurfaceProvider`; this composable just inflates a [PreviewView]
 * and hands its `surfaceProvider` to the service on every update.
 * The service's `setSurfaceProvider` is idempotent, so recompositions
 * don't churn the camera graph.
 *
 * `PreviewView.ScaleType.FILL_CENTER` matches iOS's
 * `videoGravity = .resizeAspectFill` — fills the bounds and crops
 * overflow rather than letterboxing.
 */
@Composable
fun CameraPreview(
    cameraService: CameraService,
    modifier: Modifier = Modifier,
) {
    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            PreviewView(ctx).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
        },
        update = { previewView ->
            cameraService.attachPreview(previewView.surfaceProvider)
        },
    )
}
