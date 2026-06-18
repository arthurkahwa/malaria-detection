package com.malaria.android.ui.onboarding.microscopist

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.background
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.onboarding.components.WizardStepContainer
import kotlinx.coroutines.launch

/**
 * Phase 2 step 4 (spec §10): three-page orientation. Skippable. Last page
 * advances to operational state via `OnboardingState.finishOrientation()`.
 *
 * Mirrors `iosApp/Views/Onboarding/Microscopist/OrientationStep.swift`.
 */
@Composable
fun OrientationStep() {
    val onboarding = LocalOnboardingState.current
    val scope = rememberCoroutineScope()

    val pages = listOf(
        OrientationPage(
            icon = Icons.Filled.PhotoCamera,
            title = "How to capture",
            body = "Tap the Capture button on the Home tab to take a photo through the microscope. The app classifies each cell in 50–300 ms and shows the result inline.",
        ),
        OrientationPage(
            icon = Icons.Filled.Warning,
            title = "How to override",
            body = "If you disagree with the model, tap Override next to the prediction. Pick the corrected verdict and a reason. The original prediction is preserved in the audit log.",
        ),
        OrientationPage(
            icon = Icons.Filled.Lock,
            title = "How to lock",
            body = "Tap the lock icon in the corner — or just background the app — to lock the session. Re-unlock with your biometric. Auto-logout fires after the timeout the admin configured.",
        ),
    )

    val pagerState = rememberPagerState(pageCount = { pages.size })
    val isLastPage = pagerState.currentPage == pages.lastIndex

    WizardStepContainer(
        title = "Quick orientation",
        subtitle = "Three short cards. Swipe to navigate or skip to the end.",
        stepIndicator = "Step 4 of 4",
        primaryLabel = if (isLastPage) "Begin screening" else "Next",
        onPrimary = {
            if (isLastPage) {
                onboarding.finishOrientation()
            } else {
                scope.launch { pagerState.animateScrollToPage(pagerState.currentPage + 1) }
            }
        },
        secondaryLabel = "Skip orientation",
        onSecondary = { onboarding.finishOrientation() },
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(360.dp),
        ) {
            HorizontalPager(
                state = pagerState,
                modifier = Modifier.fillMaxSize(),
            ) { pageIndex ->
                OrientationPageCard(page = pages[pageIndex])
            }
        }

        // Dot indicator
        androidx.compose.foundation.layout.Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 8.dp),
            horizontalArrangement = Arrangement.Center,
        ) {
            for (idx in pages.indices) {
                val color = if (pagerState.currentPage == idx) {
                    MaterialTheme.colorScheme.primary
                } else {
                    MaterialTheme.colorScheme.outlineVariant
                }
                Box(
                    modifier = Modifier
                        .padding(horizontal = 4.dp)
                        .background(color, CircleShape)
                        .height(8.dp)
                        .width(8.dp),
                )
            }
        }
    }
}

private data class OrientationPage(
    val icon: ImageVector,
    val title: String,
    val body: String,
)

@Composable
private fun OrientationPageCard(page: OrientationPage) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 4.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.Start,
    ) {
        Icon(
            imageVector = page.icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier
                .padding(top = 16.dp)
                .height(64.dp),
        )
        Text(
            text = page.title,
            style = MaterialTheme.typography.titleLarge,
        )
        Text(
            text = page.body,
            style = MaterialTheme.typography.bodyMedium,
        )
    }
}
