package com.malaria.android.ui.screens

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Shown when the device has been admin-provisioned but no microscopist has
 * claimed it yet (spec §10, Phase 2 of onboarding). Phase 7 implements the
 * claim flow.
 */
@Composable
fun ProvisioningIncompleteView() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 16.dp, end = 16.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text("Provisioning incomplete — Phase 8")
    }
}
