package com.malaria.android.ui.screens

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.malaria.android.R

/**
 * Shown on Home and History tabs when [com.malaria.android.services.AuthGate]
 * reports `Locked`. Phase 8 replaces the static text with the unlock flow.
 */
@Composable
fun LockedPlaceholder() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(start = 16.dp, end = 16.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(stringResource(R.string.tap_to_unlock) + " — Phase 8")
    }
}
