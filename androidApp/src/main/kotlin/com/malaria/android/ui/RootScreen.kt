package com.malaria.android.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import com.malaria.android.R
import com.malaria.android.ui.screens.AboutScreen
import com.malaria.android.ui.screens.HistoryScreen
import com.malaria.android.ui.screens.HomeScreen
import com.malaria.android.ui.screens.SettingsScreen

/**
 * Four-tab bottom navigation per spec §11. Home + History will be locked
 * behind [com.malaria.android.services.AuthGate] from Phase 7 onward; until
 * then the placeholder screens render unconditionally.
 */
@Composable
fun RootScreen() {
    var selected by remember { mutableIntStateOf(0) }

    val tabs = listOf(
        Tab(stringResource(R.string.tab_home), Icons.Default.Home),
        Tab(stringResource(R.string.tab_history), Icons.Default.History),
        Tab(stringResource(R.string.tab_settings), Icons.Default.Settings),
        Tab(stringResource(R.string.tab_about), Icons.Default.Info),
    )

    Scaffold(
        bottomBar = {
            NavigationBar {
                tabs.forEachIndexed { index, tab ->
                    NavigationBarItem(
                        selected = selected == index,
                        onClick = { selected = index },
                        icon = { Icon(tab.icon, contentDescription = tab.label) },
                        label = { Text(tab.label) },
                    )
                }
            }
        },
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            when (selected) {
                0 -> HomeScreen()
                1 -> HistoryScreen()
                2 -> SettingsScreen()
                3 -> AboutScreen()
            }
        }
    }
}

private data class Tab(val label: String, val icon: ImageVector)
