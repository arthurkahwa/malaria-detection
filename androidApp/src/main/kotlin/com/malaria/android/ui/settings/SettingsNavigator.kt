package com.malaria.android.ui.settings

import androidx.compose.runtime.Composable
import androidx.compose.runtime.Stable
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.remember

/**
 * Minimal back-stack manager for the Settings tab. Same pattern as
 * `HistoryNavigator`.
 */
@Stable
class SettingsNavigator(initial: SettingsDestination = SettingsDestination.Root) {

    private val stack = mutableStateListOf(initial)

    val current: SettingsDestination
        get() = stack.last()

    val canGoBack: Boolean
        get() = stack.size > 1

    fun push(destination: SettingsDestination) {
        stack.add(destination)
    }

    fun pop() {
        if (stack.size > 1) stack.removeAt(stack.size - 1)
    }

    fun popToRoot() {
        while (stack.size > 1) stack.removeAt(stack.size - 1)
    }
}

@Composable
fun rememberSettingsNavigator(): SettingsNavigator = remember { SettingsNavigator() }
