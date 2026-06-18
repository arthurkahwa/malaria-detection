package com.malaria.android.ui.history

import androidx.compose.runtime.Composable
import androidx.compose.runtime.Stable
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.remember

/**
 * Minimal back-stack manager for the History tab.
 *
 * The composables push by appending and pop by removing the tail. The
 * `current` getter returns the top destination; `canGoBack` is used by
 * the Scaffold's top-bar back button to mirror iOS's `.navigationBar`
 * leading button behaviour.
 *
 * Out of scope: deep-linking, restore-from-savedInstanceState. Phase 11
 * can replace this with `androidx.navigation.compose` if multi-tab deep
 * linking becomes a requirement.
 */
@Stable
class HistoryNavigator(initial: HistoryDestination = HistoryDestination.Subsections) {

    private val stack = mutableStateListOf(initial)

    val current: HistoryDestination
        get() = stack.last()

    val canGoBack: Boolean
        get() = stack.size > 1

    fun push(destination: HistoryDestination) {
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
fun rememberHistoryNavigator(): HistoryNavigator = remember { HistoryNavigator() }
