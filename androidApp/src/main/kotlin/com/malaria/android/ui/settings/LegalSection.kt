package com.malaria.android.ui.settings

import androidx.compose.runtime.Composable
import com.malaria.android.ui.onboarding.components.LegalText

/**
 * Legal section (spec §11). Privacy + ToS are placeholder notes (v1 has
 * no in-app copies). Disclaimer + OSS acknowledgements open in-app
 * readers.
 */
@Composable
fun LegalSection(navigator: SettingsNavigator) {
    SectionScaffold(header = "Legal") {
        EditableRow(
            label = "Privacy policy",
            value = "",
            enabled = true,
            onClick = {
                navigator.push(
                    SettingsDestination.LegalDocument(
                        title = "Privacy policy",
                        body = "The full privacy policy is published at the project's source-code repository. v1 ships without an in-app copy because the wording is finalised by each deploying clinic.",
                    ),
                )
            },
        )
        EditableRow(
            label = "Terms of service",
            value = "",
            enabled = true,
            onClick = {
                navigator.push(
                    SettingsDestination.LegalDocument(
                        title = "Terms of service",
                        body = "The terms of service are governed by the Hippocratic License 3.0 (HL3-FULL). The full text is shown in the Decision-support disclaimer below and in the About tab.",
                    ),
                )
            },
        )
        EditableRow(
            label = "Decision-support disclaimer",
            value = "",
            enabled = true,
            onClick = {
                navigator.push(
                    SettingsDestination.LegalDocument(
                        title = "Decision-support disclaimer",
                        body = LegalText.DISCLAIMER_BODY,
                    ),
                )
            },
        )
        EditableRow(
            label = "Open-source acknowledgements",
            value = "",
            enabled = true,
            onClick = {
                navigator.push(
                    SettingsDestination.LegalDocument(
                        title = "Open-source acknowledgements",
                        body = "This software is built on open-source components including SwiftUI, SwiftData, Core ML, Kotlin Multiplatform, Compose, Room, SQLCipher, and the Hippocratic License. Full attribution travels with each release.",
                    ),
                )
            },
        )
    }
}
