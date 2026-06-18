package com.malaria.android.ui.onboarding.components

import android.content.Context

/**
 * Loader for the two legal-text blobs shown during onboarding (spec §10
 * steps 3 and 4 — Hippocratic License acknowledgement and medical-device
 * disclaimer).
 *
 * [licenseBody] reads `LICENSE.txt` from Android assets (copied into
 * `androidApp/src/main/assets/` at build time). [disclaimerBody] mirrors
 * `NOTICE` verbatim.
 *
 * Stored versions match what the wizard passes to
 * `OnboardingState.acceptHippocraticLicense(version)` /
 * `acceptMedicalDisclaimer(version)` — bumping these constants and the
 * passed-in version string in lockstep lets the consent table answer
 * "did this clinician accept version X" queries cleanly.
 *
 * Mirrors `iosApp/Views/Onboarding/Components/LegalTextLoader.swift`
 * value-for-value.
 */
object LegalText {

    const val LICENSE_VERSION = "v3.0"
    const val DISCLAIMER_VERSION = "v1.0"

    fun licenseBody(context: Context): String = try {
        context.assets.open("LICENSE.txt").bufferedReader().use { it.readText() }
    } catch (_: Exception) {
        "Hippocratic License 3.0 (HL3-FULL) — see LICENSE at the project root."
    }

    const val DISCLAIMER_BODY: String = """Malaria Detector — NOTICE

This software is provided for research and educational purposes only. It is NOT certified as a medical device under FDA SaMD, EU MDR 2017/745, the Kenya Health Act medical-devices regulations, or any other regulatory framework. It must NOT be used as the basis for clinical diagnostic decisions without conformance assessment by the deploying party under their local regulations. The authors and contributors disclaim all liability for clinical use. Deployers assume full responsibility for regulatory compliance, patient safety, and clinical validation in their jurisdiction."""
}
