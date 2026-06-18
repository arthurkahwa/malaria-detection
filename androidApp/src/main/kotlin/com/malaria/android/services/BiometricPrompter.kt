package com.malaria.android.services

import androidx.biometric.BiometricManager.Authenticators.BIOMETRIC_STRONG
import androidx.biometric.BiometricManager.Authenticators.DEVICE_CREDENTIAL
import androidx.biometric.BiometricPrompt
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import kotlin.coroutines.resume
import kotlinx.coroutines.suspendCancellableCoroutine

/**
 * Suspend wrapper around [androidx.biometric.BiometricPrompt] (spec §9, §19).
 *
 * `BIOMETRIC_STRONG | DEVICE_CREDENTIAL` allows biometric OR PIN/passcode
 * fallback. The two biometric steps in onboarding are *enrolment* — proving
 * the user can authenticate — so device-credential fallback is the correct
 * permissive policy. Routine post-onboarding unlocks (handled elsewhere)
 * may tighten this if a deployer requires biometric-only.
 *
 * `onAuthenticationFailed` is deliberately not resumed — the OS allows the
 * user to retry a few times before surfacing an error; resuming the
 * continuation on a single failed try would short-circuit that affordance.
 */
class BiometricPrompter(private val activity: FragmentActivity) {

    sealed interface Outcome {
        data object Success : Outcome
        data class Failure(val reason: String) : Outcome
        data object Cancelled : Outcome
    }

    suspend fun prompt(
        title: String,
        subtitle: String?,
        negativeButtonText: String = "Cancel",
    ): Outcome = suspendCancellableCoroutine { cont ->
        val info = BiometricPrompt.PromptInfo.Builder()
            .setTitle(title)
            .setSubtitle(subtitle ?: "")
            .setAllowedAuthenticators(BIOMETRIC_STRONG or DEVICE_CREDENTIAL)
            .build()

        val prompt = BiometricPrompt(
            activity,
            ContextCompat.getMainExecutor(activity.applicationContext),
            object : BiometricPrompt.AuthenticationCallback() {
                override fun onAuthenticationSucceeded(result: BiometricPrompt.AuthenticationResult) {
                    if (cont.isActive) cont.resume(Outcome.Success)
                }

                override fun onAuthenticationFailed() {
                    // User tried, failed once — let the OS surface a retry.
                    // Do not resume; the next callback (success or error)
                    // will resolve the continuation.
                }

                override fun onAuthenticationError(code: Int, message: CharSequence) {
                    if (!cont.isActive) return
                    val outcome = when (code) {
                        BiometricPrompt.ERROR_USER_CANCELED,
                        BiometricPrompt.ERROR_CANCELED,
                        BiometricPrompt.ERROR_NEGATIVE_BUTTON -> Outcome.Cancelled
                        else -> Outcome.Failure(message.toString())
                    }
                    cont.resume(outcome)
                }
            },
        )

        prompt.authenticate(info)
        cont.invokeOnCancellation { prompt.cancelAuthentication() }
    }
}
