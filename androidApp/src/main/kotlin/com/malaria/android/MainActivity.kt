package com.malaria.android

import android.app.ActivityManager
import android.app.KeyguardManager
import android.content.Context
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.malaria.android.BuildConfig
import com.malaria.android.data.BuildEnvironment
import com.malaria.android.data.LanguagePreference
import com.malaria.android.data.MalariaDatabase
import com.malaria.android.services.AuditLog
import com.malaria.android.services.AuthGate
import com.malaria.android.services.CameraService
import com.malaria.android.services.ClassifierService
import com.malaria.android.services.CrashLogStore
import com.malaria.android.services.CrashLogWriter
import com.malaria.android.services.ExportService
import com.malaria.android.services.ModelDownloadService
import com.malaria.android.services.ModelRegistryService
import com.malaria.android.services.OnboardingState
import com.malaria.android.services.PredictionStore
import com.malaria.android.services.ResetDeviceCoordinator
import com.malaria.android.services.SettingsStore
import com.malaria.android.ui.RootScreen
import com.malaria.android.ui.locals.LocalAuditLog
import com.malaria.android.ui.locals.LocalAuthGate
import com.malaria.android.ui.locals.LocalCameraService
import com.malaria.android.ui.locals.LocalClassifier
import com.malaria.android.ui.locals.LocalCrashLogStore
import com.malaria.android.ui.locals.LocalDatabase
import com.malaria.android.ui.locals.LocalModelDownloadService
import com.malaria.android.ui.locals.LocalExportService
import com.malaria.android.ui.locals.LocalModelRegistry
import com.malaria.android.ui.locals.LocalOnboardingState
import com.malaria.android.ui.locals.LocalPredictionStore
import com.malaria.android.ui.locals.LocalResetDeviceCoordinator
import com.malaria.android.ui.locals.LocalSettingsStore
import com.malaria.android.ui.onboarding.OnboardingFlow
import com.malaria.android.ui.theme.MalariaTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Composition root for the Android app.
 *
 * Per spec §19, services are constructed here once and live for the
 * activity's lifetime. Composables consume them via the CompositionLocal
 * keys in [com.malaria.android.ui.locals].
 *
 * Extends [FragmentActivity] (not `ComponentActivity`) because Phase 7
 * uses `androidx.biometric.BiometricPrompt`, which requires a
 * `FragmentActivity` host. `FragmentActivity` includes all the lifecycle
 * + Compose hooks that `ComponentActivity` does; `activity-compose`'s
 * `setContent {}` works the same.
 */
class MainActivity : FragmentActivity() {

    private val uiScope: CoroutineScope by lazy {
        CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    }
    private val computeScope: CoroutineScope by lazy {
        CoroutineScope(SupervisorJob() + Dispatchers.Default)
    }

    private val database by lazy { MalariaDatabase.create(applicationContext) }

    private val classifier by lazy {
        ClassifierService(modelId = "BNLeaky_Keras", scope = computeScope)
    }
    private val cameraService by lazy {
        CameraService(context = applicationContext, scope = computeScope)
    }
    private val modelRegistry by lazy {
        ModelRegistryService(context = applicationContext, scope = computeScope)
    }
    private val auditLog by lazy { AuditLog(dao = database.auditDao()) }
    private val authGate by lazy {
        AuthGate(
            clinicians = database.clinicianDao(),
            audit = auditLog,
            scope = uiScope,
        )
    }
    private val predictionStore by lazy {
        PredictionStore(
            dao = database.predictionDao(),
            audit = auditLog,
            clinicians = database.clinicianDao(),
            scope = uiScope,
        )
    }
    private val onboardingState by lazy {
        OnboardingState(
            clinicians = database.clinicianDao(),
            consents = database.consentDao(),
            audit = auditLog,
        )
    }
    private val languagePreference by lazy { LanguagePreference(applicationContext) }
    private val settingsStore by lazy {
        SettingsStore(
            auditDao = database.auditDao(),
            audit = auditLog,
            clinicians = database.clinicianDao(),
            languagePreference = languagePreference,
        )
    }
    private val resetDeviceCoordinator by lazy {
        ResetDeviceCoordinator(
            clinicians = database.clinicianDao(),
            audit = auditLog,
            onboarding = onboardingState,
            settings = settingsStore,
        )
    }
    private val exportService by lazy {
        ExportService(
            context = applicationContext,
            predictions = database.predictionDao(),
            audits = database.auditDao(),
            clinicians = database.clinicianDao(),
            consents = database.consentDao(),
            settings = settingsStore,
            auditLog = auditLog,
        )
    }
    private val crashLogStore by lazy {
        CrashLogStore(context = applicationContext, auditLog = auditLog)
    }
    private val modelDownloadService by lazy {
        ModelDownloadService(context = applicationContext, scope = uiScope)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Phase 14 (spec §16): install the JVM uncaught-exception handler
        // before any other work so the very first crash is captured. The
        // environment is computed once here so the handler doesn't touch
        // BuildConfig / Build / package manager during the crash path.
        // Activity context is passed via the app-context hook because
        // `EncryptedFile` builds against a Context.
        CrashLogWriter.attachAppContext(applicationContext)
        val keyguard = applicationContext.getSystemService(Context.KEYGUARD_SERVICE) as? KeyguardManager
        val activityManager = applicationContext
            .getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        CrashLogWriter.install(
            CrashLogWriter.Environment(
                directory = java.io.File(applicationContext.filesDir, CrashLogStore.CRASHLOGS_SUBDIR),
                appVersion = BuildEnvironment.appVersion,
                osVersion = BuildEnvironment.osVersion,
                deviceModelClass = CrashLogWriter.currentDeviceModelClass(),
                isUnlocked = { keyguard?.isDeviceLocked == false },
                activityManager = activityManager,
            ),
        )
        // Materialize the lazy store so its 30-day sweep runs on every
        // launch (spec §16).
        crashLogStore

        val skipOnboarding = BuildConfig.DEBUG &&
            intent.getBooleanExtra("SKIP_ONBOARDING", false)

        uiScope.launch {
            authGate.bootstrap()
            onboardingState.rehydrate()
            settingsStore.hydrate()
            if (skipOnboarding) onboardingState.skipForTesting()
        }

        setContent {
            MalariaTheme {
                CompositionLocalProvider(
                    LocalClassifier provides classifier,
                    LocalAuthGate provides authGate,
                    LocalCameraService provides cameraService,
                    LocalModelRegistry provides modelRegistry,
                    LocalPredictionStore provides predictionStore,
                    LocalAuditLog provides auditLog,
                    LocalOnboardingState provides onboardingState,
                    LocalDatabase provides database,
                    LocalSettingsStore provides settingsStore,
                    LocalResetDeviceCoordinator provides resetDeviceCoordinator,
                    LocalExportService provides exportService,
                    LocalCrashLogStore provides crashLogStore,
                    LocalModelDownloadService provides modelDownloadService,
                ) {
                    val phase by onboardingState.phase.collectAsStateWithLifecycle()
                    if (phase == OnboardingState.Phase.Complete) {
                        RootScreen()
                    } else {
                        OnboardingFlow()
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        uiScope.coroutineContext[Job]?.cancel()
        computeScope.coroutineContext[Job]?.cancel()
    }
}
