import SwiftData
import SwiftUI
import UIKit
@preconcurrency import Shared

@main
struct MalariaDetectorApp: App {

    // Services constructed once at the composition root and injected via
    // SwiftUI's type-based environment (iOS 17+). The keypath-based form
    // doesn't work because SwiftUI evaluates the `defaultValue` during
    // `WritableKeyPath` SET; see `Environment/EnvironmentKeys.swift`.

    let modelContainer: ModelContainer

    @State private var classifierBridge: ClassifierBridge
    @State private var authGate: AuthGate
    @State private var cameraService: CameraService
    @State private var modelRegistry: ModelRegistryService
    @State private var predictionStore: PredictionStore
    @State private var auditLog: AuditLog
    @State private var onboardingState: OnboardingState
    @State private var settingsStore: SettingsStore
    @State private var resetCoordinator: ResetDeviceCoordinator
    @State private var exportService: ExportService
    @State private var crashLogStore: CrashLogStore
    @State private var modelDownloadService: ModelDownloadService

    @MainActor
    init() {
        let container: ModelContainer
        do {
            container = try ModelContainerFactory.make()
        } catch {
            fatalError("Failed to create ModelContainer: \(error)")
        }
        self.modelContainer = container

        let context = container.mainContext
        let predictionRepo = PredictionRepository(context: context)
        let auditRepo = AuditRepository(context: context)
        let clinicianRepo = ClinicianRepository(context: context)
        let consentRepo = ConsentRepository(context: context)

        let auditService = AuditLog(repository: auditRepo)
        let authGateService = AuthGate(audit: auditService, clinicians: clinicianRepo)
        authGateService.bootstrap()

        _classifierBridge = State(initialValue: ClassifierBridge(classifier: Classifier(modelId: "BNLeaky_Keras")))
        _authGate = State(initialValue: authGateService)
        _cameraService = State(initialValue: CameraService())
        _modelRegistry = State(initialValue: try! ModelRegistryService())
        _auditLog = State(initialValue: auditService)
        _predictionStore = State(initialValue: PredictionStore(
            predictions: predictionRepo,
            audit: auditService,
            clinician: clinicianRepo
        ))
        let onboarding = OnboardingState(
            clinicians: clinicianRepo,
            consents: consentRepo,
            audit: auditService
        )
#if DEBUG
        if ProcessInfo.processInfo.arguments.contains("--skip-onboarding") {
            onboarding.skipForTesting()
        }
#endif
        _onboardingState = State(initialValue: onboarding)
        let settings = SettingsStore(
            auditRepo: auditRepo,
            audit: auditService,
            clinicians: clinicianRepo
        )
        settings.hydrate()
        _settingsStore = State(initialValue: settings)
        _resetCoordinator = State(initialValue: ResetDeviceCoordinator(
            clinicians: clinicianRepo,
            audit: auditService,
            onboarding: onboarding,
            settings: settings
        ))
        _exportService = State(initialValue: ExportService(
            predictions: predictionRepo,
            audits: auditRepo,
            clinicians: clinicianRepo,
            consents: consentRepo,
            settings: settings,
            auditLog: auditService
        ))
        let crashStore = CrashLogStore(
            auditLog: auditService,
            authGate: authGateService,
            clinicians: clinicianRepo
        )
        _crashLogStore = State(initialValue: crashStore)
        _modelDownloadService = State(initialValue: ModelDownloadService())

        // Spec §16: install the global uncaught-exception handler. The
        // environment is captured once here so the C handler trampoline
        // doesn't have to query Bundle / utsname / UIDevice during the
        // crash path. The locked/unlocked readout uses
        // `UIApplication.isProtectedDataAvailable` as a best-effort
        // proxy on iOS (no first-party "locked" API exists).
        let fm = FileManager.default
        let documents = (try? fm.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let crashlogsDir = documents.appendingPathComponent("crashlogs", isDirectory: true)
        CrashLogWriter.install(
            environment: CrashLogWriter.Environment(
                directoryURL: crashlogsDir,
                appVersion: BuildEnvironment.appVersion,
                osVersion: BuildEnvironment.osVersion,
                deviceModelClass: CrashLogWriter.currentDeviceModelClass(),
                isUnlocked: { @Sendable in
                    // Reading `isProtectedDataAvailable` outside the main
                    // actor is safe — UIKit documents this as thread-safe
                    // for read access.
                    MainActor.assumeIsolated {
                        UIApplication.shared.isProtectedDataAvailable
                    }
                }
            )
        )
    }

    var body: some Scene {
        WindowGroup {
            // Phase 6: the onboarding wizard owns the screen until
            // `OnboardingState.phase == .complete`. The environment
            // chain stays above this branch so both `OnboardingFlow`
            // and `RootView` see the same service instances.
            Group {
                if onboardingState.phase == .complete {
                    RootView()
                } else {
                    OnboardingFlow()
                }
            }
            .modelContainer(modelContainer)
            .environment(classifierBridge)
            .environment(authGate)
            .environment(cameraService)
            .environment(modelRegistry)
            .environment(predictionStore)
            .environment(auditLog)
            .environment(onboardingState)
            .environment(settingsStore)
            .environment(resetCoordinator)
            .environment(exportService)
            .environment(crashLogStore)
            .environment(modelDownloadService)
        }
    }
}
