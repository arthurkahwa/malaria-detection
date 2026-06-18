import org.jetbrains.kotlin.gradle.plugin.mpp.apple.XCFramework

plugins {
    kotlin("multiplatform")
    kotlin("plugin.serialization")
    id("com.android.library")
}

kotlin {
    jvmToolchain(17)

    androidTarget()

    val xcframeworkName = "Shared"
    val xcf = XCFramework(xcframeworkName)

    listOf(
        iosX64(),
        iosArm64(),
        iosSimulatorArm64()
    ).forEach {
        it.binaries.framework {
            baseName = xcframeworkName
            isStatic = true
            xcf.add(this)
        }
    }

    // Required so `iosMain` (parent of iosX64Main / iosArm64Main / iosSimulatorArm64Main)
    // exists as a source set. Without this the `val iosMain by getting` below fails.
    applyDefaultHierarchyTemplate()

    sourceSets {
        val commonMain by getting {
            dependencies {
                // `api` for types that appear in the shared public surface
                // (suspend in Classifier; Instant on Prediction) so consumers
                // see them transitively.
                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.1")
                api("org.jetbrains.kotlinx:kotlinx-datetime:0.6.1")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
                implementation("com.benasher44:uuid:0.8.4")
                implementation("org.kotlincrypto.hash:sha2:0.5.6")
                // HMAC-SHA256 for Phase 13 export bundle signing (spec §14).
                // Same kotlincrypto family as the sha2 dep above; identical
                // version pin means a single artifact-set across the family.
                implementation("org.kotlincrypto.macs:hmac-sha2:0.5.6")
                // Note: no Ktor, no SQLDelight, no cloud-tier libraries
                // Persistence is platform-native; cloud tier is deferred
            }
        }
        val iosMain by getting {
            dependencies {
                // Core ML, Vision, AVFoundation, LocalAuthentication, SwiftData via cinterop
            }
        }
        val androidMain by getting {
            dependencies {
                // Phase 3 deviation from spec §5 line 519-521: spec said TFLite
                // lives in androidApp, with shared:androidMain only "interfacing"
                // with it. That phrasing is impractical — the actual class itself
                // needs to import org.tensorflow.lite.Interpreter to function, so
                // TFLite has to be visible to androidMain's compile classpath.
                // We declare TFLite here (not in androidApp); androidApp consumes
                // the symbols transitively via :shared. The deps are kept in
                // androidApp/build.gradle.kts too because some androidApp code
                // (NDK signal handling in Phase 15, eventually) may reference
                // TFLite native libs directly. `api` so the symbols flow through.
                api("org.tensorflow:tensorflow-lite:2.16.1")
                api("org.tensorflow:tensorflow-lite-gpu:2.16.1")
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.1")
            }
        }
    }
}

android {
    namespace = "com.malaria.shared"
    compileSdk = 36
    defaultConfig {
        minSdk = 36
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

// Note: `assembleSharedXCFramework` is auto-registered by `XCFramework("Shared")`
// above and produces both debug and release artifacts. Use
// `assembleSharedReleaseXCFramework` for the SPM-consumed release build.
