// androidApp/build.gradle.kts
//
// Per spec §19 "Android app — Gradle module".

plugins {
    id("com.android.application")
    kotlin("android")
    kotlin("plugin.compose")
    id("androidx.room")
    id("kotlin-kapt")
}

android {
    namespace = "com.malaria.android"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.malaria.detector"
        minSdk = 36
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        // Required so BuildEnvironment can read VERSION_NAME from BuildConfig.
        buildConfig = true
    }

    testOptions {
        // JVM unit tests reference `BuildEnvironment.osVersion` indirectly via
        // `AuditLog`; that calls `android.os.Build.VERSION.RELEASE`. Without
        // this flag the stub `android.jar` throws "not mocked" instead of
        // returning a default. Mock-Build is sufficient for the
        // OnboardingState state-machine tests; Phase 5 already pins the JSON
        // stub the same way (see `org.json:json` testImplementation).
        unitTests.isReturnDefaultValues = true
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    bundle {
        // Single APK preferred for v1 — disable per-density and per-ABI splits.
        density { enableSplit = false }
        abi { enableSplit = false }
    }

    packaging {
        resources.excludes.add("META-INF/*")
    }

    room {
        // Schema export for Phase 5 migration testing.
        schemaDirectory("$projectDir/schemas")
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(project(":shared"))

    // Phase 14: the crash log writer + store serialize `CrashLogRecord`
    // (defined in :shared) via kotlinx-serialization-json. `:shared`
    // declares the dep as `implementation`, so the androidApp module
    // needs its own explicit reference for the symbols to be visible
    // here. Same pinned version as :shared/build.gradle.kts.
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")

    // Compose BOM (April 2026 release)
    implementation(platform("androidx.compose:compose-bom:2026.04.01"))
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.activity:activity-compose:1.10.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.0") // collectAsStateWithLifecycle
    implementation("androidx.navigation:navigation-compose:2.9.0")
    // Note: no lifecycle-viewmodel-compose; the app does not use ViewModels (spec §4).

    // Room with SQLCipher
    implementation("androidx.room:room-runtime:2.7.0")
    implementation("androidx.room:room-ktx:2.7.0")
    kapt("androidx.room:room-compiler:2.7.0")
    implementation("net.zetetic:sqlcipher-android:4.6.1@aar")
    implementation("androidx.sqlite:sqlite-ktx:2.4.0")

    // ML inference (TensorFlow Lite). Note: the spec §19 listed both
    // `tensorflow-lite` and `com.google.ai.edge.litert:litert` — they're
    // duplicates (LiteRT is the rebrand of TFLite) and shipping both causes
    // a duplicate `libtensorflowlite_jni.so` merge failure. Keeping the
    // canonical `org.tensorflow.lite.Interpreter` API package; switch to
    // `litert` later if the Google AI Edge migration completes.
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")

    // Camera (CameraX)
    implementation("androidx.camera:camera-core:1.4.1")
    implementation("androidx.camera:camera-camera2:1.4.1")
    implementation("androidx.camera:camera-lifecycle:1.4.1")
    implementation("androidx.camera:camera-view:1.4.1")

    // Security and biometrics
    implementation("androidx.biometric:biometric:1.2.0-alpha05")
    implementation("androidx.security:security-crypto-ktx:1.1.0-alpha06")

    // Preferences (unencrypted; for language only)
    implementation("androidx.datastore:datastore-preferences:1.1.1")

    // Networking (for Hugging Face model downloads)
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("androidx.work:work-runtime-ktx:2.10.0") // resumable downloads

    // Test — JVM
    testImplementation("junit:junit:4.13.2")
    testImplementation("androidx.room:room-testing:2.7.0")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.1")
    // Real `org.json` for the JVM unit tests (the Android stub throws
    // "not mocked"). Used only by `SchemaDriftTest`.
    testImplementation("org.json:json:20240303")
    // Test — Android instrumented (Phase 5 Room + SQLCipher verification)
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
    androidTestImplementation("androidx.room:room-testing:2.7.0")
    androidTestImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.1")
}
