# Project keep rules. R8/ProGuard runs only on release builds (build.gradle.kts).

# Keep all app code so reflective references from Room/CameraX/TFLite don't
# crash at runtime. Phase 15 will tighten this once the shrinker output has
# been audited.
-keep class com.malaria.** { *; }

# Kotlin metadata is required for reflection-driven libraries (Room).
-keepattributes RuntimeVisibleAnnotations,RuntimeInvisibleAnnotations
-keepattributes Signature,InnerClasses,EnclosingMethod
-keep class kotlin.Metadata { *; }

# Coroutines: keep service loader hooks.
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembers class kotlinx.coroutines.** { volatile <fields>; }

# Compose: the compiler plugin emits classes Compose tooling expects to keep.
-keep class androidx.compose.runtime.** { *; }

# TensorFlow Lite / LiteRT JNI bridges.
-keep class org.tensorflow.lite.** { *; }
-keep class com.google.ai.edge.litert.** { *; }

# SQLCipher native bindings.
-keep class net.sqlcipher.** { *; }
-keep class net.zetetic.** { *; }
