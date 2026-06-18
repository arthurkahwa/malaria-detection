package com.malaria.export

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Spec §14 bundle-build invariants:
 *
 *  - schemaVersion is always "1.0";
 *  - same input → byte-identical output (cross-platform invariant);
 *  - the signature on the produced bundle equals the HMAC of the
 *    re-serialized unsigned form (i.e. a verifier following the same
 *    algorithm can validate the bundle without a back-channel);
 *  - field ordering is the spec's declared order.
 */
class ExportBundleBuilderTest {

    private fun sampleFixture(): Fixture {
        val clinician = ExportedClinicianProfile(
            actorId = "00000000-0000-0000-0000-000000000001",
            role = "microscopist",
            initials = "AB",
            enrolledAt = "2026-01-01T08:00:00Z",
            biometricEnrolled = true,
        )

        val p1 = ExportedPrediction(
            id = "11111111-1111-1111-1111-111111111111",
            sessionId = "session-1",
            timestamp = "2026-01-01T08:01:00Z",
            modelId = "BNLeaky_Keras",
            modelVersion = "BNLeaky_Keras",
            parasitizedProb = 0.91,
            uninfectedProb = 0.09,
            label = "Parasitized",
            threshold = 0.30,
            flaggedForReview = false,
            inferenceMs = 25,
            imageHash = "a".repeat(64),
            clinicianOverride = null,
            overrideContext = null,
            duplicateOfId = null,
            sessionLabel = null,
            appVersion = "0.1.0",
            osVersion = "iOS 26.0",
        )
        val p2 = p1.copy(
            id = "22222222-2222-2222-2222-222222222222",
            timestamp = "2026-01-01T08:02:00Z",
            parasitizedProb = 0.15,
            uninfectedProb = 0.85,
            label = "Uninfected",
        )

        val a1 = ExportedAuditEntry(
            id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            seq = 1L,
            timestamp = "2026-01-01T08:00:00Z",
            actorId = clinician.actorId,
            actorRoleAtTime = clinician.role,
            action = "session_unlocked",
            resourceType = null,
            resourceId = null,
            metadataJson = "{}",
            overrideContext = null,
            overrideReason = null,
            overrideNotes = null,
            contextReviewed = null,
            overrideActorInitials = null,
            appVersion = "0.1.0",
            osVersion = "iOS 26.0",
        )
        val a2 = a1.copy(
            id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            seq = 2L,
            timestamp = "2026-01-01T08:01:00Z",
            action = "prediction_created",
        )
        val a3 = a1.copy(
            id = "cccccccc-cccc-cccc-cccc-cccccccccccc",
            seq = 3L,
            timestamp = "2026-01-01T08:02:00Z",
            action = "prediction_created",
        )

        val summary = ExportSummary(
            predictionCount = 2,
            sessionCount = 1,
            auditEntryCount = 3,
            consentRecordCount = 0,
            firstPredictionAt = p1.timestamp,
            lastPredictionAt = p2.timestamp,
        )

        return Fixture(
            exportTimestamp = "2026-01-01T09:00:00Z",
            exportedByActorId = clinician.actorId,
            deviceUuid = "DEVICE-UUID-FIXTURE",
            platform = ExportPlatform.IOS,
            clinicName = "Kisumu District Health Centre",
            jurisdiction = "ke_dpa",
            lawfulBasis = "vital_interests",
            appVersion = "0.1.0",
            osVersion = "iOS 26.0",
            summary = summary,
            clinicianProfiles = listOf(clinician),
            predictions = listOf(p1, p2),
            auditLog = listOf(a1, a2, a3),
        )
    }

    @Test
    fun bundleSchemaVersionIsOneDotZero() {
        val fx = sampleFixture()
        val json = ExportBundleBuilder().build(
            exportTimestamp = fx.exportTimestamp,
            exportedByActorId = fx.exportedByActorId,
            deviceUuid = fx.deviceUuid,
            platform = fx.platform,
            clinicName = fx.clinicName,
            jurisdiction = fx.jurisdiction,
            lawfulBasis = fx.lawfulBasis,
            appVersion = fx.appVersion,
            osVersion = fx.osVersion,
            summary = fx.summary,
            clinicianProfiles = fx.clinicianProfiles,
            predictions = fx.predictions,
            auditLog = fx.auditLog,
        )
        // Spec §14 requires `"schemaVersion": "1.0"` — the encoder emits
        // fields in declaration order so this is the first key.
        assertTrue(
            json.startsWith("{\"schemaVersion\":\"1.0\""),
            "expected leading schemaVersion=1.0, got: ${json.take(60)}",
        )
    }

    @Test
    fun buildTwiceWithSameInputIsByteIdentical() {
        val fx = sampleFixture()
        val builder = ExportBundleBuilder()
        val a = builder.build(
            exportTimestamp = fx.exportTimestamp,
            exportedByActorId = fx.exportedByActorId,
            deviceUuid = fx.deviceUuid,
            platform = fx.platform,
            clinicName = fx.clinicName,
            jurisdiction = fx.jurisdiction,
            lawfulBasis = fx.lawfulBasis,
            appVersion = fx.appVersion,
            osVersion = fx.osVersion,
            summary = fx.summary,
            clinicianProfiles = fx.clinicianProfiles,
            predictions = fx.predictions,
            auditLog = fx.auditLog,
        )
        val b = builder.build(
            exportTimestamp = fx.exportTimestamp,
            exportedByActorId = fx.exportedByActorId,
            deviceUuid = fx.deviceUuid,
            platform = fx.platform,
            clinicName = fx.clinicName,
            jurisdiction = fx.jurisdiction,
            lawfulBasis = fx.lawfulBasis,
            appVersion = fx.appVersion,
            osVersion = fx.osVersion,
            summary = fx.summary,
            clinicianProfiles = fx.clinicianProfiles,
            predictions = fx.predictions,
            auditLog = fx.auditLog,
        )
        assertEquals(a, b)
    }

    @Test
    fun bundleByteIdenticalAcrossPlatformsForIdenticalInput() {
        // Spec §14 "byte-identical between platforms" invariant: if the
        // same content (including the same platform tag) is built on iOS
        // and Android, the JSON bytes are identical. The shared module
        // performs all serialization, so this property comes for free —
        // but pinning it here guards against any platform-divergent
        // float / number formatting that might creep into kotlinx-
        // serialization in a future upgrade.
        //
        // We can't actually run iOS and Android side-by-side from a JVM
        // unit test, so the next best thing is to assert that the
        // *shared* path produces identical bytes for identical input on
        // every JVM invocation (which is what runs on Android's JVM
        // unit-test target *and* what gets compiled into the iOS Kotlin/
        // Native target's `commonMain` source set).
        val fx = sampleFixture()
        val builder = ExportBundleBuilder()
        val a = builder.build(
            exportTimestamp = fx.exportTimestamp,
            exportedByActorId = fx.exportedByActorId,
            deviceUuid = fx.deviceUuid,
            platform = ExportPlatform.IOS,
            clinicName = fx.clinicName,
            jurisdiction = fx.jurisdiction,
            lawfulBasis = fx.lawfulBasis,
            appVersion = fx.appVersion,
            osVersion = fx.osVersion,
            summary = fx.summary,
            clinicianProfiles = fx.clinicianProfiles,
            predictions = fx.predictions,
            auditLog = fx.auditLog,
        )
        val b = builder.build(
            exportTimestamp = fx.exportTimestamp,
            exportedByActorId = fx.exportedByActorId,
            deviceUuid = fx.deviceUuid,
            platform = ExportPlatform.IOS,
            clinicName = fx.clinicName,
            jurisdiction = fx.jurisdiction,
            lawfulBasis = fx.lawfulBasis,
            appVersion = fx.appVersion,
            osVersion = fx.osVersion,
            summary = fx.summary,
            clinicianProfiles = fx.clinicianProfiles,
            predictions = fx.predictions,
            auditLog = fx.auditLog,
        )
        assertEquals(a, b)
        // And the signature is the same too (transitively, since it's
        // computed over the same bytes).
        val sigA = a.substringAfterLast("\"signature\":\"").substringBefore("\"")
        val sigB = b.substringAfterLast("\"signature\":\"").substringBefore("\"")
        assertEquals(sigA, sigB)
    }

    @Test
    fun bundleSignatureVerifiesOverReSerializedUnsignedForm() {
        val fx = sampleFixture()
        val builder = ExportBundleBuilder()
        val signed = builder.build(
            exportTimestamp = fx.exportTimestamp,
            exportedByActorId = fx.exportedByActorId,
            deviceUuid = fx.deviceUuid,
            platform = fx.platform,
            clinicName = fx.clinicName,
            jurisdiction = fx.jurisdiction,
            lawfulBasis = fx.lawfulBasis,
            appVersion = fx.appVersion,
            osVersion = fx.osVersion,
            summary = fx.summary,
            clinicianProfiles = fx.clinicianProfiles,
            predictions = fx.predictions,
            auditLog = fx.auditLog,
        )

        // A verifier re-serializes the unsigned form (same field order),
        // re-runs HMAC, and compares to the bundle's `signature` field.
        val unsigned = UnsignedExportBundle(
            schemaVersion = EXPORT_SCHEMA_VERSION,
            exportTimestamp = fx.exportTimestamp,
            exportedByActorId = fx.exportedByActorId,
            deviceUuid = fx.deviceUuid,
            platform = fx.platform,
            clinicName = fx.clinicName,
            jurisdiction = fx.jurisdiction,
            lawfulBasis = fx.lawfulBasis,
            appVersion = fx.appVersion,
            osVersion = fx.osVersion,
            summary = fx.summary,
            clinicianProfiles = fx.clinicianProfiles,
            predictions = fx.predictions,
            auditLog = fx.auditLog,
        )
        val unsignedJson = ExportBundleBuilder.JSON.encodeToString(
            UnsignedExportBundle.serializer(),
            unsigned,
        )
        val expectedSig = ExportSigner.sign(
            unsignedJson = unsignedJson,
            deviceUuid = fx.deviceUuid,
            timestampSalt = fx.exportTimestamp,
        )
        // Extract the bundle's `signature` field. It's the trailing
        // `"signature":"<hex>"}` pattern; the encoder doesn't emit
        // whitespace so a simple substring extract suffices.
        val marker = "\"signature\":\""
        val start = signed.lastIndexOf(marker) + marker.length
        val end = signed.indexOf('"', start)
        val embeddedSig = signed.substring(start, end)
        assertEquals(expectedSig, embeddedSig)
    }

    @Test
    fun bundleContainsTimestampsAsIso8601Strings() {
        val fx = sampleFixture()
        val json = ExportBundleBuilder().build(
            exportTimestamp = fx.exportTimestamp,
            exportedByActorId = fx.exportedByActorId,
            deviceUuid = fx.deviceUuid,
            platform = fx.platform,
            clinicName = fx.clinicName,
            jurisdiction = fx.jurisdiction,
            lawfulBasis = fx.lawfulBasis,
            appVersion = fx.appVersion,
            osVersion = fx.osVersion,
            summary = fx.summary,
            clinicianProfiles = fx.clinicianProfiles,
            predictions = fx.predictions,
            auditLog = fx.auditLog,
        )
        // Spot-check a few — every timestamp is a quoted ISO-8601 string
        // ending in "Z" (UTC). The fixture above uses that form everywhere
        // so any boundary serialisation that swapped to numbers would
        // break this assertion.
        assertTrue(json.contains("\"exportTimestamp\":\"2026-01-01T09:00:00Z\""))
        assertTrue(json.contains("\"firstPredictionAt\":\"2026-01-01T08:01:00Z\""))
        assertTrue(json.contains("\"enrolledAt\":\"2026-01-01T08:00:00Z\""))
    }

    private data class Fixture(
        val exportTimestamp: String,
        val exportedByActorId: String,
        val deviceUuid: String,
        val platform: String,
        val clinicName: String,
        val jurisdiction: String,
        val lawfulBasis: String,
        val appVersion: String,
        val osVersion: String,
        val summary: ExportSummary,
        val clinicianProfiles: List<ExportedClinicianProfile>,
        val predictions: List<ExportedPrediction>,
        val auditLog: List<ExportedAuditEntry>,
    )
}
