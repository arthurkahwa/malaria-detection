package com.malaria.crashlogs

import kotlinx.serialization.Serializable

/**
 * Serializable shape of an on-device crash log (spec §16).
 *
 * Mirrors the spec's "What goes in a crash log" list exactly. Anything that
 * isn't on this struct is deliberately absent — most notably, no prediction
 * data, no override notes, no clinician initials or actor UUIDs, no image
 * hashes, no clinic config, and no consent records. The crash log is meant
 * to survive a database wipe (spec §16) and to be shareable as a self-
 * contained diagnostic blob; the privacy posture relies on the struct
 * literally not containing those fields.
 *
 * Authored cross-platform so the same JSON shape is produced by iOS
 * `CrashLogWriter.swift` and Android `CrashLogWriter.kt`. The struct lives
 * in the shared module rather than per-platform so a third-party reviewer
 * can verify the contract in one place.
 *
 * `timestampIso8601` is an ISO-8601 UTC string (e.g. `2026-05-21T12:34:56Z`)
 * — kept as a plain String here so the writer can format it with the
 * platform's own time API without dragging `kotlinx-datetime` into the
 * signal/exception path.
 */
@Serializable
data class CrashLogRecord(
    /** Generated UUID for this incident (spec §16: "Generated incident UUID"). */
    val incidentId: String,

    /** ISO-8601 UTC timestamp of when the crash handler fired. */
    val timestampIso8601: String,

    /** Spec §16: "App version". */
    val appVersion: String,

    /** Spec §16: "OS version". */
    val osVersion: String,

    /**
     * Spec §16: "device model class (e.g. iPhone15,2 / Pixel 9 Pro — model
     * identifiers, not personally identifying)". For iOS this is the `utsname`
     * machine string; for Android it's a compact `manufacturer / model`.
     */
    val deviceModelClass: String,

    /** Stack trace (multi-line string). */
    val stackTrace: String,

    /**
     * Spec §16: "Last 50 audit log action types (action strings only — no
     * resource IDs, no metadata, no actor IDs)". Pulled from
     * [RecentActionRing] at handler entry.
     */
    val recentActionTypes: List<String>,

    /**
     * Spec §16: "Memory pressure at time of crash". Captured as a rough
     * `kind=value` string (e.g. `resident_mb=148`) because the exact source
     * differs per platform. The string form keeps the contract
     * platform-flexible without changing the JSON schema later.
     */
    val memoryPressure: String,

    /** Spec §16: "Whether device was locked / unlocked at time of crash". */
    val deviceUnlocked: Boolean,
)
