package com.malaria.crashlogs

/**
 * Ring buffer of the last 50 audit action canonical strings (spec §16).
 *
 * The crash log writer must capture "Last 50 audit log action types (action
 * strings only — no resource IDs, no metadata, no actor IDs)" at handler
 * entry. Reading the audit DB from a signal handler is unsafe; instead each
 * platform's `AuditLog.write()` pushes the canonical action string into this
 * ring as a side-effect of every entry, and the crash handler reads
 * [snapshot] without touching disk.
 *
 * The ring deliberately holds only the canonical action strings — not the
 * actor id, resource id, metadata, or timestamp — so a snapshot satisfies
 * the spec's "action strings only" constraint by construction.
 *
 * Concurrency: implementation holds an immutable [List] in a backing field
 * and replaces it on every push. Reads return the current immutable
 * reference, so a snapshot is always self-consistent. A racing `push` can
 * lose an entry under heavy concurrent writes — that's an accepted tradeoff
 * because spec §16 only requires "the last 50" approximately, and audit
 * writes happen on the UI main thread on both platforms anyway. The
 * `commonMain` source set has no `synchronized` / `AtomicReference`
 * available on Kotlin 2.1, so doing this "well enough" without platform
 * indirection is the pragmatic choice for v0.1.
 *
 * The Phase 14 v0.1 path uses `Thread.setDefaultUncaughtExceptionHandler` /
 * `NSSetUncaughtExceptionHandler` on the JVM/Swift exception paths only —
 * not raw POSIX signals — so the lack of true atomicity is sufficient. See
 * the README "Known limitations" for the Phase 15 polish note.
 */
class RecentActionRing(private val capacity: Int = DEFAULT_CAPACITY) {

    private var entries: List<String> = emptyList()

    /** Append [action] (the canonical lowercase-snake string). */
    fun push(action: String) {
        val current = entries
        entries = if (current.size < capacity) {
            current + action
        } else {
            // Drop oldest, append newest. The list is immutable so we build
            // a fresh one each time.
            current.subList(current.size - capacity + 1, current.size) + action
        }
    }

    /** Immutable snapshot in insertion order (oldest first, newest last). */
    fun snapshot(): List<String> = entries

    /** Current count, exposed for the count-readout in Settings. */
    fun size(): Int = entries.size

    /** Empties the ring. Used by tests. */
    fun clear() {
        entries = emptyList()
    }

    companion object {
        /**
         * Spec §16 fixes the size at 50. Centralized so platform writers can
         * read the same constant and tests can verify wrap-around.
         */
        const val DEFAULT_CAPACITY: Int = 50

        /**
         * Process-wide singleton fed by both platforms' `AuditLog.write()`.
         * The same instance is read by the crash handler at incident time.
         * A singleton (rather than a per-`AuditLog` field) is intentional:
         * even if a future code path creates multiple `AuditLog` instances
         * — e.g. tests, an alternate composition root — the crash log
         * always sees the most-recent activity.
         */
        val shared: RecentActionRing = RecentActionRing()
    }
}
