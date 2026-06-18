package com.malaria.crashlogs

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Spec §16: the ring must retain only the last 50 action strings, drop the
 * oldest on overflow, and report them in insertion order.
 */
class RecentActionRingTest {

    @Test
    fun empty_snapshotIsEmpty() {
        val ring = RecentActionRing()
        assertTrue(ring.snapshot().isEmpty())
        assertEquals(0, ring.size())
    }

    @Test
    fun retainsInsertionOrder_underCapacity() {
        val ring = RecentActionRing(capacity = 4)
        listOf("a", "b", "c").forEach { ring.push(it) }
        assertEquals(listOf("a", "b", "c"), ring.snapshot())
    }

    @Test
    fun atExactCapacity_keepsAll() {
        val ring = RecentActionRing(capacity = 3)
        listOf("a", "b", "c").forEach { ring.push(it) }
        assertEquals(listOf("a", "b", "c"), ring.snapshot())
        assertEquals(3, ring.size())
    }

    @Test
    fun overCapacity_dropsOldest_keepsNewest() {
        // Push 7 into a capacity-of-3 ring. Final state should be the last 3.
        val ring = RecentActionRing(capacity = 3)
        listOf("a", "b", "c", "d", "e", "f", "g").forEach { ring.push(it) }
        assertEquals(listOf("e", "f", "g"), ring.snapshot())
        assertEquals(3, ring.size())
    }

    @Test
    fun fiftyElementCapacity_retainsLastFifty() {
        // Exercise the spec's 50-element ceiling exactly.
        val ring = RecentActionRing(capacity = RecentActionRing.DEFAULT_CAPACITY)
        for (i in 0 until 100) ring.push("action_$i")
        val snap = ring.snapshot()
        assertEquals(50, snap.size)
        assertEquals("action_50", snap.first())
        assertEquals("action_99", snap.last())
    }

    @Test
    fun sharedSingleton_isUsable() {
        // Smoke test: the platform AuditLogs both push into this instance.
        // Clear first to keep the test isolated from other tests in the same
        // process.
        RecentActionRing.shared.clear()
        RecentActionRing.shared.push("smoke_test")
        assertEquals(listOf("smoke_test"), RecentActionRing.shared.snapshot())
        RecentActionRing.shared.clear()
    }
}
