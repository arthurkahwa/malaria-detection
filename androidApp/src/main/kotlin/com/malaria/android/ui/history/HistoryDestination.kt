package com.malaria.android.ui.history

/**
 * In-memory navigation state for the History tab.
 *
 * Phase 10 deliberately avoids pulling in Compose Navigation as a new
 * dependency: there's only one navigation tree under the History tab, and
 * the destinations are well-typed. Stack is managed by [HistoryNavigator]
 * in a `remember { }` slot — configuration changes reset back to
 * [Subsections]; persistence across rotation lands when Compose
 * Navigation is wired (Phase 11 / 13 task).
 *
 * Cross-tab navigation stays in `RootScreen` (Phase 7).
 *
 * Mirrors the iOS `NavigationStack` surface in `HistoryTab.swift` — every
 * `NavigationLink` becomes a `Destination` here, and the destination
 * arguments are passed by id (Prediction/Session) rather than by-value
 * because the detail view re-fetches via the DAO.
 */
sealed interface HistoryDestination {

    /** Root subsections list. */
    data object Subsections : HistoryDestination

    data object RecentPredictions : HistoryDestination

    data object FlaggedForReview : HistoryDestination

    data object Sessions : HistoryDestination

    data object AuditLog : HistoryDestination

    data object DataManagement : HistoryDestination

    data class PredictionDetail(val predictionId: String) : HistoryDestination

    data class SessionDetail(val sessionId: String) : HistoryDestination

    data class AuditEntryDetail(val auditEntryId: String) : HistoryDestination

    data class MarkAsDuplicate(val duplicatePredictionId: String) : HistoryDestination

    data class SessionRelabel(val sessionId: String) : HistoryDestination

    /** Phase 9 review-override screen (spec §12 review override). */
    data class ReviewOverride(val predictionId: String) : HistoryDestination

    /** Phase 11 reset-device flow (spec §10 re-onboarding). */
    data object ResetDevice : HistoryDestination
}
