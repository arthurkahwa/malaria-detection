package com.malaria.domain

/**
 * Canonical English `lowercase_snake_case` form for an enum value.
 *
 * The `canonical` string is the persisted form on both platforms. UI translates
 * to localized display strings at render time; the stored data is always the
 * canonical English form.
 */
private fun toCanonical(name: String): String = name.lowercase()

enum class ClinicianRole {
    ADMIN,
    MICROSCOPIST,
    OBSERVER;

    val canonical: String get() = toCanonical(name)
}

enum class OverrideContext {
    LIVE,
    REVIEW;

    val canonical: String get() = toCanonical(name)
}

enum class OverrideReason {
    IMAGE_QUALITY,
    ATYPICAL_MORPHOLOGY,
    MODEL_FALSE_POSITIVE,
    MODEL_FALSE_NEGATIVE,
    OTHER;

    val canonical: String get() = toCanonical(name)
}

enum class Jurisdiction {
    US_HIPAA,
    EU_GDPR_DE,
    EU_GDPR_FR,
    EU_GDPR_GENERIC,
    KE_DPA,
    OTHER;

    val canonical: String get() = toCanonical(name)
}

enum class LawfulBasis {
    EXPLICIT_CONSENT,
    VITAL_INTERESTS,
    HEALTH_PROVISION;

    val canonical: String get() = toCanonical(name)
}
