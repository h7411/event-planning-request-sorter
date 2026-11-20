"""Event planning request categories and dataset for training and evaluation."""

from __future__ import annotations

from typing import Any

# Standard event planning request categories
CATEGORIES: list[str] = [
    "venue_inquiry",
    "catering_request",
    "budget_estimate",
    "schedule_coordination",
    "entertainment_request",
]

# Special tokens mapped to each category
CATEGORY_TOKENS: dict[str, str] = {
    cat: f"<|{cat}|>" for cat in CATEGORIES
}

# Balanced dataset of realistic client event requests
RAW_DATASET: list[dict[str, str]] = [
    # --- Venue Inquiry ---
    {
        "text": "Do you have an open ballroom available on October 14th for our annual gala?",
        "category": "venue_inquiry",
    },
    {
        "text": "What is the maximum standing guest capacity of the outdoor rooftop terrace?",
        "category": "venue_inquiry",
    },
    {
        "text": "We are looking for a rustic garden venue that can comfortably seat 150 wedding guests.",
        "category": "venue_inquiry",
    },
    {
        "text": "Is there accessible parking and step-free elevator access at the downtown event hall?",
        "category": "venue_inquiry",
    },
    {
        "text": "Can we schedule an on-site tour of the lakeside pavilion next Tuesday afternoon?",
        "category": "venue_inquiry",
    },
    {
        "text": "Does the event space have an indoor backup room in case of bad weather or rain?",
        "category": "venue_inquiry",
    },
    {
        "text": "We need a historic manor with overnight lodging accommodations nearby for our retreat.",
        "category": "venue_inquiry",
    },
    {
        "text": "Are bridal suites and private green rooms included with the main auditorium rental?",
        "category": "venue_inquiry",
    },

    # --- Catering Request ---
    {
        "text": "Can you provide a 3-course plated vegan dinner for 80 attendees?",
        "category": "catering_request",
    },
    {
        "text": "We have several corporate guests with severe peanut and gluten food allergies for lunch.",
        "category": "catering_request",
    },
    {
        "text": "Do you offer open bar beverage packages that include signature non-alcoholic mocktails?",
        "category": "catering_request",
    },
    {
        "text": "What are the hot and cold hors d'oeuvres options for a 90-minute cocktail reception?",
        "category": "catering_request",
    },
    {
        "text": "Could we arrange a hot breakfast buffet with fresh coffee and pastries for the workshop?",
        "category": "catering_request",
    },
    {
        "text": "Can you accommodate a certified kosher and halal banquet dinner for 200 guests?",
        "category": "catering_request",
    },
    {
        "text": "What is the per-person pricing for prime rib and seafood carving stations?",
        "category": "catering_request",
    },
    {
        "text": "Does the catering package include fine china, linen napkins, and full waitstaff service?",
        "category": "catering_request",
    },

    # --- Budget Estimate ---
    {
        "text": "Could you provide a detailed line-item cost estimate for a 100-person anniversary party?",
        "category": "budget_estimate",
    },
    {
        "text": "Our nonprofit committee has a strict budget ceiling of $8,000 for this charity auction.",
        "category": "budget_estimate",
    },
    {
        "text": "What is the required booking deposit percentage and what does the cancellation policy cost?",
        "category": "budget_estimate",
    },
    {
        "text": "Can you send an itemized pricing breakdown comparing standard versus premium event tiers?",
        "category": "budget_estimate",
    },
    {
        "text": "Are service gratuities, cleaning fees, and local sales taxes included in your price quote?",
        "category": "budget_estimate",
    },
    {
        "text": "We need a comprehensive budget proposal for a 3-day multi-track technology conference.",
        "category": "budget_estimate",
    },
    {
        "text": "Are there any hidden vendor coordination fees or overtime surcharges we should budget for?",
        "category": "budget_estimate",
    },
    {
        "text": "How do payment milestones and installment schedules work prior to the event date?",
        "category": "budget_estimate",
    },

    # --- Schedule Coordination ---
    {
        "text": "What time will the loading dock open for audio and floral vendors to begin morning setup?",
        "category": "schedule_coordination",
    },
    {
        "text": "Can we shift the executive keynote address from 10:00 AM to 10:30 AM on the run of show?",
        "category": "schedule_coordination",
    },
    {
        "text": "Please provide the master timeline and schedule sequence for Saturday's award ceremony.",
        "category": "schedule_coordination",
    },
    {
        "text": "When is the municipal amplified sound curfew for the outdoor courtyard reception?",
        "category": "schedule_coordination",
    },
    {
        "text": "How much transition buffer time is allotted between breakout sessions for attendee flow?",
        "category": "schedule_coordination",
    },
    {
        "text": "We would like to book a 45-minute rehearsal walkthrough the evening before the wedding.",
        "category": "schedule_coordination",
    },
    {
        "text": "Can the cleaning and teardown crew have access until 1:30 AM after the event concludes?",
        "category": "schedule_coordination",
    },
    {
        "text": "What is the expected arrival and staging schedule for the guest bus transport shuttles?",
        "category": "schedule_coordination",
    },

    # --- Entertainment Request ---
    {
        "text": "Do you provide wireless lapel microphones, dual projectors, and a concert sound system?",
        "category": "entertainment_request",
    },
    {
        "text": "We would like to hire an energetic 5-piece live jazz ensemble for the cocktail hour.",
        "category": "entertainment_request",
    },
    {
        "text": "Can you recommend an experienced bilingual DJ and MC who specializes in corporate galas?",
        "category": "entertainment_request",
    },
    {
        "text": "We want to rent a 360-degree slow-motion photo booth with custom branded digital backdrops.",
        "category": "entertainment_request",
    },
    {
        "text": "Do you have dedicated AV and lighting technicians to operate stage spotlights and mixers?",
        "category": "entertainment_request",
    },
    {
        "text": "Can you help us book an illusionist or roaming close-up magician for tableside entertainment?",
        "category": "entertainment_request",
    },
    {
        "text": "Is there an elevated stage and parquet dance floor available for cultural dance performances?",
        "category": "entertainment_request",
    },
    {
        "text": "What are the three-phase electrical power requirements for the headliner band's sound rig?",
        "category": "entertainment_request",
    },
]


def get_categories() -> list[str]:
    """Return the list of event request categories."""
    return list(CATEGORIES)


def get_dataset() -> list[dict[str, str]]:
    """Return a copy of the raw dataset examples."""
    return [dict(item) for item in RAW_DATASET]


def get_category_counts() -> dict[str, int]:
    """Return the count of examples per category."""
    counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    for item in RAW_DATASET:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    return counts
