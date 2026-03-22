from datetime import date

from flowbase.storage.manifest import (
    ManifestEntry,
    ResolutionPolicy,
    Tier,
    infer_entry_from_key,
    resolve_for_date,
)


def test_infer_entry_from_key_daily_monthly_yearly() -> None:
    daily = infer_entry_from_key("bsp", "data_2026_03_14.parquet")
    assert daily is not None
    assert daily.tier == Tier.DAILY
    assert daily.start_date == date(2026, 3, 14)
    assert daily.end_date == date(2026, 3, 14)

    monthly = infer_entry_from_key("bsp", "data_2026_03.parquet")
    assert monthly is not None
    assert monthly.tier == Tier.MONTHLY
    assert monthly.start_date == date(2026, 3, 1)
    assert monthly.end_date == date(2026, 3, 31)

    yearly = infer_entry_from_key("bsp", "data_2026.parquet")
    assert yearly is not None
    assert yearly.tier == Tier.YEARLY
    assert yearly.start_date == date(2026, 1, 1)
    assert yearly.end_date == date(2026, 12, 31)


def test_resolve_for_date_obeys_tier_policy() -> None:
    entries = [
        ManifestEntry(
            table="bsp",
            key="data_2026.parquet",
            tier=Tier.YEARLY,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
        ),
        ManifestEntry(
            table="bsp",
            key="data_2026_03.parquet",
            tier=Tier.MONTHLY,
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 31),
        ),
        ManifestEntry(
            table="bsp",
            key="data_2026_03_14.parquet",
            tier=Tier.DAILY,
            start_date=date(2026, 3, 14),
            end_date=date(2026, 3, 14),
        ),
    ]

    target = date(2026, 3, 14)

    resolved_daily = resolve_for_date(entries, target, ResolutionPolicy.DAILY_PREFER)
    assert resolved_daily is not None
    assert resolved_daily.key == "data_2026_03_14.parquet"

    resolved_monthly = resolve_for_date(entries, target, ResolutionPolicy.MONTHLY_PREFER)
    assert resolved_monthly is not None
    assert resolved_monthly.key == "data_2026_03.parquet"

    resolved_yearly = resolve_for_date(entries, target, ResolutionPolicy.YEARLY_PREFER)
    assert resolved_yearly is not None
    assert resolved_yearly.key == "data_2026.parquet"
