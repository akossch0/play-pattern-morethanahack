"""Prepare the live-possession timeline before the fitted stoppage corrections."""

from .observed_exit import (
    add_persistent_exits,
    apply_early_exit,
    apply_persistent_exit,
    combine_receiver_exits,
    exit_events,
)
from .physical_timing import apply_first_contact, apply_restart_context
from .receiver_timing import predict_receiver_timing
from .stoppage_kernel import contact_evidence


def prepare_ball_out(
    tracking,
    ownership,
    forecast_exit_reference,
    typed_forecasts,
    tactics,
    observed_delay,
    crossing_delay,
):
    """Combine contact timing and rule-based ball exits on an ownership timeline."""
    period = tracking["period"]
    if ownership.shape != period.shape or forecast_exit_reference.shape != period.shape:
        raise ValueError("Both starting timelines must align with the tracking frames")

    evidence = contact_evidence(tracking)
    before_contacts = apply_restart_context(ownership, tracking, evidence=evidence)
    contacts = apply_first_contact(before_contacts, tracking, evidence=evidence)
    receiver = predict_receiver_timing(
        tracking, contacts, before_contacts, tactics, typed_forecasts
    )

    events, _ = exit_events(tracking)
    persistent = apply_persistent_exit(
        forecast_exit_reference, tracking, observed_delay, events=events
    )
    contact_with_exits = add_persistent_exits(
        contacts, forecast_exit_reference, persistent, period
    )
    receiver_with_exits = combine_receiver_exits(
        receiver, contacts, contact_with_exits, period
    )
    return apply_early_exit(
        receiver_with_exits, tracking, crossing_delay, events=events
    )
