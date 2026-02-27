from datetime import date, timedelta

def current_week_bucket() -> date:
    """Return the Monday of the current ISO week."""
    today = date.today()
    return today - timedelta(days=today.weekday())