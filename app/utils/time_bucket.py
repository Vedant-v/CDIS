from datetime import date, timedelta

def current_week_bucket():
    today = date.today()
    return today - timedelta(days=today.weekday())
