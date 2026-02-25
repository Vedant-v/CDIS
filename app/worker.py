from celery import Celery

celery = Celery(
    "expertise_worker",
    broker="redis://redis:6379/0"
)

import app.services.expertise_worker
import app.services.entropy_worker
