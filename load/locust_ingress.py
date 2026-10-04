"""locustfile.py через Ingress: Python на macOS не резолвит *.localhost,
поэтому стучимся в 127.0.0.1, а Traefik выбирает маршрут по заголовку Host.

    uv run --with locust locust -f load/locust_ingress.py --headless -u 60 -r 20 -t 3m --host http://127.0.0.1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from locustfile import EstateUser  # noqa: E402


class EstateIngressUser(EstateUser):
    def on_start(self):
        self.client.headers["Host"] = "estate.localhost"
        super_start = getattr(super(), "on_start", None)
        if super_start:
            super_start()


EstateUser.abstract = True
