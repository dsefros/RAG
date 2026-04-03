import uvicorn

from src.api.main import app
from src.config.settings import get_settings


if __name__ == "__main__":
    s = get_settings()
    uvicorn.run(app, host=s.api.host, port=s.api.port)
