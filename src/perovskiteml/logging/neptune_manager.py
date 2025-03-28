import os
import neptune
from contextlib import contextmanager
from pydantic import BaseModel


class NeptuneConfig(BaseModel):
    enabled: bool = True
    project: str
    api_token: str
    group_tags: list[str] = []
    tags: list[str] = []
    description: str = ""


@contextmanager
def neptune_context(config: NeptuneConfig | dict):
    """Neptune context manager that handles setup/teardown"""
    if isinstance(config, dict):
        config = NeptuneConfig(**config)

    if not config.enabled:
        yield None
        return

    # Expand environment variables in token
    api_token = os.path.expandvars(config.api_token)

    try:
        run = neptune.init_run(
            project=config.project,
            api_token=api_token,
            tags=config.tags,
            description=config.description
        )
        run["sys/group_tags"].add(config.group_tags)
        yield run
    finally:
        if 'run' in locals():
            run.stop()
