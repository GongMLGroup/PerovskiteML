import os
import neptune
from contextlib import contextmanager
from pydantic import BaseModel, ConfigDict
from typing import Optional


class NeptuneConfig(BaseModel):
    enabled: bool = False
    api_token: Optional[str] = None
    group_tags: list[str] = []
    model_config = ConfigDict(extra="allow")


@contextmanager
def neptune_context(config: NeptuneConfig | dict):
    """Neptune context manager that handles setup/teardown"""
    if isinstance(config, dict):
        config = NeptuneConfig(**config)

    if not config.enabled:
        yield None
        return

    # Expand environment variables in token
    api_token = os.path.expandvars(config.api_token) if config.api_token else None

    try:
        run = neptune.init_run(
            **config.model_dump(exclude={"enabled", "api_token", "group_tags"}),
            api_token=api_token,
        )
        run["sys/group_tags"].add(config.group_tags)
        yield run
    finally:
        if 'run' in locals():
            run.stop()
