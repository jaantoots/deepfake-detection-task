"""Implementational utility functions"""
from functools import wraps

import click


def progress(**style_kwargs):
    """Show progressbar on loader"""

    def decorator(func):
        @wraps(func)
        def wrapper(model, loader, *args, **kwargs):
            with click.progressbar(
                loader,
                label=click.style(
                    f"{func.__name__.replace('_', ' ').title():8s}", **style_kwargs,
                ),
            ) as loader_:
                return func(model, loader_, *args, **kwargs)

        return wrapper

    return decorator
