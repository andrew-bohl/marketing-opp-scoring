import logging
import os
import sys

from flask import Flask, jsonify
from raven.contrib.flask import Sentry

from src.api import api
from src.config import BaseConfig

__all__ = ['create_app']


def create_app(config=None, app_name=None):
    """Creates the Flask app.

    Creates a instance of the flask application as it sets up
    the configuration, api blueprints, and logging.
    Args:
        config: optional configuration input
        app_name: optional name for application. Will default to app
    Returns:
        flask application instance
    """
    def dump_config():
        print(f"Starting {config.APP_NAME} with config:")
        for k, v in vars(config).items():
            if not k.startswith("__"):
                print(f"{k}: {v}")
        print("\n\n")

    dump_config()
    app_name = app_name or config.APP_NAME
    app = Flask(app_name)

    _configure_app(app, config)
    _configure_sentry(app)  # for catching silent exception logging
    _configure_blueprints(app)
    _configure_logging(app)
    return app


def _configure_sentry(app):
    SENTRY_KEY = app.config["SENTRY_DSN_KEY"]
    SENTRY_PROJECT = app.config["SENTRY_PROJECT"]

    #SENTRY_KEY = os.getenv("SENTRY_DSN_KEY")
    #SENTRY_PROJECT = os.getenv("SENTRY_PROJECT")
    app.sentry = Sentry(app, dsn=f"https://{SENTRY_KEY}@sentry.io/{SENTRY_PROJECT}")


def _configure_app(app, config=None):
    """Establishes a dev, staging, or prod configuration."""
    app.config.from_object(config)


def _configure_blueprints(app):
    """Configure all routing blueprints."""
    for bp in [api]:  # this allows us to add other modules easily.
        app.register_blueprint(bp)


def _configure_logging(app):
    """Environment-based logging setup."""
    _set_stdout_based_logging()
    if app.debug or app.testing:
        app.logger.setLevel(logging.INFO)
    else:
        app.logger.setLevel(logging.WARN)


def _set_stdout_based_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)


app = create_app(config=BaseConfig)


@app.route('/health-check', methods=['GET'])
@app.route('/_ah/health', methods=['GET'])
def health_check():
    """health check for api call
    A GET api call to this address will return a json payload
    with a OK value. This is created to have a simple test to
    ensure the api layer can be hit with no problems.
    Returns:
        json payload with 'OK' value
    """
    return jsonify(status="OK")


if __name__ == '__main__':
    app = create_app(config=BaseConfig)
    app.run()
