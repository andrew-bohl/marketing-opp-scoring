"""API Routing module"""
from datetime import datetime, time, timedelta
import threading

from flask import Blueprint, jsonify, request, current_app

from src import model

api = Blueprint('api', __name__, url_prefix='/opp-scoring')

DATE_FMT = "%Y-%m-%d"

def _format_date(payload, key, default_time):
    _date = payload.get(key, default_time)
    _date = datetime.strptime(_date, DATE_FMT)
    return _date


@api.route('/score-leads', methods=['GET', 'POST'])
def score_leads():
    """
    Scores marketing leads over a given date range from the latest model.

    If run from AppEngine cron, this is called via a GET request and we set
    the default date range to:
        start_date = 30 days ago
        end_date = current date
    because model is trained on leads >= 30 days ago, since last 30 days haven't had
    the benefit of a full conversion cycle
    called via a POST request with valid start_date and end_dates, we'll use
    those dates instead.
    """
    midnight = datetime.combine(datetime.today(), time.min)
    default_start = (midnight - timedelta(days=14)).strftime(DATE_FMT)
    default_end = midnight.strftime(DATE_FMT)

    payload = request.get_json() or {}
    start_date = _format_date(payload, "start_date", default_start)
    end_date = _format_date(payload, "end_date", default_end)
    flask_config = current_app.config

    t = threading.Thread(group=None,
                         target=model.infer,
                         name="lead-scoring",
                         args=(start_date.date(), end_date.date(), flask_config))
    t.start()
    return jsonify(status="Started",)


@api.route('/train-model', methods=['GET', 'POST'])
def train_model():
    """
    Retrains the marketing leads model for a given time frame.

    If run from AppEngine cron, this is called via a GET request and we set
    the default date range to:
        start_date = 90 days ago
        end_date = 30 days ago
    because leads younger than 30 days haven't had time to convert. If this is
    called via a POST request with valid start_date and end_dates, we'll use
    those dates instead.
    """

    midnight = datetime.combine(datetime.today(), time.min)
    default_start = (midnight - timedelta(days=210)).strftime(DATE_FMT)
    default_end = (midnight - timedelta(days=21)).strftime(DATE_FMT)

    payload = request.get_json() or {}
    start_date = _format_date(payload, "start_date", default_start)
    end_date = _format_date(payload, "end_date", default_end)
    flask_config = current_app.config

    t = threading.Thread(group=None,
                         target=model.train,
                         name="retraining-model",
                         args=(start_date, end_date, flask_config))
    t.start()
    return jsonify(status="Retraining")
