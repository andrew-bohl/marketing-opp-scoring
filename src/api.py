"""API Routing module"""

from flask import Blueprint, jsonify, request

api = Blueprint('api', __name__, url_prefix='/lead-scoring')


@api.route('/score-leads', methods=['GET'])
def score_leads():
    # This will be where Tian's code gets called.
    request
    return jsonify(status=200)
