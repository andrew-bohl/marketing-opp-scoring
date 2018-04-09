"""api error module that contains various error classes

All of the error classes defined here represents the various
api errors that can occur.
"""


class ApiError(Exception):
    """The generic api error class"""
    def to_dict(self):
        """sets the message value"""
        return dict(message=self.message)


class ItemNotFoundError(ApiError):
    """Error class for when item not found"""
    status_code = 404
    message = "The requested resource was not found."


class BadRequestError(ApiError):
    """Error class for when bad request is made.
     This error can be raised when our clients make
      invalid calls to our api or when our code makes
       invalid calls against clients our api is dependant on"""
    status_code = 400

    def __init__(self, message):
        super(BadRequestError, self).__init__()
        self.message = message


class ServiceNotAvailableError(ApiError):
    """Error class for when service is not available. This error is raised
    when the api clients we depend on are temporarily unavailable"""
    status_code = 503

    def __init__(self, message):
        super(ServiceNotAvailableError, self).__init__()
        self.message = message
