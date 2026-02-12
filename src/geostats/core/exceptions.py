"""
Custom exception classes for the GeoStats library
"""


class GeoStatsError(Exception):

    pass


class ValidationError(GeoStatsError):

    pass


class FittingError(GeoStatsError):

    pass


class KrigingError(GeoStatsError):

    pass


class ConvergenceError(GeoStatsError):

    pass


class ModelError(GeoStatsError):

    pass


class DataError(GeoStatsError):

    pass


class SimulationError(GeoStatsError):

    pass
