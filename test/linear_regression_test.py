import init_tests
import pytest
from linear_regression import *
import numpy as np

class TestLinearRegression():

    @pytest.fixture
    def solution(self):
        return np.array([ 1.596163,  1.265087, 0.9449, 0.614449, 0.26072 , -0.021276, -0.399424])

    @pytest.fixture
    def x(self):
        return np.array([[-0.99768], [-0.69574], [-0.40373], [-0.10236], [0.22024], [0.47742], [0.82229]])

    @pytest.fixture
    def y(self):
        return np.array([2.0885, 1.1646, 0.3287, 0.46013, 0.44808, 0.10013, -0.32952])

    class TestBatchGradient():

        @pytest.fixture
        def subject(self):
            return BatchGradient()

        def test_result(self, subject, x, y, solution):
            subject.fit(x, y)
            np.testing.assert_array_almost_equal(subject.predict(x), solution, decimal=1)


    class TestLeastSquares():

        @pytest.fixture
        def subject(self):
            return LeastSquares()

        def test_result(self, subject, x, y, solution):
            subject.fit(x, y)
            np.testing.assert_array_almost_equal(subject.predict(x), solution, decimal=1)
