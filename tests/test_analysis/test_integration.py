from typing import Any

import numpy as np
import pytest
from pytest_mock import MockerFixture

from pdftools.analysis.integration import IntegrationSettings, subtract_background, integrate, average_images
from pyFAI.integrator.azimuthal import AzimuthalIntegrator

def test_subtract_background_invalid_shapes():
    raw_image = np.zeros((10, 10))
    background_image = np.zeros((5, 5))
    with pytest.raises(ValueError, match="Unmatched shape between two images"):
        subtract_background(raw_image, background_image)


def test_subtract_background_no_scaling():
    raw_image = np.array([[1, 2], [3, 4]])
    background_image = np.array([[0.5, 1], [1.5, 2]])
    expected = np.array([[0.5, 1], [1.5, 2]])
    result = subtract_background(raw_image, background_image)
    np.testing.assert_array_equal(result, expected)


def test_subtract_background_with_scaling():
    raw_image = np.array([[1, 2], [3, 4]])
    background_image = np.array([[0.5, 1], [1.5, 2]])
    background_scale = 2.0
    expected = np.array([[0, 0], [0, 0]])
    result = subtract_background(raw_image, background_image, background_scale)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(("integration_settings"), [
        ({"npt": 100}),
        ({"npt": 100, "unit": "2th_deg"}),
        ({})
    ]
)
def test_integrate_settings_used(mocker: MockerFixture, integration_settings: dict[str, Any]):
    mock_ai = mocker.MagicMock()
    mock_ai.integrate1d.return_value = (np.array([0, 1]), np.array([0, 1]))
    image = np.zeros((10, 10))
    _, settings = integrate(image, mock_ai, integration_settings=integration_settings)
    default_settings = IntegrationSettings()
    for attr in settings.__dataclass_fields__:
        expected_value = integration_settings.get(attr, getattr(default_settings, attr))
        assert getattr(settings, attr) == expected_value


@pytest.mark.parametrize(("weights"), [None, [0.5, 0.5], [0.2, 0.8]])
def test_average_images(weights):
    images = [np.ones((10, 10)), np.ones((10, 10)) * 2]
    expected = np.average(np.stack(images, axis=0), axis=0, weights=weights)
    result = average_images(images, weights=weights)
    np.testing.assert_array_equal(result, expected)