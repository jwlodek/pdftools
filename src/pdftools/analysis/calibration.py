from bluesky_tiled_plugins import BlueskyRun
from pathlib import Path
from pyFAI.io.ponifile import PoniFile


def export_calibration_poni_file_from_run(run: BlueskyRun, output_path: Path):
    """Export a calibration file (poni) from a bluesky run.

    This function assumes that the run contains a stream with the necessary metadata
    to construct a poni file. The metadata should be in the 'primary' stream and
    should include keys such as 'dist', 'poni1', 'poni2', 'wavelength', etc.

    Parameters
    ----------
    run : BlueskyRun
        The bluesky run containing the calibration metadata.
    output_path : Path
        The path where the poni file will be saved.
    """

    start = run.start

    # Extract necessary metadata from the start document

    pf = PoniFile()