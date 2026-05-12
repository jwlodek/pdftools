import asyncio

import pytest
from pdftools.sample_environment import LinkamT96
from ophyd_async.core import init_devices, set_mock_value, callback_on_mock_put
from bluesky.run_engine import RunEngine
from bluesky import plan_stubs as bps

@pytest.fixture
def linkam(RE: RunEngine):
    with init_devices(mock=True):
        linkam = LinkamT96(prefix="TEST:LINKAM", name="linkam")
    return linkam

async def test_moving_linkam_fails_w_rr_zero(linkam: LinkamT96):
    """Test that moving the Linkam fails if the ramp rate is set to zero."""
    set_mock_value(linkam.ramp_rate, 0)
    with pytest.raises(ValueError, match="Ramp rate cannot be zero when calculating timeout."):
        await linkam.set(100)


async def test_moving_linkam_emits_updates(linkam: LinkamT96):
    """Test that moving the Linkam emits updates with the expected structure."""
    set_mock_value(linkam.temperature, 0)
    set_mock_value(linkam.ramp_rate, 100)
    set_mock_value(linkam.ramping, True)

    target = 50

    async def ramp_temperature(_: float):
        current = await linkam.temperature.get_value()
        if current < target:
            await asyncio.sleep(0.1)
            new_temp = current + 10
            set_mock_value(linkam.temperature, new_temp)
            if new_temp >= target:
                set_mock_value(linkam.ramping, False)

    callback_on_mock_put(linkam.setpoint, ramp_temperature)

    updates = []
    status = linkam.set(target)
    status.watch(lambda **kwargs: updates.append(kwargs))
    await status

    assert len(updates) == 5
    for update in updates:
        assert hasattr(update, "current")
        assert hasattr(update, "initial")
        assert hasattr(update, "target")
        assert hasattr(update, "name")
        assert hasattr(update, "unit")
        assert hasattr(update, "precision")