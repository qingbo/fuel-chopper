#!/usr/bin/env python3
"""Set a USB-connected iPhone near Australia's best selected fuel price.

Examples:
    uv run fuel-chopper --fuel U91
    uv run fuel-chopper --state VIC --fuel Diesel --yes
    uv run fuel-chopper --stop
"""

# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 The fuel-chopper contributors
#
# Derived from GeoPort <https://github.com/davesc63/GeoPort>.
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

try:
    from pymobiledevice3.bonjour import DEFAULT_BONJOUR_TIMEOUT
    from pymobiledevice3.exceptions import AccessDeniedError
    from pymobiledevice3.lockdown import create_using_usbmux
    from pymobiledevice3.remote.remote_service_discovery import RemoteServiceDiscoveryService
    from pymobiledevice3.remote.tunnel_service import (
        CoreDeviceTunnelProxy,
        create_core_device_tunnel_service_using_rsd,
    )
    from pymobiledevice3.remote.utils import (
        get_rsds,
        resume_remoted_if_required,
        stop_remoted_if_required,
    )
    from pymobiledevice3.services.dvt.instruments.dvt_provider import DvtProvider
    from pymobiledevice3.services.dvt.instruments.location_simulation import LocationSimulation
    from pymobiledevice3.services.simulate_location import DtSimulateLocation
    from pymobiledevice3.usbmux import list_devices
except ImportError as error:
    print(
        "pymobiledevice3 is required. Install it with: "
        "uv sync",
        file=sys.stderr,
    )
    raise SystemExit(2) from error


API_URL = "https://projectzerothree.info/api.php?format=json"
STATE_CHOICES = ("ACT", "NSW", "QLD", "VIC", "WA")
FUEL_CHOICES = ("E10", "U91", "U95", "U98", "Diesel", "LPG")
EARTH_RADIUS_METRES = 6_371_000
MIN_OFFSET_METRES = 100
MAX_OFFSET_METRES = 900


@dataclass(frozen=True)
class FuelPrice:
    name: str
    fuel_type: str
    price: float
    suburb: str
    state: str
    postcode: str
    latitude: float
    longitude: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set a USB-connected iPhone near the best selected Australian fuel price."
    )
    parser.add_argument(
        "--state",
        type=str.upper,
        choices=STATE_CHOICES,
        help="Limit the result to one state (default: all available Australian prices).",
    )
    parser.add_argument(
        "--fuel",
        type=normalise_fuel,
        choices=FUEL_CHOICES,
        help="Fuel type to compare. Required unless --stop is used.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Set the nearby location without asking for confirmation.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Clear the simulated location on the connected device and exit.",
    )
    return parser.parse_args()


def normalise_fuel(value: str) -> str:
    return next((fuel for fuel in FUEL_CHOICES if fuel.lower() == value.lower()), value)


def fetch_best_price(state: str | None, fuel_type: str) -> FuelPrice:
    try:
        with urlopen(API_URL, timeout=20) as response:  # noqa: S310 -- fixed HTTPS URL
            payload = json.load(response)
    except (OSError, URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not fetch fuel prices: {error}") from error

    region_name = state or "All"
    region = next(
        (item for item in payload.get("regions", []) if item.get("region") == region_name),
        None,
    )
    if region is None:
        raise RuntimeError(f"The fuel API has no price data for {region_name}.")

    price = next(
        (item for item in region.get("prices", []) if item.get("type") == fuel_type),
        None,
    )
    if price is None:
        raise RuntimeError(f"The fuel API has no {fuel_type} price for {region_name}.")

    try:
        return FuelPrice(
            name=str(price["name"]),
            fuel_type=str(price["type"]),
            price=float(price["price"]),
            suburb=str(price["suburb"]),
            state=str(price["state"]),
            postcode=str(price["postcode"]),
            latitude=float(price["lat"]),
            longitude=float(price["lng"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("The fuel API returned an incomplete location.") from error


def print_price(price: FuelPrice) -> None:
    print("Best price found")
    print(f"  Fuel:     {price.fuel_type}")
    print(f"  Price:    {price.price:.1f} c/L")
    print(f"  Station:  {price.name}")
    print(f"  Location: {price.suburb}, {price.state} {price.postcode}")
    print(f"  Coords:   {price.latitude:.6f}, {price.longitude:.6f}")


def nearby_coordinate(latitude: float, longitude: float) -> tuple[float, float, int]:
    """Return a random point 100--900 m from the supplied coordinate."""
    distance = random.randint(MIN_OFFSET_METRES, MAX_OFFSET_METRES)
    bearing = random.uniform(0, 2 * math.pi)
    angular_distance = distance / EARTH_RADIUS_METRES
    start_latitude = math.radians(latitude)
    start_longitude = math.radians(longitude)

    target_latitude = math.asin(
        math.sin(start_latitude) * math.cos(angular_distance)
        + math.cos(start_latitude) * math.sin(angular_distance) * math.cos(bearing)
    )
    target_longitude = start_longitude + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(start_latitude),
        math.cos(angular_distance) - math.sin(start_latitude) * math.sin(target_latitude),
    )
    return math.degrees(target_latitude), math.degrees(target_longitude), distance


async def require_usb_device() -> Any:
    usb_devices = [device for device in await list_devices() if device.connection_type == "USB"]
    if not usb_devices:
        raise RuntimeError("No physically connected iPhone found. Connect one by USB and try again.")
    if len(usb_devices) > 1:
        raise RuntimeError("More than one USB-connected iPhone found. Disconnect all but one and try again.")
    return usb_devices[0]


async def require_developer_mode(lockdown: Any) -> None:
    if not await lockdown.get_developer_mode_status():
        raise RuntimeError(
            "Developer Mode is disabled on the iPhone. Enable it in Settings > Privacy & Security."
        )


def ios_major_version(lockdown: Any) -> int:
    version = str(lockdown.product_version)
    try:
        return int(version.split(".", maxsplit=1)[0])
    except ValueError as error:
        raise RuntimeError(f"Could not determine the iPhone's iOS version ({version!r}).") from error


def require_macos_tunnel_privileges() -> None:
    """iOS 17+ tunnels must temporarily suspend macOS's remoted service."""
    if sys.platform == "darwin" and os.geteuid() != 0:
        raise RuntimeError(
            "iOS 17+ location simulation on macOS needs administrator privileges. "
            "Run 'uv sync' first, then rerun this command as "
            "'sudo .venv/bin/fuel-chopper ...' from the project directory."
        )


async def run_modern_location(lockdown: Any, udid: str, coordinate: tuple[float, float] | None) -> None:
    """Set or clear a location on iOS 17+ while a USB tunnel is open."""
    version = str(lockdown.product_version)
    try:
        minor_version = int(version.split(".")[1]) if "." in version else 0
    except ValueError as error:
        raise RuntimeError(f"Could not determine the iPhone's iOS version ({version!r}).") from error
    use_quic = version.startswith("17.") and minor_version <= 3

    stop_remoted_if_required()
    tunnel_proxy = None
    if use_quic:
        rsds = await get_rsds(DEFAULT_BONJOUR_TIMEOUT, udid=udid)
        rsd = next(iter(rsds), None)
        if rsd is None:
            raise RuntimeError("Could not discover the USB device's tunnel service.")
        tunnel_service = await create_core_device_tunnel_service_using_rsd(rsd, autopair=True)
        tunnel_context = tunnel_service.start_quic_tunnel()
    else:
        tunnel_proxy = await CoreDeviceTunnelProxy.create(lockdown)
        tunnel_context = tunnel_proxy.start_tcp_tunnel()

    try:
        async with tunnel_context as tunnel:
            resume_remoted_if_required()
            async with RemoteServiceDiscoveryService((tunnel.address, tunnel.port)) as rsd:
                async with DvtProvider(rsd) as dvt, LocationSimulation(dvt) as simulation:
                    if coordinate is None:
                        await simulation.clear()
                        print("Simulated location cleared.")
                        return

                    await simulation.set(*coordinate)
                    print("Location set. Press Ctrl-C to clear it and exit.")
                    try:
                        await asyncio.Event().wait()
                    finally:
                        await simulation.clear()
                        print("Simulated location cleared.")
    finally:
        resume_remoted_if_required()
        if tunnel_proxy is not None:
            await tunnel_proxy.close()


async def run_legacy_location(lockdown: Any, coordinate: tuple[float, float] | None) -> None:
    """Set or clear a location on iOS 16 and earlier."""
    simulation = DtSimulateLocation(lockdown)
    if coordinate is None:
        await simulation.clear()
        print("Simulated location cleared.")
        return

    await simulation.clear()
    await simulation.set(*coordinate)
    print("Location set. Press Ctrl-C to clear it and exit.")
    try:
        await asyncio.Event().wait()
    finally:
        await simulation.clear()
        print("Simulated location cleared.")


async def set_or_clear_location_async(
    coordinate: tuple[float, float] | None, device: Any | None = None
) -> None:
    device = device or await require_usb_device()
    lockdown = await create_using_usbmux(
        device.serial,
        connection_type=device.connection_type,
        autopair=True,
    )
    try:
        await require_developer_mode(lockdown)
        if ios_major_version(lockdown) >= 17:
            require_macos_tunnel_privileges()
            await run_modern_location(lockdown, device.serial, coordinate)
        else:
            await run_legacy_location(lockdown, coordinate)
    finally:
        await lockdown.close()


def set_or_clear_location(coordinate: tuple[float, float] | None, device: Any | None = None) -> None:
    asyncio.run(set_or_clear_location_async(coordinate, device))


def confirm() -> bool:
    try:
        return input("Set the location near this station? [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def main() -> int:
    arguments = parse_arguments()
    if arguments.stop:
        try:
            set_or_clear_location(None)
        except (RuntimeError, OSError, AccessDeniedError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        return 0

    if arguments.fuel is None:
        print("Error: --fuel is required unless --stop is used.", file=sys.stderr)
        return 2

    try:
        price = fetch_best_price(arguments.state, arguments.fuel)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_price(price)
    try:
        device = asyncio.run(require_usb_device())
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Using USB device {device.serial}.")

    if not arguments.yes and not confirm():
        print("Location was not changed.")
        return 0

    latitude, longitude, distance = nearby_coordinate(price.latitude, price.longitude)
    print(f"Using a random point {distance} m from the station: {latitude:.6f}, {longitude:.6f}")
    try:
        set_or_clear_location((latitude, longitude), device)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except (RuntimeError, OSError, AccessDeniedError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
