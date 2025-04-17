# Copyright © SixtyFPS GmbH <info@slint.dev>
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

# This file is automatically generated by pyo3_stub_gen
# ruff: noqa: E501, F401

import builtins
import datetime
import os
import pathlib
import typing
from typing import Any, List
from collections.abc import Callable
from enum import Enum, auto

class RgbColor:
    red: int
    green: int
    blue: int

class RgbaColor:
    red: int
    green: int
    blue: int
    alpha: int

class Color:
    red: int
    green: int
    blue: int
    alpha: int
    def __new__(
        cls,
        maybe_value: typing.Optional[
            str | RgbaColor | RgbColor | typing.Dict[str, int]
        ] = None,
    ) -> "Color": ...
    def brighter(self, factor: float) -> "Color": ...
    def darker(self, factor: float) -> "Color": ...
    def transparentize(self, factor: float) -> "Color": ...
    def mix(self, other: "Image", factor: float) -> "Color": ...
    def with_alpha(self, alpha: float) -> "Color": ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...

class Brush:
    color: Color
    def __new__(cls, maybe_value: typing.Optional[Color]) -> "Brush": ...
    def is_transparent(self) -> bool: ...
    def is_opaque(self) -> bool: ...
    def brighter(self, factor: float) -> "Brush": ...
    def darker(self, factor: float) -> "Brush": ...
    def transparentize(self, amount: float) -> "Brush": ...
    def with_alpha(self, alpha: float) -> "Brush": ...
    def __eq__(self, other: object) -> bool: ...

class Image:
    r"""
    Image objects can be set on Slint Image elements for display. Construct Image objects from a path to an
    image file on disk, using `Image.load_from_path`.
    """

    size: tuple[int, int]
    width: int
    height: int
    path: typing.Optional[pathlib.Path]
    def __new__(
        cls,
    ) -> "Image": ...
    @staticmethod
    def load_from_path(path: str | os.PathLike[Any] | pathlib.Path) -> "Image":
        r"""
        Loads the image from the specified path. Returns None if the image can't be loaded.
        """
        ...

    @staticmethod
    def load_from_svg_data(data: typing.Sequence[int]) -> "Image":
        r"""
        Creates a new image from a string that describes the image in SVG format.
        """
        ...

class TimerMode(Enum):
    SingleShot = auto()
    Repeated = auto()

class Timer:
    running: bool
    interval: datetime.timedelta
    def __new__(
        cls,
    ) -> "Timer": ...
    def start(
        self, mode: TimerMode, interval: datetime.timedelta, callback: typing.Any
    ) -> None: ...
    @staticmethod
    def single_shot(duration: datetime.timedelta, callback: typing.Any) -> None: ...
    def stop(self) -> None: ...
    def restart(self) -> None: ...

def set_xdg_app_id(app_id: str) -> None: ...
def run_event_loop() -> None: ...
def quit_event_loop() -> None: ...

class PyModelBase:
    def init_self(self, *args: Any) -> None: ...
    def row_count(self) -> int: ...
    def row_data(self, row: int) -> typing.Optional[Any]: ...
    def set_row_data(self, row: int, value: Any) -> None: ...
    def notify_row_changed(self, row: int) -> None: ...
    def notify_row_removed(self, row: int, count: int) -> None: ...
    def notify_row_added(self, row: int, count: int) -> None: ...

class PyStruct(Any): ...

class ValueType(Enum):
    Void = auto()
    Number = auto()
    String = auto()
    Bool = auto()
    Model = auto()
    Struct = auto()
    Brush = auto()
    Image = auto()

class DiagnosticLevel(Enum):
    Error = auto()
    Warning = auto()

class PyDiagnostic:
    level: DiagnosticLevel
    message: str
    line_number: int
    column_number: int
    source_file: typing.Optional[str]

class ComponentInstance:
    def show(self) -> None: ...
    def hide(self) -> None: ...
    def run(self) -> None: ...
    def invoke(self, callback_name: str, *args: Any) -> Any: ...
    def invoke_global(
        self, global_name: str, callback_name: str, *args: Any
    ) -> Any: ...
    def set_property(self, property_name: str, value: Any) -> None: ...
    def get_property(self, property_name: str) -> Any: ...
    def set_callback(
        self, callback_name: str, callback: Callable[..., Any]
    ) -> None: ...
    def set_global_callback(
        self, global_name: str, callback_name: str, callback: Callable[..., Any]
    ) -> None: ...
    def set_global_property(
        self, global_name: str, property_name: str, value: Any
    ) -> None: ...
    def get_global_property(self, global_name: str, property_name: str) -> Any: ...

class ComponentDefinition:
    def create(self) -> ComponentInstance: ...
    name: str
    globals: list[str]
    functions: list[str]
    callbacks: list[str]
    properties: dict[str, ValueType]
    def global_functions(self, global_name: str) -> list[str]: ...
    def global_callbacks(self, global_name: str) -> list[str]: ...
    def global_properties(self, global_name: str) -> typing.Dict[str, ValueType]: ...

class CompilationResult:
    component_names: list[str]
    diagnostics: list[PyDiagnostic]
    named_exports: list[typing.Tuple[str, str]]
    structs_and_enums: typing.Dict[str, PyStruct]
    def component(self, name: str) -> ComponentDefinition: ...

class Compiler:
    include_paths: list[os.PathLike[Any] | pathlib.Path]
    library_paths: dict[str, os.PathLike[Any] | pathlib.Path]
    translation_domain: str
    style: str
    def build_from_path(
        self, path: os.PathLike[Any] | pathlib.Path
    ) -> CompilationResult: ...
    def build_from_source(
        self, source: str, path: os.PathLike[Any] | pathlib.Path
    ) -> CompilationResult: ...
