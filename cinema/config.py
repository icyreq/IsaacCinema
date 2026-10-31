"""Resolve independent appearance, command-overlay and camera modules."""
from pathlib import Path
import yaml

MODULES = ('appearance', 'commands', 'camera')


def load_settings(path, root, **overrides):
    """A module is either an inline mapping, a bundled name, or a YAML path.

    Bundled names resolve under presets/<module>/; explicit relative paths are
    relative to the shot file. CLI module choices replace the entire module.
    """
    path, root = Path(path).resolve(), Path(root).resolve()
    settings = yaml.safe_load(path.read_text())
    for section in MODULES:
        choice = overrides.get(section)
        value = settings[section] if choice is None else choice
        if isinstance(value, str):
            ref = Path(value)
            if ref.suffix or len(ref.parts) > 1:
                ref = ref if ref.is_absolute() else path.parent / ref
            else:
                ref = root / 'presets' / section / (value + '.yaml')
            value = yaml.safe_load(ref.read_text())
        if not isinstance(value, dict):
            raise ValueError(f'{section} must be a module name, YAML path or mapping')
        settings[section] = value
    return settings
