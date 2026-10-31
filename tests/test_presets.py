import itertools
from pathlib import Path
import tempfile
import unittest
import numpy as np
import yaml
from cinema.config import load_settings
from cinema.overlay_geometry import rotate, segment_pose, yaw_rotation

ROOT = Path(__file__).resolve().parents[1]


class PresetTests(unittest.TestCase):
    def test_camera_override_does_not_change_appearance_or_control(self):
        path = ROOT/'presets/blue_grid_orbit.yaml'
        before = load_settings(path, ROOT)
        after = load_settings(path, ROOT, camera='fixed_side')
        self.assertEqual(before['appearance'], after['appearance'])
        self.assertEqual(before['commands'], after['commands'])
        self.assertEqual(before['scene'], after['scene'])
        self.assertEqual(after['camera']['mode'], 'fixed')
        self.assertEqual(before['camera']['mode'], 'orbit')

    def test_independent_combinations(self):
        for appearance, commands, camera in itertools.product(
                ['spotlight', 'blue_grid'], ['none', 'overhead'], ['orbit_pullback', 'close_orbit', 'fixed_side']):
            with self.subTest(appearance=appearance, commands=commands, camera=camera):
                settings = load_settings(ROOT/'presets/orbit.yaml', ROOT,
                                         appearance=appearance, commands=commands, camera=camera)
                self.assertIsInstance(settings['appearance']['lighting'], dict)
                self.assertEqual(settings['commands']['enabled'], commands == 'overhead')
                self.assertEqual(settings['scene']['num_envs'], 4096)

    def test_relative_paths_and_inline_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'appearance.yaml').write_text('ground: {type: asset}\n')
            path = root/'shot.yaml'
            path.write_text(yaml.safe_dump({'appearance': './appearance.yaml', 'commands': {'enabled': False}, 'camera': {'mode': 'fixed'}}))
            result = load_settings(path, ROOT)
            self.assertEqual(result['appearance']['ground']['type'], 'asset')
            self.assertFalse(result['commands']['enabled'])

    def test_unknown_module_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            load_settings(ROOT/'presets/orbit.yaml', ROOT, appearance='does_not_exist')


class GeometryTests(unittest.TestCase):
    def test_segments_reach_endpoints_in_all_directions(self):
        rng = np.random.default_rng(42)
        vectors = np.r_[np.eye(3), -np.eye(3), rng.normal(size=(50, 3))]
        starts = rng.normal(size=vectors.shape)
        quaternions, scales = segment_pose(starts, starts+vectors, .018)
        np.testing.assert_allclose(rotate(quaternions, np.c_[scales[:, 0], np.zeros((len(scales), 2))]), vectors, atol=1e-12)
        np.testing.assert_allclose(scales[:, 1:], .018)

    def test_zero_segments_are_finite_and_have_zero_length(self):
        q, scale = segment_pose(np.zeros((2, 3)), np.zeros((2, 3)), .01)
        self.assertTrue(np.isfinite(q).all())
        np.testing.assert_array_equal(scale[:, 0], 0)

    def test_yaw_frame_ignores_pitch_and_roll(self):
        # A 90-degree yaw maps a forward command to world +Y.
        q = np.array([[np.sqrt(.5), 0., 0., np.sqrt(.5)]])
        yaw, flat = yaw_rotation(q)
        np.testing.assert_allclose(yaw, np.pi/2)
        np.testing.assert_allclose(rotate(flat, np.array([[1., 0., 0.]])), [[0., 1., 0.]], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
