import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from data.prompt_dataset import PromptDataSet
from scripts.image_output import existing_image_names, reserve_image_stem, write_image


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_test_output_namer():
    # Exercise the test entry point's naming without importing the CLIP tokenizer.
    path = REPO_ROOT / "test_from_dataset.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_output_names"
    )
    namespace = {
        "Path": Path,
        "existing_image_names": existing_image_names,
        "reserve_image_stem": reserve_image_stem,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_output_names"]


class ImageOutputTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "融合输出"
        self.output.mkdir()
        self.image = np.array([[[11, 129, 254], [255, 0, 21]]], dtype=np.uint8)

    def test_png_round_trips_on_unicode_path(self):
        path = self.output / "融合图.001.png"
        write_image(path, self.image)
        decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, self.image)

    def test_existing_image_is_preserved(self):
        path = self.output / "sample.png"
        write_image(path, self.image)
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_image(path, np.zeros_like(self.image))
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.output.iterdir()), [path])

    def test_inference_reserves_native_png_and_existing_names(self):
        output_names = load_test_output_namer()
        paths = ["sample.jpg", "sample.png", "sample__2.jpg", "solo.jpg"]
        original_names = output_names(paths, self.output)
        self.assertEqual(original_names, {
            "sample.png": "sample.png",
            "sample.jpg": "sample__3.png",
            "sample__2.jpg": "sample__2.png",
            "solo.jpg": "solo.png",
        })
        for name in original_names.values():
            (self.output / name).touch()
        repeated_names = output_names(paths, self.output)
        self.assertTrue(set(original_names.values()).isdisjoint(repeated_names.values()))
        self.assertEqual(len(set(repeated_names.values())), len(paths))

    def test_validation_keeps_dots_and_reserves_each_three_file_group(self):
        first, second = [], []
        for task, paths, filenames in (
            ("first", first, ("scene.001.png", "scene.002.png", "scene.001vis.png")),
            ("second", second, ("scene.001.png",)),
        ):
            folder = self.output / task
            folder.mkdir()
            for filename in filenames:
                path = folder / filename
                write_image(path, self.image)
                paths.append(path)

        empty = ([], [])
        dataset = PromptDataSet(empty, (first, first), empty, (second, second),
                                empty, empty, empty, empty, phase="val")
        names = [dataset[index][-1] for index in range(len(dataset))]
        self.assertEqual(names, ["scene.001", "scene.002", "scene.001vis", "scene.001"])

        (self.output / "SCENE.002.PNG").touch()
        used = existing_image_names(self.output)
        endings = (".png", "vis.png", "ir.png")
        stems = [reserve_image_stem(name, endings, used) for name in names]
        self.assertEqual(stems, ["scene.001", "scene.002__2", "scene.001vis__2", "scene.001__2"])
        self.assertEqual(len({stem + ending for stem in stems for ending in endings}), 12)
