import numpy as np

import cnn_constants as C
import cnn_scenario as sc


def test_make_dataset_is_reproducible():
    ds1 = sc.make_dataset(10, seed=3, k=10, anchor_range=sc.full_anchor_range(10))
    ds2 = sc.make_dataset(10, seed=3, k=10, anchor_range=sc.full_anchor_range(10))
    np.testing.assert_array_equal(ds1.X, ds2.X)
    np.testing.assert_array_equal(ds1.y, ds2.y)


def test_make_dataset_has_all_classes_balanced():
    ds = sc.make_dataset(15, seed=1, k=10, anchor_range=sc.full_anchor_range(10))
    for label in range(C.N_CLASSES):
        assert np.sum(ds.y == label) == 15


def test_shape_stamp_fits_within_image_at_any_valid_anchor():
    k = 10
    lo, hi = sc.full_anchor_range(k)
    for label in range(C.N_CLASSES):
        rng = np.random.default_rng(0)
        for _ in range(20):
            img = sc.make_image(rng, label, k, (lo, hi), noise_std=0.0)
            assert img.shape == (k, k)


def test_center_anchor_range_is_subset_of_full_range():
    for k in (8, 10, 16):
        c_lo, c_hi = sc.center_anchor_range(k)
        f_lo, f_hi = sc.full_anchor_range(k)
        assert f_lo <= c_lo <= c_hi <= f_hi
