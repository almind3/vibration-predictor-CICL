import numpy as np
from app.extract_features import extract_features

def test_extract_features_length():
    s = np.random.randn(2048)
    feats = extract_features(s, 20000000)
    assert isinstance(feats, (list, tuple))
    assert len(feats) > 0