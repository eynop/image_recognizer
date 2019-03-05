import pytest
import recognizer as rec


factory = rec.detector.DetectorFactory()


def test_get_detector_case(): 
    detector = factory.get_detector('SIFT')
    detector = factory.get_detector('sift')
    detector = factory.get_detector('sIfT')


def test_raises_exception():
    with pytest.raises(rec.detector.DetectorNotFound):
        detector = factory.get_detector('non_existing')


def test_sift():
    detector = factory.get_detector('sift')
    assert isinstance(detector, rec.detector.SiftDetector)
    

def test_surf():
    detector = factory.get_detector('surf')
    assert isinstance(detector, rec.detector.SurfDetector)


def test_kaze():
    detector = factory.get_detector('kaze')
    assert isinstance(detector, rec.detector.KazeDetector)


def test_akaze():
    detector = factory.get_detector('akaze')
    assert isinstance(detector, rec.detector.AkazeDetector)


def test_brisk():
    detector = factory.get_detector('brisk')
    assert isinstance(detector, rec.detector.BriskDetector)


def test_orb():
    detector = factory.get_detector('orb')
    assert isinstance(detector, rec.detector.OrbDetector)

