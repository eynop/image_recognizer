import pytest
import recognizer as rec


factory = rec.matcher.MatcherFactory()


def test_get_matcher_case():
    matcher = factory.get_matcher('FLANN')
    matcher = factory.get_matcher('flann')
    matcher = factory.get_matcher('fLaNN')


def test_raises_exception():
    with pytest.raises(rec.matcher.MatcherNotFound):
        matcher = factory.get_matcher('non_existing')


def test_flann():
    matcher = factory.get_matcher('flann')
    assert isinstance(matcher, rec.matcher.FlannMatcher)


def test_bf():
    matcher = factory.get_matcher('bf')
    assert isinstance(matcher, rec.matcher.BfMatcher)


def test_bfl2():
    matcher = factory.get_matcher('bfl2')
    assert isinstance(matcher, rec.matcher.BfL2Matcher)


def test_bfhamming():
    matcher = factory.get_matcher('bfh')
    assert isinstance(matcher, rec.matcher.BfHammingMatcher)

