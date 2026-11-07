import resource

from nanoeval.setup import _raise_open_file_limit


def test_raise_open_file_limit_caps_at_finite_hard_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(resource, "getrlimit", lambda _resource: (1_024, 4_096))
    monkeypatch.setattr(
        resource,
        "setrlimit",
        lambda which, limits: calls.append((which, limits)),
    )

    _raise_open_file_limit()

    assert calls == [(resource.RLIMIT_NOFILE, (4_096, 4_096))]


def test_raise_open_file_limit_preserves_unlimited_hard_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        resource,
        "getrlimit",
        lambda _resource: (1_024, resource.RLIM_INFINITY),
    )
    monkeypatch.setattr(
        resource,
        "setrlimit",
        lambda which, limits: calls.append((which, limits)),
    )

    _raise_open_file_limit()

    assert calls == [
        (resource.RLIMIT_NOFILE, (131_072, resource.RLIM_INFINITY))
    ]


def test_raise_open_file_limit_does_not_lower_existing_soft_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        resource,
        "getrlimit",
        lambda _resource: (200_000, resource.RLIM_INFINITY),
    )
    monkeypatch.setattr(
        resource,
        "setrlimit",
        lambda which, limits: calls.append((which, limits)),
    )

    _raise_open_file_limit()

    assert calls == []
