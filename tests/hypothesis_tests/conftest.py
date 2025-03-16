import os

from hypothesis import HealthCheck, Verbosity, settings

# Configure Hypothesis profiles
settings.register_profile(
    "ci",
    max_examples=10,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large
    ],
    verbosity=Verbosity.verbose
)

settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow]
)

# Load CI profile when running in GitHub Actions
if os.environ.get("GITHUB_ACTIONS") == "true":
    settings.load_profile("ci")
else:
    settings.load_profile("dev")

# No clean_registry fixture here to preserve model registrations
