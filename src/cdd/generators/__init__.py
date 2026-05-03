"""Reference and built-in CDD generators.

Generators here are part of the CDD distribution. The Anthropic,
OpenAI, and fal.ai generators land in v0.2.0 (S3); for now this
module hosts the test/reference :class:`EchoGenerator` only.
"""

from cdd.generators.echo import EchoGenerator

__all__ = ["EchoGenerator"]
