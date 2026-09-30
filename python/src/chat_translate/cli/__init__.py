"""``chat-translate`` command-line tools for UI message catalogs.

A thin layer over the SDK: argument parsing, file I/O and progress output. The
SDK never imports this package. Providers are built from the environment with
``config_from_env`` / ``completion_config_from_env``; see the "CLI" section of
the package README.
"""
