"""
pawel_aidb

Command-line tool to fetch random news items from the web and assess the likelihood
that a news text is fake using a lightweight AI model (TF–IDF + Logistic Regression).

Entry point: `main()`.
"""

from pawel_aidb.news_ai import cli_main as main  # re-export for console_script entry point

__all__ = ["main"]
