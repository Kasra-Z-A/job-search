"""The live demo at jobs.kasra.page: a static page plus a search backend.

`bundle` packs what the backend needs from a snapshot (slim job records,
embeddings, allowed field values) and the evaluation results for the page.
`search` runs one query the way `systems.filtered` does on the metadata
track; `handler` wraps it for AWS Lambda; `devserver` serves page and API
locally.
"""
