from pathlib import Path

NOTEBOOK_FOLDER = Path(__file__).parent.parent / 'notebook'
SRC_FOLDER = Path(__file__).parent.parent / 'src'
DOCS_FOLDER = Path(__file__).parent.parent / 'docs'
ENV_FILE = Path(__file__).parent.parent / '.env'

print(ENV_FILE)