#!/usr/bin/env bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

in-venv() (
  . "$HERE/venv/bin/activate"
  "$@"
)

init() {
  python -m venv "$HERE/venv"
  in-venv pip install --upgrade pip
  in-venv pip install -r "$HERE/requirements.txt"
}

make() {
  in-venv command make "$@"
}

view() {
  in-venv python -m webbrowser -t "file://$HERE/how-many-and-when.html"
}

"$@"
