#!/usr/bin/env bash
MODE=${1:-async}
shift

if [ "$MODE" = "sync" ]; then
  SERVER="pytorchexample.server_app:app"
  CLIENT="pytorchexample.client_app:app"
else
  SERVER="pytorchexample.server_app_async:app"
  CLIENT="pytorchexample.client_app_async:app"
fi

cp pyproject.toml pyproject.toml.bak
sed -i '' "s|serverapp = .*|serverapp = \"$SERVER\"|" pyproject.toml
sed -i '' "s|clientapp = .*|clientapp = \"$CLIENT\"|" pyproject.toml

flwr run . --stream "$@"

mv pyproject.toml.bak pyproject.toml
