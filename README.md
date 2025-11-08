# Mahjong Card Builder

This project generates Mahjong card data (currently focused on the 2014 National Mah Jongg League card) and exports the hands and corresponding tile sets to a JSON file.

## Repository Layout

- `load_card.py` – Utility script that parses a source JSON (for example `nmjl_2014_claude.json`) and scaffolds a `generateYYYY.py` builder.  
  - **Important:** The `main()` call in `load_card.py` is intentionally commented out so the script will not overwrite the hand-tuned `generate2014.py`. Uncomment the call only if you need to regenerate the builder from scratch and are prepared to merge the generated output back into your custom changes.
- `generate2014.py` – Hand-edited generator that builds the 2014 card hands in memory, produces tile sets, and exports `card2014.json`.
- `hand.py`, `tile.py` – Supporting classes and tile mappings used by the generator.

## Installation

```bash
git clone <your-repo-url>
cd cardReader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## (Optional) Regenerating the Builder Skeleton

If you have an updated source JSON and need to recreate the builder script:

1. Place the JSON file (for example `nmjl_2014_claude.json`) in the repository root.
2. Temporarily uncomment the `main()` invocation at the bottom of `load_card.py`.
3. Run:
   ```bash
   python3 load_card.py
   ```
4. Review the regenerated `generate2014.py`, merge any desired changes, then re-comment the `main()` call to avoid future accidental overwrites.

## Generating the 2014 Card JSON

The generator script produces tile sets for every hand and writes the JSON export in the expected inline format.

```bash
python3 generate2014.py
```

Running the script prints a summary of every hand, the total number of tiles produced, and writes `card2014.json` at the repository root.

## Contributing

Contributions and refinements are welcome. Please open an issue or submit a pull request.

## License

MIT

