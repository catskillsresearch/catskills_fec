# FEC.pdf source note

`docs/FEC.pdf` is built from **`fec.md`’s table of contents**:

```bash
python3 scripts/build_fec_pdf.py
```

The PDF contents list matches the first two levels of the nested list in `fec.md` (sections + subsections named there). Deeper headings in chapter files still appear in the body but not in that contents list (`tocdepth=2`).
