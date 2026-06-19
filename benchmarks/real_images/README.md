# real_images — real-image evaluation slot

Drop real ArUco images here together with a `manifest.json` to enable the
`--real-dir` mode of `robustness.py`.

## Quick start

1. Copy `manifest.example.json` → `manifest.json`.
2. Populate the JSON array — one object per image.
3. Drop the image files into this directory (or use relative paths from here).
4. Run:

```
python benchmarks/robustness.py --real-dir benchmarks/real_images
```

## manifest.json schema

See `manifest.example.json` for the full example.  Each entry is:

| field           | type            | description                                                  |
|-----------------|-----------------|--------------------------------------------------------------|
| `path`          | string          | path relative to `manifest.json` (e.g. `"img_001.png"`)     |
| `dictionary`    | string          | `nf.Dict` name, e.g. `"DICT_4X4_50"`                        |
| `expected_ids`  | list of ints    | marker IDs that must be detected for the entry to pass       |
| `pose`          | object or null  | optional ground-truth pose metadata (not yet consumed)       |

## Notes

- Images are loaded as 8-bit grayscale; colour images are converted automatically.
- Missing image files are skipped with a warning; `manifest.json` itself must exist.
- No real images are shipped with the library — this slot is for your own captures.
