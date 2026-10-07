# Evidence atoms

Store evidence as JSON Lines (`.jsonl`), one claim-sized object per line.
Atoms should be short enough to retrieve independently and must retain an
exact page/section/text locator. Follow
`../schema/evidence_atom.schema.json`.

Only manually checked atoms should use `review_status: reviewed`.
