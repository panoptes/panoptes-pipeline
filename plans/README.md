# Plans

Working documents for the PANOPTES algorithm rebuild.

| Document | Short name | What it is |
|---|---|---|
| [algorithm-design.md](algorithm-design.md) | algorithm design | What the algorithm is, independent of implementation, and the architecture that follows |
| [conformance-audit.md](conformance-audit.md) | conformance audit | How far the code has drifted from the published algorithm, and every defect found |
| [improvement-plan.md](improvement-plan.md) | improvement plan | The plan for reaching 0.5% precision, with metrics, sequencing and open action items |
| [data-contract.md](data-contract.md) | data contract | What the pipeline reads and produces, where it is stored, and how POCS, `panoptes-utils`, `panoptes-pipeline` and `panoptes-data` agree on the vocabulary |

The published paper is a building block, not a specification. Start with the
algorithm design; the audit and plan describe the existing implementation and
may not survive a rebuild.

Reference: Gee et al., *On-sky Demonstration of Precision Photometry with Bayer
Color Filter Arrays*. Paper section numbers are cited directly, e.g. "paper
section 3.2.3".

## Conventions

- Cite a section as short name plus number: "conformance audit 5.3",
  "improvement plan 3.6", "data contract 2.3". Never a bare section number.
- These are living documents. When an item is done, **delete it** -- do not
  strike it through or annotate it "complete". Git history is the record of
  what was finished; these files state where things stand now.
- **These documents are the reasoning; the [issue
  tracker](https://github.com/panoptes/panoptes-pipeline/issues) is the state.**
  Why a thing is worth doing, and what is known about it, belongs here. Whether
  it is planned, in progress or done belongs there. Do not keep both.
- Anything needing a human decision is a GitHub issue with the `decision` label,
  under the Decisions milestone, filed immediately. Not raised in conversation
  and left to be remembered.
