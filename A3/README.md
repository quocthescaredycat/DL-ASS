# Assignment 3 - M1

Selected dataset: [PairDX, Zenodo record 17452236](https://zenodo.org/records/17452236).
Task: six-class classification from a medical image and its caption.

- [Editable proposal](proposal.md)
- [Submission PDF](G09_A3_M1_Proposal.pdf)
- [Assignment page](../assignment3.html)

The archive contains 103,384 candidate records, rather than the 22,665 balanced pairs
advertised in the record. The proposal explains the measured distribution, remaining
pair checks, article-level split plan, models and compute budget. Instructor approval
is pending; no main experiments have been run.

To rebuild the PDF and assignment page after editing the proposal:

```powershell
python -m pip install reportlab markdown-it-py
python A3/build_proposal.py
```

Run from the repository root. The builder uses Windows Arial fonts and writes only
the PDF and assignment page; it creates no run logs. The raw dataset is not bundled.
