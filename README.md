# Jev Python command-line file classifier

Requires Python 3.10 or newer. Uses only the Python standard library; no pip install or web server is needed.

Set `TYPESAFE_API_KEY` in your environment or in a `.env` file beside `classify.py`. Use `.env.example` as a template. Optionally set `TYPESAFE_MODEL`; the default is `jev-latest`.

```powershell
python classify.py "C:\path\to\input.txt"
```

Successful output is only a JSON object containing probabilities from 0 to 1:

```json
{
  "medical": 0.95,
  "pii": 0.91,
  "credentials": 0.02,
  "financial": 0.04,
  "business": 0.03,
  "legal": 0.01,
  "security": 0.02,
  "other": 0.03
}
```

These numbers are illustrative, not a real classification. Categories are independent and may overlap; their probabilities do not have to sum to 1. `medical` includes substantive educational health information as well as patient records. `credentials` includes API keys, passwords, tokens, and private keys. The other categories cover private financial, business, legal/personnel, security, and otherwise confidential information.

To save the output:

```powershell
python classify.py "C:\path\to\input.txt" > probabilities.json
```

Only printable ASCII (bytes 32–126), tabs, CR, and LF are accepted. PDFs, images, binary content, non-ASCII text, BOMs, and recognised non-text formats are rejected before any Jev request. This includes an ASCII PDF renamed to `.txt`. Format checking is conservative, not a universal format detector. Empty files and files exceeding 32 KiB are rejected; there is no truncation or partial classification.

For unsupported files the tool exits with:

```text
Can't process non-text information. Only ASCII text files are supported.
```

Exit codes: **0** = probabilities returned, **2** = rejected input, **1** = file access, configuration, or API error. Errors go to stderr; stdout is empty on failure. Successful classification exits 0 even when high probabilities indicate sensitive content.

Accepted text is sent to the TypeSafe API in one request with eight independent Noul questions. There is no keyword or demo fallback. The tool does not save or print input text, matched secrets, or the API key. Probabilities are model judgments, not guarantees that a file is safe to share. Actual model quality requires evaluation with your API key and representative data.

Run tests with `python -m unittest -v test_classify`. Tests stub the remote API; they do not make paid calls.

This folder contains the standalone Python classifier and its tests.



## Included sample files

- `sensitive.txt`: invented personal, medical, financial, business and legal records, plus nonfunctional credential-shaped strings. No real identities or live secrets are included. Because the credentials are explicitly synthetic, Jev may assign them a lower probability.
- `non_sensitive.txt`: a public gardening guide with no personal records or confidential material.

```powershell
cd C:\Users\thoma\projects\jev-test
python classify.py sensitive.txt
python classify.py non_sensitive.txt
```

Set your API key in `.env` or the environment before classification. Both samples are ASCII. Model probabilities must be obtained from Jev; the file names do not determine the result.
