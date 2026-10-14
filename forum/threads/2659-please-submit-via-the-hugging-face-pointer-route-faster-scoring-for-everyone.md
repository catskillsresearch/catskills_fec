# Please submit via the Hugging Face pointer route — faster scoring for everyone

- Thread: https://www.codabench.org/forums/17429/2659/
- Started by: sophelio
- Date created: Aug 03, 2026
- Pinned: yes
- Posts: 1

---

## Post 1 — sophelio

*Aug. 3, 2026, 4:28 p.m.*

Hi all,

Short version: **if you can, submit using the Hugging Face pointer route rather than uploading your `.npz` files straight to Codabench.** It makes scoring dramatically faster — for you, and for everyone else waiting behind you.

### Why it matters

Measured from the scoring machine, Codabench's file storage sustains about **0.5 MB/s**. The Hugging Face CDN delivers about **50 MB/s** to that same machine. So a ~2 GB direct upload spends roughly **an hour in transfer before scoring even starts**. The pointer route takes under a minute.

The scoring worker runs **one job at a time**. That hour isn't only your wait — it is queue time everyone shares. A single large direct upload can hold up every submission behind it.

With the pointer route, what you upload to Codabench is a few hundred bytes. Your predictions travel from Hugging Face instead, and they stay in a **private** repo that no other team can see.

### How to do it

Three steps. The first two happen once; the third every time you submit.

**Step 1 · Create your predictions repo**

```bash
uv run python submission_skeleton.py --max-shots 5 --repo your-username/fusion-eq-predictions
```

Use your own username as the namespace unless you have write access to an organization. This builds 5 shots as a quick format check and creates the private repo. It prints the repo URL and the exact recipe for step 2.

**Step 2 · Create a fine-grained read token**

At [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **New token** → **Fine-grained** → under *Repository permissions* pick the repo you just created → tick **only** "Read access to contents of selected repos".

Steps 1 and 2 are separate because Hugging Face cannot scope a token to a repo that does not exist yet.

**Step 3 · Build the real submission**

```bash
uv run python submission_skeleton.py --max-shots 0 \
    --repo your-username/fusion-eq-predictions \
    --read-token hf_...
```

This ends with `==> Upload this to Codabench: .../submission_pointer.zip`. That file is your entire submission — upload it on the **Submit** tab.

### Three things that trip people up

**`--max-shots 0` is required.** The default is 5. A submission built without this flag is structurally valid but scores almost everything as missing. This is the single most common way to waste a submission slot.

**The build is not stuck.** A full run streams the public test fold once to build, again to validate, and only then uploads. It takes a while.

**There are two different tokens.** The **write** token uploads your `.npz` (via `huggingface-cli login`) and never leaves your machine. The **fine-grained read** token goes inside your submission so the scorer can read your private repo. The script deliberately **refuses** a write token or a classic read token in the `--read-token` slot — that is not a bug. Both grant far more than the scorer needs, and that token travels with your submission. Revoke it when the competition ends.

### Not using the starter kit?

You don't have to. The zip you upload contains a single file, `manifest.json`, at the archive root:

```json
{"repo_id": "your-username/fusion-eq-predictions",
 "revision": "0123456789abcdef0123456789abcdef01234567",
 "token": "hf_..."}
```

`revision` must be the full **40-character commit SHA** — not `main`, not a tag. The `.npz` files must sit at the **root** of a **dataset** repo, named exactly `diii_d_public_test.npz` and `mast_public_test.npz` (in the Final phase, `diii_d_private_test.npz` / `mast_private_test.npz`).

Full details of the key names, shapes, and scoring contract are on the **Submission** page, and the starter kit is at
**https://github.com/Sophelio/fusion-equilibrium-challenge-starter**

### Direct upload still works

To be clear: this is a request, not a rule change. Uploading the `.npz` in a zip directly is **fully supported and scores identically**. Just expect about an hour of transfer per submission, and note that it counts against your 15 GB Codabench storage quota — the pointer route costs essentially none.

### Running into problems?

Please email **fusionec26@sophelio.io** with the details of what you're hitting — the command you ran, the error message, and your team/username. We'll help you get unblocked.

Thanks, and good luck!

— The Fusion Equilibrium Challenge organizers
