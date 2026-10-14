# Two submissions stuck in the queue and never scored (943711, 940120), plus a 500 error on upload

- Thread: https://www.codabench.org/forums/17429/2811/
- Started by: anulum
- Date created: Sep 25, 2026
- Pinned: no
- Posts: 4

---

## Post 1 — anulum

*Sept. 25, 2026, 3:57 a.m.*

Hello, and thank you for running the challenge.

Two of our submissions never reached scoring, and each one used up a daily slot. Could you please look into them?

1. **Submission 943711** (team *anulum fortis*, created 2026-09-24 20:39:47 UTC).
   - It has been in status **"Submitted"** ever since, without a scoring host assigned: `status_details` is empty.
   - Submissions by other teams created after it (for example 943766, 943823, 943905 and 943931) were scored normally in the meantime.
   - Its pointer and files are intact. Both files resolve on the Hugging Face hub at the pointer's revision with the expected SHA-256.
   - We tried to re-upload the same pointer on 2026-09-25 at about 02:53 UTC. The page answered: **"Submission upload failed. Server returned: 500 error"**.

2. **Submission 940120** (created 2026-09-23 00:01 UTC).
   - It stayed **"Running"** in ingestion on host `ousaisrvr` with empty logs, well past the phase's 5400 s execution limit.
   - Other teams' submissions made about an hour later finished normally.
   - We later removed it from our list.

Could you re-queue 943711, or cancel it and restore the slot it used? Could you also restore the slot 940120 used, if the platform allows it?

For completeness: submission 943412 (2026-09-24 16:46 UTC) was scored and shown on the board, but stayed "Running" for over 90 minutes and then turned "Cancelled". Its scores had already been written, so nothing was lost there. We mention it only in case it points to the same cause.

Thank you,
anulum fortis

---

## Post 2 — sophelio

*Sept. 25, 2026, 3:12 p.m.*

Hello anulum fortis,

There was some issues with the COdabecnh platform yesterday and hoping things have resolved today. Could you confirm if the scoring for the submissions is still taking a long time, as you mentioned? I see you were able to submit a few submissions successfully since yesterday.

Thanks and Best Regards
Ganatma

---

## Post 3 — anulum

*Sept. 25, 2026, 4:40 p.m.*

Hello Ganatma,

Thank you for following up. Everything works for us again, and scoring is fast: each of our three submissions today was scored within a few minutes.

For your records, in case it helps with the platform issue:

- 943711 stayed "Submitted" until today. Two re-uploads of the same zip file (about 02:53 and 05:24 UTC) failed with a 500 error at `PUT /api/datasets/completed/<uuid>/`, so no submission was created.
- At 11:07 UTC we uploaded the same pointer re-packed into a new zip file (identical content, different file bytes). It went through as 944539 and was scored normally, and 943711 then showed "Cancelled". We do not know whether the new file bytes or a cancellation on your side made the difference.
- 944731 and 944781 afterwards were also scored normally.

The only open point is that 940120 (23 September) and 943711 each used a daily slot without being scored. If the platform can restore those two slots we would be grateful, but it is not urgent.

Thank you again for your help,
anulum fortis

---

## Post 4 — sophelio

*Oct. 2, 2026, 3:48 p.m.*

Hello anulum fortis,

Thank you for the detailed follow-up, and sorry the slot question sat open.

**Your two slots are restored.** Codabench counts cancelled and removed submissions toward the
100-submission total, so we deleted 940120 and 943711 on the platform. That takes them out of the
count, and your total should now read two lower.

**What happened, for the record.** In late September, Codabench's servers intermittently answered
requests with server errors (codalab/codabench#2557). Our worker logs show exactly where each of
yours was hit:

- **940120:** the status update that tells Codabench to start scoring got a server error, so
  scoring never began.
- **943412:** the same failure at the last step. Its scores were already written and stand; only
  the final "Finished" update was lost, which is why it later showed "Cancelled".
- **943711, and the 500 on `PUT /api/datasets/completed/`:** both happen on Codabench's side,
  before a submission reaches our worker. We doubt the re-packed zip mattered; your new upload more
  likely just landed when the server was healthy.

**What we changed.** Our worker now retries those status updates when Codabench answers with a
server error, instead of giving up after one attempt. That closes the 940120 and 943412 failures
on our side. We can't fix the upload and dispatch errors, which are on Codabench's side.

**For the Final phase**, where each team has only three submissions: if one sticks or fails for a
reason on our side, email us and we will remove it so the slot comes back. The rules now say so.

Best regards,
Ganatma
