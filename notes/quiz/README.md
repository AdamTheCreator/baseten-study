# Quiz — Baseten SA Interview Prep

An interactive, instructor-graded quiz that drills the fundamentals from
Philip Kiely's *Inference Engineering* (Baseten Books, 2026) **and** how an
SA applies them in front of a customer.

## How to run it

Open a Claude Code session in this repo and say:

> "Quiz me. Resume from my scorecard."

Claude will:
1. Read `scorecard.md` to see where you left off and what's weak.
2. Ask one question at a time, interview-style.
3. Grade each answer (what you nailed / what you missed), teach the gap,
   then escalate.
4. Append the result to `scorecard.md`.

## Commands you can say mid-quiz

- `skip` — move on, marked unanswered
- `hint` — get a nudge without the answer
- `teach me first` — get taught the concept before being asked
- `harder` / `easier` — adjust difficulty
- `score` — running tally + weak areas
- `save` — checkpoint the scorecard to git

## Source of truth

The book (`Inference Engineering.pdf`) and the cross-referenced notes in
`../`. The `08-book-corrections.md` errata is tied to specific chapters/pages
and is the most reliable mapping to the book's framing.

## Files

- `question-bank.md` — categorized questions, easy → hard, with grading keys
- `scorecard.md` — running log of attempts, scores, and weak areas
