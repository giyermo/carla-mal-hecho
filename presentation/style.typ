// Shared style for the RL-for-CARLA handouts.

#let ink = rgb("#1E2B26")
#let muted = rgb("#5E6963")
#let accent = rgb("#A8481F")
#let teal = rgb("#2F6B6E")

#let doc(title: "", subtitle: "", body) = {
  set document(title: title, author: "[Your name]")
  set page(
    paper: "a4",
    margin: (x: 2.2cm, top: 2.4cm, bottom: 2.4cm),
    footer: context [
      #set text(size: 9pt, fill: muted)
      #title #h(1fr) #counter(page).display("1 / 1", both: true)
    ],
  )
  set text(font: "Libertinus Serif", size: 11pt, fill: ink, lang: "en")
  set par(justify: true, leading: 0.65em)
  set list(indent: 0.6em, spacing: 0.7em)
  set enum(indent: 0.6em, spacing: 0.7em)
  set table(stroke: 0.5pt + rgb("#C9C2B1"), inset: 6pt)
  show table.cell.where(y: 0): set text(weight: "bold")
  show link: set text(fill: teal)
  show heading.where(level: 1): it => block(above: 1.8em, below: 0.9em, stack(
    spacing: 0.5em,
    text(size: 16pt, weight: "bold", fill: ink, it.body),
    line(length: 100%, stroke: 0.6pt + accent),
  ))
  show heading.where(level: 2): it => block(above: 1.3em, below: 0.6em,
    text(size: 12.5pt, weight: "bold", fill: teal, it.body))
  show table: set par(justify: false)
  show table: set block(breakable: false)

  text(size: 9pt, weight: "bold", fill: accent, tracking: 1.5pt)[BACHELOR'S THESIS (TFG) · UC3M]
  v(0.3em)
  text(size: 22pt, weight: "bold")[#title]
  v(0.1em)
  text(size: 12pt, fill: muted)[#subtitle]
  v(0.2em)
  text(size: 10pt, fill: muted)[[Your name] · Supervisor: [name] · September 2026]
  v(1em)

  body
}

#let caveat(body) = block(
  fill: rgb("#F6E4D9"), inset: 9pt, radius: 4pt, width: 100%,
)[*Caveat:* #body]

#let proposal(body) = block(
  fill: rgb("#E3EEEE"), inset: 9pt, radius: 4pt, width: 100%,
)[*My proposal (not from the papers):* #body]

#let note(body) = text(size: 9.5pt, fill: muted, body)
