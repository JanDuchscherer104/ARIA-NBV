
// Copyright 2024 Thomas Gingele https://github.com/B1TC0R3
// Copyright 2024 Felix Schladt https://github.com/FelixSchladt

#let note_inset = 1em
#let note_border_radius = 0.5em

#let note_info_border_color = black
#let note_info_background_color = gray.lighten(80%)
#let note_warning_border_color = red
#let note_warning_background_color = orange.lighten(80%)
#let note_good_border_color = green
#let note_good_background_color = lime.lighten(80%)

#let note-box(
  content,
  width: auto,
  background: note_info_background_color,
  border: note_info_border_color,
  bold: true,
) = {
  let weight = "light"
  if bold {
    weight = "semibold"
  }

  block(
    stroke: 1pt + border,
    fill: background,
    inset: note_inset,
    radius: note_border_radius,
    width: width,
  )[
    #set text(fill: black, weight: weight)
    #content
  ]
}

#let note(content, width: auto, background: note_info_background_color, border: note_info_border_color, bold: true) = {
  note-box(content, width: width, background: background, border: border, bold: bold)
}

#let warning-note(content, width: auto) = {
  note-box(
    content,
    width: width,
    background: note_warning_background_color,
    border: note_warning_border_color,
    bold: true,
  )
}

#let good-note(content, width: auto) = {
  note-box(
    content,
    width: width,
    background: note_good_background_color,
    border: note_good_border_color,
    bold: true,
  )
}

#let todo() = {
  set text(black)
  text(size: 120pt)[#emoji.chicken.baby #text(fill: gradient.linear(..color.map.rainbow))[TUDÜ]]
}
