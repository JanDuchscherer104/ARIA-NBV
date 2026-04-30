#import "fonts.typ": *
#import "branding.typ": hm-colors

#let acknowledgement(body) = {
  set page(
    margin: (left: 30mm, right: 30mm, top: 40mm, bottom: 40mm),
    numbering: none,
    number-align: center,
  )

  set text(font: fonts.body, size: 12pt, lang: "en")
  set par(leading: 1em, justify: true)

  align(left, text(font: fonts.sans, 20pt, weight: 700, fill: hm-colors.blue, "Acknowledgements"))
  v(12pt)
  body
}
