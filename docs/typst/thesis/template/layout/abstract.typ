#import "fonts.typ": *
#import "branding.typ": hm-colors

#let abstract(body, lang: "en") = {
  let title = (en: "Abstract", de: "Zusammenfassung")

  set page(
    margin: (left: 30mm, right: 30mm, top: 40mm, bottom: 40mm),
    numbering: none,
    number-align: center,
  )

  set text(font: fonts.body, size: 12pt, lang: lang)
  set par(leading: 1em, justify: true)

  v(1fr)
  align(center, text(font: fonts.body, 1.15em, weight: "semibold", fill: hm-colors.blue, title.at(lang)))
  v(8mm)
  body
  v(1fr)
}
