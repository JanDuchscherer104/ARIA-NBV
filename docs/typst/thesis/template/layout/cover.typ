#import "fonts.typ": *
#import "branding.typ": hm-brand, hm-colors

#let cover(
  title: "",
  degree: "",
  program: "",
  author: "",
) = {
  set page(
    margin: (left: 30mm, right: 30mm, top: 36mm, bottom: 36mm),
    numbering: none,
    number-align: center,
  )

  set text(font: fonts.body, size: 12pt, lang: "en")
  set par(leading: 1em)

  place(top + right, rect(width: 100%, height: 5mm, fill: hm-colors.blue))

  v(8mm)
  align(center, image(hm-brand.logo, width: 34%))

  v(10mm)
  align(center, text(font: fonts.sans, 2em, weight: 700, fill: hm-colors.blue, hm-brand.organization))

  v(4mm)
  align(center, text(font: fonts.sans, 1.25em, weight: 500, fill: hm-colors.gray, hm-brand.department))

  v(18mm)
  align(center, text(font: fonts.sans, 1.25em, weight: 500, degree + " Thesis in " + program))

  v(14mm)
  align(center, text(font: fonts.sans, 2em, weight: 700, fill: hm-colors.dark-blue, title))

  v(12mm)
  align(center, text(font: fonts.sans, 1.65em, weight: 500, author))
}
