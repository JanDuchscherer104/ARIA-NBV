#import "titlepage.typ": *
#import "transparency_ai_tools.typ": transparency_ai_tools as transparency_ai_tools_layout
#import "fonts.typ": *
#import "branding.typ": hm-colors
#import "../utils/print_page_break.typ": *

#let proposal(
  title: "",
  titleGerman: "",
  thesisKindEnglish: "Master's Thesis Proposal",
  thesisKindGerman: "Proposal zur Masterarbeit",
  academicDegree: "Master of Science (M.Sc.)",
  program: "",
  specialization: "",
  universityEnglish: "",
  universityGerman: "",
  facultyEnglish: "",
  facultyGerman: "",
  firstExaminer: "",
  secondExaminer: "",
  supervisors: (),
  author: "",
  email: "",
  matriculationNumber: "",
  startDate: datetime,
  submissionDate: datetime,
  submissionDateText: "",
  transparency_ai_tools: "",
  is_print: false,
  body,
) = {
  titlepage(
    title: title,
    titleGerman: titleGerman,
    thesisKindEnglish: thesisKindEnglish,
    thesisKindGerman: thesisKindGerman,
    academicDegree: academicDegree,
    program: program,
    specialization: specialization,
    universityEnglish: universityEnglish,
    universityGerman: universityGerman,
    facultyEnglish: facultyEnglish,
    facultyGerman: facultyGerman,
    firstExaminer: firstExaminer,
    secondExaminer: secondExaminer,
    supervisors: supervisors,
    author: author,
    email: email,
    matriculationNumber: matriculationNumber,
    startDate: startDate,
    submissionDate: submissionDate,
    submissionDateText: submissionDateText,
  )

  print_page_break(print: is_print, to: "even")
  transparency_ai_tools_layout(transparency_ai_tools)
  print_page_break(print: is_print)

  set page(
    margin: (left: 30mm, right: 30mm, top: 35mm, bottom: 35mm),
    numbering: "1",
    number-align: center,
  )

  set text(font: fonts.body, size: 12pt, lang: "en")
  set par(leading: 1em, justify: true)
  set cite(style: "alphanumeric")
  show math.equation: set text(weight: 400)
  show heading: set block(below: 0.85em, above: 1.75em)
  show heading: set text(font: fonts.body)
  set heading(numbering: "1.1")

  show outline.entry.where(level: 1): it => {
    v(10pt, weak: true)
    strong(it)
  }
  outline(
    title: {
      text(font: fonts.body, 1.5em, weight: 700, fill: hm-colors.blue, "Contents")
      v(12mm)
    },
    indent: 2em,
  )
  pagebreak()
  counter(page).update(1)

  body

  pagebreak()
  bibliography("/references.bib", style: "ieee")
}
