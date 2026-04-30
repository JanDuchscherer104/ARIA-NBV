#import "template/layout/thesis_template.typ": *
#import "metadata.typ": *
#import "../shared/macros.typ": *
#import "../shared/glossary.typ": *

#set document(title: titleEnglish, author: author)
#set text(font: "New Computer Modern")

#show: thesis.with(
  title: titleEnglish,
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
  abstract_en: [
    This thesis investigates quality-driven #NBV planning for egocentric 3D reconstruction in #ASE. It builds on the ARIA-NBV seminar paper and frames Relative Reconstruction Improvement (#RRI) as the central supervision and evaluation signal for choosing informative future views.
  ],
  abstract_de: [
    TODO: Deutsche Zusammenfassung ergaenzen.
  ],
  acknowledgement: [
    TODO: Acknowledgements.
  ],
  transparency_ai_tools: [
    TODO: Document AI tools used during thesis writing and implementation according to the final institutional requirements.
  ],
)

#include "sections/01-introduction.typ"
#include "sections/02-background.typ"
#include "sections/03-method.typ"
#include "sections/04-evaluation.typ"
#include "sections/05-conclusion.typ"
